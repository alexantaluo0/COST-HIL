#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Actor server runner for distributed HILSerl robot policy training.

This script implements the actor component of the distributed HILSerl architecture.
It executes the policy in the robot environment, collects experience,
and sends transitions to the learner server for policy updates.

Examples of usage:

- Start an actor server for real robot training with human-in-the-loop intervention:
```bash
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so100.json
```

**NOTE**: The actor server requires a running learner server to connect to. Ensure the learner
server is started before launching the actor.

**NOTE**: Human intervention is key to HILSerl training. Press the upper right trigger button on the
gamepad to take control of the robot during training. Initially intervene frequently, then gradually
reduce interventions as the policy improves.

**WORKFLOW**:
1. Determine robot workspace bounds using `lerobot-find-joint-limits`
2. Record demonstrations with `gym_manipulator.py` in record mode
3. Process the dataset and determine camera crops with `crop_dataset_roi.py`
4. Start the learner server with the training configuration
5. Start this actor server with the same configuration
6. Use human interventions to guide policy learning

For more details on the complete HILSerl training workflow, see:
https://github.com/michel-aractingi/lerobot-hilserl-guide
"""

import logging
import os
import time
from functools import lru_cache
from queue import Empty

import numpy as np
import grpc
import torch
from torch import nn
from torch.multiprocessing import Event, Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.processor import TransitionKey
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

from .gym_manipulator import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)
from .hypothesis1 import (
    InterventionScheduler,
    InterventionUIPrompt,
    estimate_actor_entropy_uncertainty,
    estimate_value_no_intervention,
)
from lerobot.utils.constants import OBS_IMAGES

# When OpenCV has no GUI support, use matplotlib for real-time display.
_display_camera_backend: str | None = None  # "cv2" | "mpl" | None
_display_camera_mpl_axes: dict = {}


def _display_camera_feed(transition, cfg: TrainRLServerPipelineConfig) -> None:
    """Display camera feeds in real-time during training when display_cameras is enabled.

    Shows processed (cropped/resized) images from observation.
    Uses cv2 when available, falls back to matplotlib when OpenCV has no GUI support.
    """
    global _display_camera_backend, _display_camera_mpl_axes
    obs_config = getattr(getattr(cfg.env.processor, "observation", None), "display_cameras", False)
    if not obs_config:
        return
    import cv2

    def _to_rgb_uint8(img_np):
        img_np = np.asarray(img_np)
        if img_np.dtype.kind == "f":
            img_np = (img_np * 255).clip(0, 255).astype("uint8")
        elif img_np.dtype != "uint8":
            img_np = img_np.astype("uint8")
        return img_np

    def _show_cv2(win_name, img_rgb):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(win_name, img_bgr)

    def _show_mpl(win_name, img_rgb):
        import matplotlib.pyplot as plt

        if win_name not in _display_camera_mpl_axes:
            fig, ax = plt.subplots()
            fig.canvas.manager.set_window_title(win_name)
            ax.axis("off")
            im = ax.imshow(img_rgb)
            _display_camera_mpl_axes[win_name] = (fig, ax, im)
            plt.ion()
            plt.show(block=False)
        else:
            _, ax, im = _display_camera_mpl_axes[win_name]
            im.set_data(img_rgb)
            ax.figure.canvas.draw()
        plt.pause(0.001)

    def _show(win_name, img_rgb):
        global _display_camera_backend
        if _display_camera_backend == "mpl":
            _show_mpl(win_name, img_rgb)
        else:
            try:
                _show_cv2(win_name, img_rgb)
                if _display_camera_backend is None:
                    _display_camera_backend = "cv2"
            except cv2.error:
                if _display_camera_backend is None:
                    _display_camera_backend = "mpl"
                    logging.warning(
                        "[ACTOR] OpenCV has no GUI support. Using matplotlib for camera display."
                    )
                _show_mpl(win_name, img_rgb)

    obs = transition.get(TransitionKey.OBSERVATION, {})
    for key in [k for k in obs if k.startswith(f"{OBS_IMAGES}.")]:
        value = obs[key]
        if not isinstance(value, torch.Tensor):
            continue
        img = value.detach().cpu()
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.ndim != 3:
            continue
        img_np = img.permute(1, 2, 0).numpy()
        img_rgb = _to_rgb_uint8(img_np)
        _show(key, img_rgb)

    if _display_camera_backend == "cv2":
        try:
            cv2.waitKey(1)
        except cv2.error:
            pass


# Main entry point


@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host,
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # If we use multithreading, we can reuse the channel
        grpc_channel.close()
        grpc_channel = None

    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue,
    )
    logging.info("[ACTOR] Policy process joined")

    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")


# Core algorithm functions


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
    parameters_queue: Queue,
    transitions_queue: Queue,
    interactions_queue: Queue,
):
    """
    Executes policy interaction within the environment.

    This function rolls out the policy in the environment, collecting interaction data and pushing it to a queue for streaming to the learner.
    Once an episode is completed, updated network parameters received from the learner are retrieved from a queue and loaded into the network.

    Args:
        cfg: Configuration settings for the interaction process.
        shutdown_event: Event to check if the process should shutdown.
        parameters_queue: Queue to receive updated network parameters from the learner.
        transitions_queue: Queue to send transitions to the learner.
        interactions_queue: Queue to send interactions to the learner.
    """
    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_policy_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor policy process logging initialized")

    logging.info("make_env online")

    online_env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    ### Instantiate the policy in both the actor and learner processes
    ### To avoid sending a SACPolicy object through the port, we create a policy instance
    ### on both sides, the learner sends the updated parameters every n steps to update the actor's parameters
    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy = policy.eval()
    assert isinstance(policy, nn.Module)

    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    list_transition_to_send_to_learner = []
    episode_intervention = False
    # Add counters for intervention rate calculation
    episode_intervention_steps = 0
    episode_total_steps = 0

    # Gripper anti-oscillation state (v0.1.5)
    gripper_locked_action = None
    gripper_cooldown_remaining = 0
    gripper_last_switch_step = -999

    # Gamepad RB toggle patch: single press to toggle intervention, instead of hold
    def _find_gamepad_controller(env):
        """Walk the gym wrapper chain to find GamepadController (has joystick, no key_states)."""
        current = env
        while current is not None:
            ctrl = getattr(current, "controller", None)
            if ctrl is not None and hasattr(ctrl, "joystick") and not hasattr(ctrl, "key_states"):
                return ctrl
            current = getattr(current, "env", None)
        return None

    _gamepad_controller = _find_gamepad_controller(online_env)
    if _gamepad_controller is not None:
        import pygame as _pygame

        # Flush stale events accumulated during pygame/joystick initialization
        _pygame.event.clear()

        def _patched_gamepad_update():
            _btns = _gamepad_controller.controller_config.get("buttons", {})
            _rb = _btns.get("rb", 5)
            _y  = _btns.get("y",  3)
            _a  = _btns.get("a",  0)
            _x  = _btns.get("x",  2)
            _lt = _btns.get("lt", 6)
            _rt = _btns.get("rt", 7)
            for _ev in _pygame.event.get():
                if _ev.type == _pygame.JOYBUTTONDOWN:
                    if _ev.button == _rb:
                        _gamepad_controller.intervention_flag = not _gamepad_controller.intervention_flag
                        _state = "人工干预" if _gamepad_controller.intervention_flag else "AI控制"
                        logging.info("[ACTOR] 手柄 RB 切换: %s", _state)
                    elif _ev.button == _y:
                        _gamepad_controller.episode_end_status = "success"
                    elif _ev.button == _a:
                        _gamepad_controller.episode_end_status = "failure"
                    elif _ev.button == _x:
                        _gamepad_controller.episode_end_status = "rerecord_episode"
                    elif _ev.button == _lt:
                        _gamepad_controller.close_gripper_command = True
                    elif _ev.button == _rt:
                        _gamepad_controller.open_gripper_command = True
                elif _ev.type == _pygame.JOYBUTTONUP:
                    if _ev.button in [_x, _a, _y]:
                        _gamepad_controller.episode_end_status = None
                    elif _ev.button == _lt:
                        _gamepad_controller.close_gripper_command = False
                    elif _ev.button == _rt:
                        _gamepad_controller.open_gripper_command = False

        def _patched_gamepad_reset():
            _gamepad_controller.intervention_flag = False
            _gamepad_controller.open_gripper_command = False
            _gamepad_controller.close_gripper_command = False
            _gamepad_controller.episode_end_status = None
            _pygame.event.clear()

        _gamepad_controller.update = _patched_gamepad_update
        _gamepad_controller.reset = _patched_gamepad_reset
        print("[ACTOR] 手柄已启用: 单按 RB 切换干预/AI控制模式")
        logging.info("[ACTOR] 手柄 RB 切换模式已激活")

    # Hypothesis 1: Adaptive intervention scheduler (separate module for ablation)
    intervention_scheduler = None
    intervention_ui_prompt = None
    if getattr(cfg, "intervention_scheduler", None) and cfg.intervention_scheduler.enabled:
        intervention_scheduler = InterventionScheduler(config=cfg.intervention_scheduler)
        sc = cfg.intervention_scheduler
        logging.info(
            "[ACTOR] Intervention scheduler enabled (uncertainty_threshold=%.2f, show_ui=%s)",
            sc.uncertainty_threshold_high,
            getattr(sc, "show_ui_prompt", True),
        )
        intervention_ui_prompt = InterventionUIPrompt(
            show_ui=getattr(sc, "show_ui_prompt", True),
            use_sound=getattr(sc, "use_sound_prompt", False),
        )

    policy_timer = TimerManager("Policy inference", log=False)

    # Auto intervention: use previous step's suggestion for next step
    suggested_intervention_prev = False
    auto_intervention_steps = 0
    is_intervention = False  # from prev step; used for force_use_teleop
    sc = getattr(cfg, "intervention_scheduler", None)
    use_auto = getattr(sc, "use_auto_intervention", False) if (sc and sc.enabled) else False
    max_auto_steps = getattr(sc, "max_auto_intervention_steps", 50) if sc else 50
    wait_for_intervention = getattr(sc, "wait_for_intervention_when_suggested", False) if (sc and sc.enabled) else False

    # Zero action for wait mode (actor pauses, env keeps stepping until Space or timeout)
    action_space = getattr(online_env, "action_space", None)
    if action_space is not None and hasattr(action_space, "shape") and len(action_space.shape) > 0:
        zero_action = torch.zeros(action_space.shape[0], dtype=torch.float32, device=device)
    else:
        zero_action = None  # fallback: will use policy action shape when first created

    try:
        for interaction_step in range(cfg.policy.online_steps):
            start_time = time.perf_counter()
            if shutdown_event.is_set():
                logging.info("[ACTOR] Shutting down act_with_policy")
                return

            observation = {
                k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
            }

            gripper_switch_penalty = 0.0

            # Wait mode: UI prompt appeared, actor pauses until user presses Space or episode ends.
            # If user never presses Space, episode timeout will reset. If user presses Space, use teleop.
            waiting_for_intervention = (
                wait_for_intervention and suggested_intervention_prev and not is_intervention
            )

            if waiting_for_intervention:
                # Actor paused: use zero action (hold), no policy inference
                action = zero_action if zero_action is not None else torch.zeros(4, dtype=torch.float32, device=device)
                force_use_teleop = False
            else:
                # Normal: policy inference
                with policy_timer:
                    action = policy.select_action(batch=observation)
                policy_fps = policy_timer.fps_last
                log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

                # Gripper anti-oscillation: cooldown + frequency penalty (v0.1.5)
                gc = getattr(cfg.policy, "gripper_control", None)
                if (
                    gc is not None
                    and getattr(gc, "enabled", False)
                    and cfg.policy.num_discrete_actions is not None
                ):
                    policy_gripper = int(round(action[0, -1].item()))
                    if gripper_cooldown_remaining > 0:
                        action[0, -1] = float(gripper_locked_action)
                        gripper_cooldown_remaining -= 1
                    else:
                        if gripper_locked_action is not None and policy_gripper != gripper_locked_action:
                            steps_since = episode_total_steps - gripper_last_switch_step
                            if steps_since < gc.penalty_window_steps:
                                gripper_switch_penalty = gc.switching_penalty
                            gripper_last_switch_step = episode_total_steps
                            gripper_locked_action = policy_gripper
                            gripper_cooldown_remaining = gc.cooldown_steps - 1
                        else:
                            if gripper_locked_action is None:
                                gripper_locked_action = policy_gripper

                # Human intervention: user pressed Space -> teleop (unified path).
                # Manual Space (no prompt) and UI-prompted Space both use same teleop flow.
                force_use_teleop = False
                if is_intervention:
                    force_use_teleop = True
                elif use_auto and suggested_intervention_prev and auto_intervention_steps < max_auto_steps:
                    force_use_teleop = True
                    auto_intervention_steps += 1

            # Use the new step function
            new_transition = step_env_and_process_transition(
                env=online_env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
                force_use_teleop=force_use_teleop,
            )

            # Extract values from processed transition
            next_observation = {
                k: v
                for k, v in new_transition[TransitionKey.OBSERVATION].items()
                if k in cfg.policy.input_features
            }

            # Teleop action is the action that was executed in the environment
            # It is either the action from the teleop device or the action from the policy
            executed_action = new_transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]

            reward = new_transition[TransitionKey.REWARD]
            done = new_transition.get(TransitionKey.DONE, False)
            truncated = new_transition.get(TransitionKey.TRUNCATED, False)

            sum_reward_episode += float(reward)
            episode_total_steps += 1

            # Check for intervention from transition info
            intervention_info = new_transition[TransitionKey.INFO]
            is_intervention = intervention_info.get(TeleopEvents.IS_INTERVENTION.value, False)
            if is_intervention:
                episode_intervention = True
                episode_intervention_steps += 1

            # Hypothesis 1: Compute suggested_intervention (skip when waiting - no policy run)
            suggested_intervention = False
            if intervention_scheduler is not None and not waiting_for_intervention:
                uncertainty_score = estimate_actor_entropy_uncertainty(policy, observation)
                value_estimate = estimate_value_no_intervention(policy, observation)
                suggested_intervention = intervention_scheduler.should_suggest_intervention(
                    uncertainty_score=uncertainty_score,
                    is_human_intervention=is_intervention,
                    value_estimate=value_estimate,
                    interaction_step=interaction_step,
                )

            # Step penalty to encourage fast task completion (v0.1.5)
            step_penalty_value = 0.0
            sp = getattr(cfg.policy, "step_penalty", None)
            if sp is not None and getattr(sp, "enabled", False):
                if episode_total_steps < sp.min_steps:
                    step_penalty_value = 0.0
                elif episode_total_steps <= sp.max_steps:
                    step_penalty_value = (
                        (episode_total_steps - sp.min_steps)
                        / (sp.max_steps - sp.min_steps)
                        * sp.max_penalty
                    )
                else:
                    step_penalty_value = sp.max_penalty

            base_discrete_penalty = new_transition[TransitionKey.COMPLEMENTARY_DATA].get(
                "discrete_penalty", 0.0
            )
            complementary_info = {
                "discrete_penalty": torch.tensor(
                    [base_discrete_penalty + gripper_switch_penalty], device=device
                ),
                "step_penalty": torch.tensor([step_penalty_value], device=device),
                TeleopEvents.IS_INTERVENTION.value: is_intervention,  # Add intervention flag to complementary_info
            }
            if intervention_scheduler is not None:
                complementary_info["suggested_intervention"] = suggested_intervention
            # Hypothesis 1: UI prompt when suggested_intervention (prompts user to press Space)
            if intervention_ui_prompt is not None:
                intervention_ui_prompt.update(suggested_intervention if not waiting_for_intervention else True)

            # Carry suggested_intervention for next step
            if waiting_for_intervention:
                if is_intervention:
                    suggested_intervention_prev = False  # User pressed Space, exit wait
                # else: keep suggested_intervention_prev True (still waiting)
            else:
                suggested_intervention_prev = suggested_intervention
            if is_intervention:
                auto_intervention_steps = 0  # Reset on human intervention

            # Hypothesis 2: Add intervention_quality for weighted loss (default 1.0, separate for ablation)
            if getattr(cfg, "weighted_intervention", None) and cfg.weighted_intervention.enabled:
                complementary_info["intervention_quality"] = torch.tensor([1.0])

            # Create transition for learner (skip when waiting - those steps use zero action, not policy)
            if not waiting_for_intervention:
                list_transition_to_send_to_learner.append(
                    Transition(
                        state=observation,
                        action=executed_action,
                        reward=reward,
                        next_state=next_observation,
                        done=done,
                        truncated=truncated,
                        complementary_info=complementary_info,
                    )
                )

            # Update transition for next iteration
            transition = new_transition

            # 实时显示摄像头画面（若启用）
            _display_camera_feed(transition, cfg)

            if done or truncated:
                logging.info(
                    f"[ACTOR] Global step {interaction_step}: Episode reward: {sum_reward_episode}, Episode steps: {episode_total_steps}"
                )

                update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)

                # 未及时干预: 提示干预后，用户未按 Space（键盘）或 RB（罗技手柄）即回合结束
                # 此类 episode 的数据不存储到 buffer，也不参与 episodic reward、intervention rate 等任何指标计算
                episode_untimely_intervention = (
                    intervention_scheduler is not None
                    and suggested_intervention_prev
                    and not episode_intervention
                )

                if not episode_untimely_intervention and len(list_transition_to_send_to_learner) > 0:
                    push_transitions_to_transport_queue(
                        transitions=list_transition_to_send_to_learner,
                        transitions_queue=transitions_queue,
                    )
                elif episode_untimely_intervention and len(list_transition_to_send_to_learner) > 0:
                    logging.info(
                        "[ACTOR] Episode untimely intervention: discarding %d transitions (not stored to buffer)",
                        len(list_transition_to_send_to_learner),
                    )
                list_transition_to_send_to_learner = []

                stats = get_frequency_stats(policy_timer)
                policy_timer.reset()

                # Calculate intervention rate (only for timely episodes)
                intervention_rate = 0.0
                if episode_total_steps > 0:
                    intervention_rate = episode_intervention_steps / episode_total_steps

                # HIL-SERL 论文指标：成功率、周期时间（秒）
                episode_success = 1 if sum_reward_episode > 0 else 0
                fps = cfg.env.fps if cfg.env and cfg.env.fps else 30
                episode_duration_sec = episode_total_steps / fps if fps > 0 else 0.0

                # Send episodic reward to the learner (include flag for learner to filter metrics)
                interactions_queue.put(
                    python_object_to_bytes(
                        {
                            "Episodic reward": sum_reward_episode,
                            "Interaction step": interaction_step,
                            "Episode intervention": int(episode_intervention),
                            "Intervention rate": intervention_rate,
                            "episode_untimely_intervention": episode_untimely_intervention,
                            "Episode success": episode_success,
                            "Episode duration (s)": episode_duration_sec,
                            **stats,
                        }
                    )
                )

                # Reset intervention counters and environment
                sum_reward_episode = 0.0
                episode_intervention = False
                episode_intervention_steps = 0
                episode_total_steps = 0
                # Reset gripper anti-oscillation state (v0.1.5)
                gripper_locked_action = None
                gripper_cooldown_remaining = 0
                gripper_last_switch_step = -999
                if intervention_scheduler is not None:
                    intervention_scheduler.reset()
                if intervention_ui_prompt is not None:
                    intervention_ui_prompt.update(False)  # Hide prompt on episode end
                suggested_intervention_prev = False
                auto_intervention_steps = 0

                # Wait before starting next episode (reset_time_s)
                reset_time_s = None
                if (
                    cfg.env.processor is not None
                    and cfg.env.processor.reset is not None
                ):
                    reset_time_s = getattr(cfg.env.processor.reset, "reset_time_s", None)
                if reset_time_s is not None and reset_time_s > 0:
                    logging.info(f"[ACTOR] Waiting {reset_time_s}s before next episode...")
                    precise_sleep(reset_time_s)

                # Reset environment and processors
                obs, info = online_env.reset()
                env_processor.reset()
                action_processor.reset()

                # Process initial observation
                transition = create_transition(observation=obs, info=info)
                transition = env_processor(transition)

            if cfg.env.fps is not None:
                dt_time = time.perf_counter() - start_time
                precise_sleep(1 / cfg.env.fps - dt_time)
    finally:
        if intervention_ui_prompt is not None:
            intervention_ui_prompt.close()
        if getattr(getattr(cfg.env.processor, "observation", None), "display_cameras", False):
            if _display_camera_backend == "mpl" and _display_camera_mpl_axes:
                import matplotlib.pyplot as plt

                for _fig, _ax, _ in _display_camera_mpl_axes.values():
                    plt.close(_fig)
            else:
                try:
                    import cv2

                    cv2.destroyAllWindows()
                except Exception:
                    pass


#  Communication Functions - Group all gRPC/messaging functions


def establish_learner_connection(
    stub: services_pb2_grpc.LearnerServiceStub,
    shutdown_event: Event,  # type: ignore
    attempts: int = 30,
):
    """Establish a connection with the learner.

    Args:
        stub (services_pb2_grpc.LearnerServiceStub): The stub to use for the connection.
        shutdown_event (Event): The event to check if the connection should be established.
        attempts (int): The number of attempts to establish the connection.
    Returns:
        bool: True if the connection is established, False otherwise.
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # Force a connection attempt and check state
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False


@lru_cache(maxsize=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    """
    Returns a client for the learner service.

    GRPC uses HTTP/2, which is a binary protocol and multiplexes requests over a single connection.
    So we need to create only one client and reuse it.
    """

    channel = grpc.insecure_channel(
        f"{host}:{port}",
        grpc_channel_options(),
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
    cfg: TrainRLServerPipelineConfig,
    parameters_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
):
    """Receive parameters from the learner.

    Args:
        cfg (TrainRLServerPipelineConfig): The configuration for the actor.
        parameters_queue (Queue): The queue to receive the parameters.
        shutdown_event (Event): The event to check if the process should shutdown.
    """
    logging.info("[ACTOR] Start receiving parameters from the Learner")
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters",
        )

    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")


def send_transitions(
    cfg: TrainRLServerPipelineConfig,
    transitions_queue: Queue,
    shutdown_event: any,  # Event,
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends transitions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Transition Data:
        - A batch of transitions (observation, action, reward, next observation) is collected.
        - Transitions are moved to the CPU and serialized using PyTorch.
        - The serialized data is wrapped in a `services_pb2.Transition` message and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_transitions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor transitions process logging initialized")

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendTransitions(
            transitions_stream(
                shutdown_event, transitions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming transitions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Transitions process stopped")


def send_interactions(
    cfg: TrainRLServerPipelineConfig,
    interactions_queue: Queue,
    shutdown_event: Event,  # type: ignore
    learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
    grpc_channel: grpc.Channel | None = None,
) -> services_pb2.Empty:
    """
    Sends interactions to the learner.

    This function continuously retrieves messages from the queue and processes:

    - Interaction Messages:
        - Contains useful statistics about episodic rewards and policy timings.
        - The message is serialized using `pickle` and sent to the learner.
    """

    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_interactions_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor interactions process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )

    try:
        learner_client.SendInteractions(
            interactions_stream(
                shutdown_event, interactions_queue, cfg.policy.actor_learner_config.queue_get_timeout
            )
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    logging.info("[ACTOR] Finished streaming interactions")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Interactions process stopped")


def transitions_stream(shutdown_event: Event, transitions_queue: Queue, timeout: float) -> services_pb2.Empty:  # type: ignore
    while not shutdown_event.is_set():
        try:
            message = transitions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Transition queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message, services_pb2.Transition, log_prefix="[ACTOR] Send transitions"
        )

    return services_pb2.Empty()


def interactions_stream(
    shutdown_event: Event,
    interactions_queue: Queue,
    timeout: float,  # type: ignore
) -> services_pb2.Empty:
    while not shutdown_event.is_set():
        try:
            message = interactions_queue.get(block=True, timeout=timeout)
        except Empty:
            logging.debug("[ACTOR] Interaction queue is empty")
            continue

        yield from send_bytes_in_chunks(
            message,
            services_pb2.InteractionMessage,
            log_prefix="[ACTOR] Send interactions",
        )

    return services_pb2.Empty()


#  Policy functions


def update_policy_parameters(policy: SACPolicy, parameters_queue: Queue, device):
    bytes_state_dict = get_last_item_from_queue(parameters_queue, block=False)
    if bytes_state_dict is not None:
        logging.info("[ACTOR] Load new parameters from Learner.")
        state_dicts = bytes_to_state_dict(bytes_state_dict)

        # TODO: check encoder parameter synchronization possible issues:
        # 1. When shared_encoder=True, we're loading stale encoder params from actor's state_dict
        #    instead of the updated encoder params from critic (which is optimized separately)
        # 2. When freeze_vision_encoder=True, we waste bandwidth sending/loading frozen params
        # 3. Need to handle encoder params correctly for both actor and discrete_critic
        # Potential fixes:
        # - Send critic's encoder state when shared_encoder=True
        # - Skip encoder params entirely when freeze_vision_encoder=True
        # - Ensure discrete_critic gets correct encoder state (currently uses encoder_critic)

        # Load actor state dict
        actor_state_dict = move_state_dict_to_device(state_dicts["policy"], device=device)
        policy.actor.load_state_dict(actor_state_dict)

        # Load discrete critic if present
        if hasattr(policy, "discrete_critic") and "discrete_critic" in state_dicts:
            discrete_critic_state_dict = move_state_dict_to_device(
                state_dicts["discrete_critic"], device=device
            )
            policy.discrete_critic.load_state_dict(discrete_critic_state_dict)
            logging.info("[ACTOR] Loaded discrete critic parameters from Learner.")


#  Utilities functions


def push_transitions_to_transport_queue(transitions: list, transitions_queue):
    """Send transitions to learner in smaller chunks to avoid network issues.

    Args:
        transitions: List of transitions to send
        message_queue: Queue to send messages to learner
        chunk_size: Size of each chunk to send
    """
    transition_to_send_to_learner = []
    for transition in transitions:
        tr = move_transition_to_device(transition=transition, device="cpu")
        for key, value in tr["state"].items():
            if torch.isnan(value).any():
                logging.warning(f"Found NaN values in transition {key}")

        transition_to_send_to_learner.append(tr)

    transitions_queue.put(transitions_to_bytes(transition_to_send_to_learner))


def get_frequency_stats(timer: TimerManager) -> dict[str, float]:
    """Get the frequency statistics of the policy.

    Args:
        timer (TimerManager): The timer with collected metrics.

    Returns:
        dict[str, float]: The frequency statistics of the policy.
    """
    stats = {}
    if timer.count > 1:
        avg_fps = timer.fps_avg
        p90_fps = timer.fps_percentile(90)
        logging.debug(f"[ACTOR] Average policy frame rate: {avg_fps}")
        logging.debug(f"[ACTOR] Policy frame rate 90th percentile: {p90_fps}")
        stats = {
            "Policy frequency [Hz]": avg_fps,
            "Policy frequency 90th-p [Hz]": p90_fps,
        }
    return stats


def log_policy_frequency_issue(policy_fps: float, cfg: TrainRLServerPipelineConfig, interaction_step: int):
    if policy_fps < cfg.env.fps:
        logging.warning(
            f"[ACTOR] Policy FPS {policy_fps:.1f} below required {cfg.env.fps} at step {interaction_step}"
        )


def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"


if __name__ == "__main__":
    actor_cli()
