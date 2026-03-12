# !/usr/bin/env python

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
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    so101_leader,  # noqa: F401
)

from lerobot.processor import TransitionKey

from .gym_manipulator import create_transition, make_processors, make_robot_env


def setup_logging(job_name="eval_default", log_dir="logs"):
    """Setup logging system to save logs to file and console.

    Args:
        job_name: Name of the job/project for the log file
        log_dir: Directory to save log files

    Returns:
        Path to the created log file
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_path / f"{job_name}_{timestamp}.log"

    # Configure logging
    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Configure root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)

    logging.info(f"Logging initialized. Log file: {log_filename}")

    return log_filename


def eval_policy(env, env_processor, policy, n_episodes, input_features, policy_cfg=None, device="cpu"):
    """Evaluate policy using the same observation processing pipeline as training.

    Args:
        env: Gym environment instance.
        env_processor: The same DataProcessorPipeline used in training actor to ensure
            consistent observation processing (image normalization, format conversion, etc.).
        policy: The SAC policy to evaluate.
        n_episodes: Number of evaluation episodes.
        input_features: Dict of policy input feature names to filter observations.
        policy_cfg: Policy configuration, used for gripper anti-oscillation control.
        device: Device for policy inference.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting evaluation for {n_episodes} episodes...")
    logger.info(f"Using deterministic action selection (mode instead of rsample)")

    # Gripper anti-oscillation config (same as training actor)
    gc = getattr(policy_cfg, "gripper_control", None)
    use_gripper_control = (
        gc is not None
        and getattr(gc, "enabled", False)
        and getattr(policy_cfg, "num_discrete_actions", None) is not None
    )
    if use_gripper_control:
        logger.info(f"Gripper anti-oscillation enabled: cooldown_steps={gc.cooldown_steps}")

    sum_reward_episode = []
    steps_per_episode = []

    for episode_idx in range(n_episodes):
        logger.info(f"Episode {episode_idx + 1}/{n_episodes} starting...")
        obs, info = env.reset()
        env_processor.reset()

        # Use the same processing pipeline as training actor
        transition = create_transition(observation=obs, info=info)
        transition = env_processor(transition)

        episode_reward = 0.0
        step_count = 0

        # Gripper anti-oscillation state (reset each episode, same as training actor)
        gripper_locked_action = None
        gripper_cooldown_remaining = 0

        while True:
            # Extract observation from transition (same as training actor)
            observation = {
                k: v for k, v in transition[TransitionKey.OBSERVATION].items()
                if k in input_features
            }

            # Use deterministic action selection for evaluation
            action = policy.select_action(observation, deterministic=True)

            # Gripper anti-oscillation: after a gripper state change, lock for cooldown_steps
            if use_gripper_control:
                policy_gripper = int(round(action[0, -1].item()))
                if gripper_cooldown_remaining > 0:
                    action[0, -1] = float(gripper_locked_action)
                    gripper_cooldown_remaining -= 1
                else:
                    if gripper_locked_action is not None and policy_gripper != gripper_locked_action:
                        gripper_locked_action = policy_gripper
                        gripper_cooldown_remaining = gc.cooldown_steps - 1
                    elif gripper_locked_action is None:
                        gripper_locked_action = policy_gripper

            # Remove batch dimension and convert to numpy
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)

            # Process observation through the same pipeline as training
            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

            episode_reward += reward
            step_count += 1
            if terminated or truncated:
                break

        sum_reward_episode.append(episode_reward)
        steps_per_episode.append(step_count)
        logger.info(f"Episode {episode_idx + 1} finished - Reward: {episode_reward:.4f}, Steps: {step_count}")

    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"Episode rewards: {[f'{r:.4f}' for r in sum_reward_episode]}")
    avg_reward = sum(sum_reward_episode) / len(sum_reward_episode)
    logger.info(f"Average reward: {avg_reward:.4f}")
    success_count = sum(1 for r in sum_reward_episode if r > 0)
    logger.info(f"Success rate: {success_count / len(sum_reward_episode) * 100:.2f}%")
    avg_steps = sum(steps_per_episode) / len(steps_per_episode)
    min_steps = min(steps_per_episode)
    max_steps = max(steps_per_episode)
    logger.info(f"Average steps per episode: {avg_steps:.1f} (min: {min_steps}, max: {max_steps})")
    logger.info("=" * 60)


@parser.wrap()
def main(cfg: TrainRLServerPipelineConfig):
    # Setup logging system
    log_file = setup_logging(job_name=cfg.job_name, log_dir="logs")
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"Starting evaluation job: {cfg.job_name}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info("=" * 60)

    logger.info("Creating environment...")
    env_cfg = cfg.env
    env, teleop_device = make_robot_env(env_cfg)
    logger.info(f"Environment created: {env_cfg.name} - {env_cfg.task}")

    # Create the same observation processing pipeline as training actor
    # This ensures consistent image normalization ([0,255] -> [0,1]),
    # format conversion (HWC -> CHW), batch dimension, and device placement.
    logger.info("Creating observation processing pipeline (same as training)...")
    env_processor, _ = make_processors(env, teleop_device, env_cfg, cfg.policy.device)
    logger.info("Observation pipeline created successfully")

    logger.info("Loading dataset...")
    dataset_cfg = cfg.dataset
    dataset = LeRobotDataset(
        repo_id=dataset_cfg.repo_id,
        root=dataset_cfg.root if hasattr(dataset_cfg, 'root') else None,
    )
    dataset_meta = dataset.meta
    logger.info(f"Dataset loaded: {dataset_cfg.repo_id}")

    logger.info("Creating policy...")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset_meta,
    )
    policy.eval()
    logger.info(f"Policy loaded from: {cfg.policy.pretrained_path}")
    logger.info(f"Policy device: {cfg.policy.device}")

    logger.info("Starting policy evaluation...")
    n_episodes = cfg.eval.n_episodes if hasattr(cfg, 'eval') and hasattr(cfg.eval, 'n_episodes') else 10
    eval_policy(
        env,
        env_processor=env_processor,
        policy=policy,
        n_episodes=n_episodes,
        input_features=cfg.policy.input_features,
        policy_cfg=cfg.policy,
        device=cfg.policy.device,
    )

    logger.info("=" * 60)
    logger.info("Evaluation job finished successfully!")
    logger.info(f"Log saved to: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
