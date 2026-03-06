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

from .gym_manipulator import make_robot_env


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


def transform_observation(obs, device="cpu"):
    """Transform environment observation to policy expected format."""
    import torch
    
    # If observation is already in the correct format, return as is
    if "observation.state" in obs:
        return obs
    
    # Transform gym_hil observation format to policy format
    transformed_obs = {}
    
    # Handle image observations
    if "pixels" in obs:
        pixels = obs["pixels"]
        # Assuming pixels is a dict with camera names as keys
        if isinstance(pixels, dict):
            for cam_name, img in pixels.items():
                # Convert numpy array to torch tensor if needed
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(img).float()
                # Ensure correct shape: (batch, C, H, W)
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                if img.shape[-1] == 3:  # (batch, H, W, C) -> (batch, C, H, W)
                    img = img.permute(0, 3, 1, 2)
                transformed_obs[f"observation.images.{cam_name}"] = img.to(device)
        else:
            # Single image case
            if not isinstance(pixels, torch.Tensor):
                pixels = torch.from_numpy(pixels).float()
            if pixels.ndim == 3:
                pixels = pixels.unsqueeze(0)
            if pixels.shape[-1] == 3:
                pixels = pixels.permute(0, 3, 1, 2)
            transformed_obs["observation.images.front"] = pixels.to(device)
    
    # Handle state observations
    if "agent_pos" in obs:
        agent_pos = obs["agent_pos"]
        if not isinstance(agent_pos, torch.Tensor):
            agent_pos = torch.from_numpy(agent_pos).float()
        if agent_pos.ndim == 1:
            agent_pos = agent_pos.unsqueeze(0)
        transformed_obs["observation.state"] = agent_pos.to(device)
    
    return transformed_obs


def eval_policy(env, policy, n_episodes, device="cpu"):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting evaluation for {n_episodes} episodes...")
    sum_reward_episode = []
    
    for episode_idx in range(n_episodes):
        logger.info(f"Episode {episode_idx + 1}/{n_episodes} starting...")
        obs, _ = env.reset()
        obs = transform_observation(obs, device=device)
        episode_reward = 0.0
        step_count = 0
        
        while True:
            action = policy.select_action(obs, deterministic=True)
            # Remove batch dimension and convert to numpy
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            obs = transform_observation(obs, device=device)
            episode_reward += reward
            step_count += 1
            if terminated or truncated:
                break
                
        sum_reward_episode.append(episode_reward)
        logger.info(f"Episode {episode_idx + 1} finished - Reward: {episode_reward:.4f}, Steps: {step_count}")

    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info(f"Episode rewards: {[f'{r:.4f}' for r in sum_reward_episode]}")
    avg_reward = sum(sum_reward_episode) / len(sum_reward_episode)
    logger.info(f"Average reward: {avg_reward:.4f}")
    logger.info(f"Success rate: {sum([1 for r in sum_reward_episode if r > 0]) / len(sum_reward_episode) * 100:.2f}%")
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
    env, _ = make_robot_env(env_cfg)
    logger.info(f"Environment created: {env_cfg.name} - {env_cfg.task}")
    
    logger.info("Loading dataset...")
    dataset_cfg = cfg.dataset
    dataset = LeRobotDataset(repo_id=dataset_cfg.repo_id, root=dataset_cfg.root if hasattr(dataset_cfg, 'root') else None)
    dataset_meta = dataset.meta
    logger.info(f"Dataset loaded: {dataset_cfg.repo_id}")

    logger.info("Creating policy...")
    policy = make_policy(
        cfg=cfg.policy,
        # env_cfg=cfg.env,
        ds_meta=dataset_meta,
    )
    policy.eval()
    logger.info(f"Policy loaded from: {cfg.policy.pretrained_path}")
    logger.info(f"Policy device: {cfg.policy.device}")

    logger.info("Starting policy evaluation...")
    n_episodes = cfg.eval.n_episodes if hasattr(cfg, 'eval') and hasattr(cfg.eval, 'n_episodes') else 10
    eval_policy(env, policy=policy, n_episodes=n_episodes, device=cfg.policy.device)
    
    logger.info("=" * 60)
    logger.info("Evaluation job finished successfully!")
    logger.info(f"Log saved to: {log_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
