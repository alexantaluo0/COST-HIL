"""
TensorBoard 日志记录工具（wandb 的替代方案）
本地运行，无需网络连接
"""
import logging
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Any
from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig


class TensorBoardLogger:
    """TensorBoard 日志记录器（wandb 的替代方案，接口与 WandBLogger 兼容）"""

    def __init__(self, cfg: TrainPipelineConfig):
        """
        初始化 TensorBoard 日志记录器

        参数:
            cfg: 训练配置对象（与 WandBLogger 接口保持一致）
        """
        self.cfg = cfg.wandb if hasattr(cfg, 'wandb') else None
        self.log_dir = Path(cfg.output_dir)
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None
        
        # 创建 tensorboard 日志目录
        tensorboard_dir = self.log_dir / "tensorboard"
        if self.job_name:
            tensorboard_dir = tensorboard_dir / self.job_name
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建 TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
        self._step = 0
        self._custom_step_keys: dict[str, int] = {}  # 存储每个 custom_step_key 的当前步数
        
        logging.info(colored("TensorBoard 日志记录已初始化", "blue", attrs=["bold"]))
        logging.info(f"TensorBoard 日志目录: {tensorboard_dir}")
        logging.info(f"启动 TensorBoard: tensorboard --logdir={tensorboard_dir.parent}")
        logging.info(f"在浏览器中打开: http://localhost:6006")

    def log_policy(self, checkpoint_dir: Path):
        """记录策略检查点（TensorBoard 不直接支持，仅记录日志）"""
        logging.info(f"策略检查点已保存: {checkpoint_dir}")
        # TensorBoard 不直接支持 artifact，但可以通过记录路径来追踪

    def log_dict(
        self, 
        d: dict[str, Any], 
        step: int | None = None, 
        mode: str = "train",
        custom_step_key: str | None = None
    ):
        """
        记录字典数据（与 WandBLogger 接口兼容）

        参数:
            d: 要记录的字典
            step: 步数（如果提供了 custom_step_key，则忽略此参数）
            mode: 模式（train/eval/expert/expert_online等）
            custom_step_key: 自定义步数键名（从字典中提取步数值）
        """
        if mode not in {"train", "eval", "pretrain", "expert_online", "expert"}:
            logging.warning(f"未知的模式: {mode}，使用默认模式 'train'")
            mode = "train"
        
        if step is None and custom_step_key is None:
            logging.warning("未提供 step 或 custom_step_key，使用内部计数器")
            step = self._step
            self._step += 1
        
        # 如果提供了 custom_step_key，从字典中提取步数值
        if custom_step_key is not None:
            if custom_step_key not in d:
                logging.warning(f"字典中未找到 custom_step_key: {custom_step_key}")
                step = self._step
                self._step += 1
            else:
                step = int(d[custom_step_key])
                # 更新该 custom_step_key 的步数记录
                if custom_step_key not in self._custom_step_keys:
                    self._custom_step_keys[custom_step_key] = step
                else:
                    self._custom_step_keys[custom_step_key] = max(
                        self._custom_step_keys[custom_step_key], step
                    )

        # 记录所有数值类型的键值对
        for k, v in d.items():
            # 跳过 custom_step_key 本身（避免重复记录）
            if k == custom_step_key:
                continue
                
            if not isinstance(v, (int, float)):
                if isinstance(v, str):
                    # TensorBoard 不支持字符串，记录为文本摘要
                    self.writer.add_text(f"{mode}/{k}", str(v), step)
                else:
                    logging.warning(
                        f'TensorBoard 日志记录中键 "{k}" 的类型 "{type(v)}" 不被支持，已跳过。'
                    )
                continue
            
            # 添加模式前缀
            tag = f"{mode}/{k}"
            self.writer.add_scalar(tag, float(v), step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        """
        记录视频

        参数:
            video_path: 视频文件路径
            step: 步数
            mode: 模式（train/eval）
        """
        if mode not in {"train", "eval"}:
            logging.warning(f"视频记录不支持模式: {mode}，使用 'train'")
            mode = "train"
        
        tag = f"{mode}/video"
        # TensorBoard 的 add_video 需要视频张量，这里记录路径作为文本
        # 如果需要实际显示视频，需要先加载视频文件
        try:
            import torchvision.io as io
            video_tensor, _, _ = io.read_video(video_path, output_format="TCHW")
            fps = self.env_fps if self.env_fps else 30
            self.writer.add_video(tag, video_tensor.unsqueeze(0), step, fps=fps)
        except Exception as e:
            logging.warning(f"无法加载视频 {video_path}: {e}，仅记录路径")
            self.writer.add_text(tag, video_path, step)

    def close(self):
        """关闭 writer"""
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
