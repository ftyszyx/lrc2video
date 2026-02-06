"""
视频生成模块 - 支持多种前后帧生成视频的模型
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class BaseVideoGenerator(ABC):
    """视频生成器基类"""

    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass

    @abstractmethod
    def unload_model(self):
        """卸载模型"""
        pass

    @abstractmethod
    def generate(
        self, frame1: Image.Image, frame2: Image.Image, duration: float, fps: int = 24
    ) -> Path:
        """
        根据前后帧生成视频

        Args:
            frame1: 起始帧
            frame2: 结束帧
            duration: 视频时长（秒）
            fps: 帧率

        Returns:
            生成的视频路径
        """
        pass


class VideoGenerator:
    """视频生成器工厂类"""

    SUPPORTED_MODELS = ["tooncrafter", "dynamicrafter", "svd", "rife"]

    def __init__(self, model_type: str = None):
        """
        初始化视频生成器

        Args:
            model_type: 模型类型
        """
        self.model_type = model_type or config.DEFAULT_VIDEO_MODEL
        self.generator = None

    def _create_generator(self) -> BaseVideoGenerator:
        """创建对应的生成器实例"""
        if self.model_type == "tooncrafter":
            from models.tooncrafter_wrapper import ToonCrafterGenerator

            return ToonCrafterGenerator()
        elif self.model_type == "dynamicrafter":
            from models.dynamicrafter_wrapper import DynamiCrafterGenerator

            return DynamiCrafterGenerator()
        elif self.model_type == "svd":
            from models.svd_wrapper import SVDGenerator

            return SVDGenerator()
        elif self.model_type == "rife":
            from models.rife_wrapper import RIFEGenerator

            return RIFEGenerator()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def load_model(self):
        """加载模型"""
        if self.generator is None:
            self.generator = self._create_generator()
        self.generator.load_model()

    def unload_model(self):
        """卸载模型"""
        if self.generator is not None:
            self.generator.unload_model()

    def generate(
        self, frame1: Image.Image, frame2: Image.Image, duration: float, fps: int = 24
    ) -> Path:
        """生成视频片段"""
        if self.generator is None:
            self.load_model()
        return self.generator.generate(frame1, frame2, duration, fps)

    def generate_batch(
        self,
        image_paths: List[Path],
        durations: List[float],
        output_dir: str = None,
        fps: int = 24,
        progress_callback=None,
    ) -> List[Path]:
        """
        批量生成视频片段

        Args:
            image_paths: 图片路径列表
            durations: 每段视频的时长列表
            output_dir: 输出目录
            fps: 帧率
            progress_callback: 进度回调

        Returns:
            视频片段路径列表
        """
        output_dir = Path(output_dir or config.CACHE_DIR / "videos")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.load_model()

        video_paths = []

        # 相邻图片两两生成视频
        for i in range(len(image_paths) - 1):
            frame1 = Image.open(image_paths[i])
            frame2 = Image.open(image_paths[i + 1])
            duration = durations[i] if i < len(durations) else 3.0

            video_path = self.generator.generate(
                frame1=frame1, frame2=frame2, duration=duration, fps=fps
            )

            # 移动到输出目录
            final_path = output_dir / f"segment_{i:04d}.mp4"
            if video_path != final_path:
                import shutil

                shutil.move(str(video_path), str(final_path))
            video_paths.append(final_path)

            if progress_callback:
                progress_callback(i + 1, len(image_paths) - 1)

        return video_paths
