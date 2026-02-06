"""
DynamiCrafter 模型封装 - 通用前后帧视频生成
"""

import sys
import tempfile
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from core.video_generator import BaseVideoGenerator


class DynamiCrafterGenerator(BaseVideoGenerator):
    """DynamiCrafter 视频生成器"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.DYNAMICRAFTER_MODEL_PATH
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载 DynamiCrafter 模型"""
        if self.model is not None:
            return

        try:
            # DynamiCrafter 需要从官方仓库加载
            # https://github.com/Doubiiu/DynamiCrafter
            print("正在加载 DynamiCrafter 模型...")

            # 实际使用时需要根据 DynamiCrafter 的具体 API 进行调整
            # from dynamicrafter import DynamiCrafterPipeline
            # self.model = DynamiCrafterPipeline.from_pretrained(
            #     self.model_path,
            #     torch_dtype=torch.float16
            # ).to(self.device)

            # 临时占位
            self.model = "placeholder"
            print("DynamiCrafter 模型加载完成（占位模式）")
            print("注意：请安装 DynamiCrafter 以获得最佳效果")

        except ImportError as e:
            print(f"DynamiCrafter 未安装: {e}")
            print("请参考 https://github.com/Doubiiu/DynamiCrafter 安装")
            self.model = "fallback"

    def unload_model(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()

    def generate(
        self, frame1: Image.Image, frame2: Image.Image, duration: float, fps: int = 24
    ) -> Path:
        """
        根据前后帧生成视频
        """
        self.load_model()

        num_frames = int(duration * fps)

        if self.model not in ["placeholder", "fallback"]:
            # DynamiCrafter 支持首尾帧插值
            video_frames = self.model.generate(
                start_image=frame1, end_image=frame2, num_frames=num_frames, fps=fps
            )
        else:
            # 后备方案：简单的线性插值
            video_frames = self._simple_interpolation(frame1, frame2, num_frames)

        # 保存为视频
        output_path = Path(tempfile.mktemp(suffix=".mp4"))
        self._save_video(video_frames, output_path, fps)

        return output_path

    def _simple_interpolation(
        self, frame1: Image.Image, frame2: Image.Image, num_frames: int
    ) -> list:
        """简单的线性插值（后备方案）"""
        img1 = np.array(frame1).astype(np.float32)
        img2 = np.array(frame2).astype(np.float32)

        frames = []
        for i in range(num_frames):
            alpha = i / (num_frames - 1) if num_frames > 1 else 0
            blended = (1 - alpha) * img1 + alpha * img2
            frames.append(Image.fromarray(blended.astype(np.uint8)))

        return frames

    def _save_video(self, frames: list, output_path: Path, fps: int):
        """保存帧序列为视频"""
        if not frames:
            return

        first_frame = frames[0]
        if isinstance(first_frame, Image.Image):
            width, height = first_frame.size
        else:
            height, width = first_frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in frames:
            if isinstance(frame, Image.Image):
                frame_np = np.array(frame)
            else:
                frame_np = frame
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
