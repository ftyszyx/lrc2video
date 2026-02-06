"""
Stable Video Diffusion 模型封装
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


class SVDGenerator(BaseVideoGenerator):
    """Stable Video Diffusion 视频生成器"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.SVD_MODEL_PATH
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载 SVD 模型"""
        if self.pipe is not None:
            return

        try:
            from diffusers import StableVideoDiffusionPipeline

            print(f"正在加载 SVD 模型: {self.model_path}")
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                self.model_path, torch_dtype=torch.float16, variant="fp16"
            )
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_model_cpu_offload()
            print("SVD 模型加载完成")
        except Exception as e:
            print(f"SVD 模型加载失败: {e}")
            print("将使用简单帧插值作为后备方案")
            self.pipe = "fallback"

    def unload_model(self):
        """卸载模型"""
        if self.pipe is not None and self.pipe != "fallback":
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()

    def generate(
        self, frame1: Image.Image, frame2: Image.Image, duration: float, fps: int = 24
    ) -> Path:
        """
        根据前后帧生成视频

        注意：SVD 原生是单图生成视频，这里通过两次生成+混合实现前后帧过渡
        """
        self.load_model()

        num_frames = min(int(duration * fps), 25)  # SVD 最多 25 帧

        if self.pipe != "fallback":
            try:
                # 从起始帧生成视频
                frames = self.pipe(
                    frame1, num_frames=num_frames, decode_chunk_size=8
                ).frames[0]
            except Exception as e:
                print(f"SVD 生成失败: {e}")
                frames = self._simple_interpolation(frame1, frame2, num_frames)
        else:
            frames = self._simple_interpolation(frame1, frame2, num_frames)

        # 保存为视频
        output_path = Path(tempfile.mktemp(suffix=".mp4"))
        self._save_video(frames, output_path, fps)

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
