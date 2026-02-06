"""
RIFE 帧插值模型封装 - 快速帧插值方案
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


class RIFEGenerator(BaseVideoGenerator):
    """RIFE 帧插值生成器"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or config.RIFE_MODEL_PATH
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载 RIFE 模型"""
        if self.model is not None:
            return

        try:
            # RIFE 模型加载（需要从官方仓库安装）
            # https://github.com/hzwer/ECCV2022-RIFE
            print("正在加载 RIFE 模型...")

            # 实际使用时需要根据 RIFE 的具体 API 进行调整
            # from rife.RIFE_HDv3 import Model
            # self.model = Model()
            # self.model.load_model(self.model_path, -1)
            # self.model.eval()
            # self.model.device()

            # 临时占位
            self.model = "placeholder"
            print("RIFE 模型加载完成（占位模式）")
            print("注意：请安装 RIFE 以获得最佳效果")

        except ImportError as e:
            print(f"RIFE 未安装: {e}")
            print("请参考 https://github.com/hzwer/ECCV2022-RIFE 安装")
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
        使用 RIFE 进行帧插值生成视频

        Args:
            frame1: 起始帧
            frame2: 结束帧
            duration: 视频时长（秒）
            fps: 帧率

        Returns:
            生成的视频路径
        """
        self.load_model()

        # 计算需要的帧数
        num_frames = int(duration * fps)

        # 转换为 numpy 数组
        img1 = np.array(frame1)
        img2 = np.array(frame2)

        if self.model not in ["placeholder", "fallback"]:
            # 使用 RIFE 递归插值生成中间帧
            frames = self._rife_interpolate(img1, img2, num_frames)
        else:
            # 后备方案：简单线性插值
            frames = self._simple_interpolation(img1, img2, num_frames)

        # 保存为视频
        output_path = Path(tempfile.mktemp(suffix=".mp4"))
        self._save_video(frames, output_path, fps)

        return output_path

    def _rife_interpolate(
        self, img1: np.ndarray, img2: np.ndarray, num_frames: int
    ) -> list:
        """使用 RIFE 递归插值生成指定数量的帧"""
        frames = [img1]

        # 计算需要的插值次数（2^n >= num_frames）
        n = 0
        while (2**n) < num_frames:
            n += 1

        # 递归插值
        def interpolate(start, end, depth):
            if depth == 0:
                return []

            # 生成中间帧
            mid = self._make_inference(start, end)

            left = interpolate(start, mid, depth - 1)
            right = interpolate(mid, end, depth - 1)

            return left + [mid] + right

        middle_frames = interpolate(img1, img2, n)
        frames.extend(middle_frames)
        frames.append(img2)

        # 截取到目标帧数
        if len(frames) > num_frames:
            step = len(frames) / num_frames
            result = [frames[int(i * step)] for i in range(num_frames)]
        else:
            result = frames

        return result

    def _make_inference(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """使用 RIFE 生成中间帧"""
        # 转换为 tensor
        img1_t = torch.from_numpy(img1.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        img2_t = torch.from_numpy(img2.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        img1_t = img1_t.to(self.device)
        img2_t = img2_t.to(self.device)

        with torch.no_grad():
            mid = self.model.inference(img1_t, img2_t)

        mid = (mid[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
        return mid

    def _simple_interpolation(
        self, img1: np.ndarray, img2: np.ndarray, num_frames: int
    ) -> list:
        """简单的线性插值（后备方案）"""
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        frames = []
        for i in range(num_frames):
            alpha = i / (num_frames - 1) if num_frames > 1 else 0
            blended = (1 - alpha) * img1 + alpha * img2
            frames.append(blended.astype(np.uint8))

        return frames

    def _save_video(self, frames: list, output_path: Path, fps: int):
        """保存帧序列为视频"""
        if not frames:
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
