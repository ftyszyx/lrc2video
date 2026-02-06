"""
FLUX.1 模型封装
"""

import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class FluxWrapper:
    """FLUX.1 模型封装类"""

    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or config.FLUX_MODEL_PATH
        self.device = device or config.FLUX_DEVICE
        self.pipe = None

    def load(self):
        """加载模型"""
        if self.pipe is not None:
            return

        from diffusers import FluxPipeline

        print(f"正在加载 FLUX.1 模型: {self.model_path}")

        dtype = torch.float16 if config.FLUX_DTYPE == "float16" else torch.bfloat16

        self.pipe = FluxPipeline.from_pretrained(self.model_path, torch_dtype=dtype)
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()

        print("FLUX.1 模型加载完成")

    def unload(self):
        """卸载模型"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            print("FLUX.1 模型已卸载")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = 1080,
        height: int = 1920,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """生成图片"""
        self.load()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or config.DEFAULT_NEGATIVE_PROMPT,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        return image
