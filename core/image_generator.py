"""
FLUX.1 文生图模块
"""

import sys
from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image
import config

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class ImageGenerator:
    """FLUX.1 文生图生成器"""

    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化图像生成器

        Args:
            model_path: 模型路径
            device: 运行设备
        """
        self.model_path = model_path or config.FLUX_MODEL_PATH
        self.device = device or config.FLUX_DEVICE
        self.pipe = None

    def load_model(self):
        """加载 FLUX.1 模型"""
        if self.pipe is not None:
            return

        from diffusers import FluxPipeline

        print(f"正在加载 FLUX.1 模型: {self.model_path}")

        self.pipe = FluxPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if config.FLUX_DTYPE == "float16" else torch.bfloat16,
            token=config.HF_TOKEN if getattr(config, "HF_TOKEN", "") else None,
        )
        self.pipe = self.pipe.to(self.device)

        # 启用内存优化
        self.pipe.enable_model_cpu_offload()

        print("FLUX.1 模型加载完成")

    def unload_model(self):
        """卸载模型释放显存"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
            print("FLUX.1 模型已卸载")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = None,
        height: int = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = None,
    ) -> Image.Image:
        """
        生成单张图片

        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            width: 图片宽度
            height: 图片高度
            num_inference_steps: 推理步数
            guidance_scale: 引导系数
            seed: 随机种子

        Returns:
            生成的 PIL Image
        """
        self.load_model()

        width = width or config.DEFAULT_VIDEO_WIDTH
        height = height or config.DEFAULT_VIDEO_HEIGHT
        negative_prompt = negative_prompt or config.DEFAULT_NEGATIVE_PROMPT

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        return image

    def generate_batch(
        self,
        prompts: List[str],
        output_dir: str = None,
        negative_prompt: str = None,
        width: int = None,
        height: int = None,
        base_seed: int = None,
        progress_callback=None,
    ) -> List[Path]:
        """
        批量生成图片

        Args:
            prompts: 提示词列表
            output_dir: 输出目录
            negative_prompt: 负向提示词
            width: 图片宽度
            height: 图片高度
            base_seed: 基础种子（每张图片会递增）
            progress_callback: 进度回调 (current, total)

        Returns:
            生成的图片路径列表
        """
        output_dir = Path(output_dir or config.CACHE_DIR / "images")
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []

        for i, prompt in enumerate(prompts):
            seed = base_seed + i if base_seed is not None else None

            image = self.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
            )

            # 保存图片
            image_path = output_dir / f"frame_{i:04d}.png"
            image.save(image_path)
            image_paths.append(image_path)

            if progress_callback:
                progress_callback(i + 1, len(prompts))

        return image_paths
