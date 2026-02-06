"""
lrc2video 配置文件
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ============== 路径配置 ==============
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"
TEMPLATES_DIR = BASE_DIR / "templates"

# 确保目录存在
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ============== LLM API 配置 ==============
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

LLM_PROVIDER = "openai"

# ============== FLUX.1 配置 ==============
FLUX_MODEL_PATH = os.getenv("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev")
FLUX_DEVICE = os.getenv("FLUX_DEVICE", "cuda")
FLUX_DTYPE = "float16"
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN", ""))

# ============== 视频生成模型配置 ==============
DEFAULT_VIDEO_MODEL = os.getenv("DEFAULT_VIDEO_MODEL", "tooncrafter")
TOONCRAFTER_MODEL_PATH = os.getenv("TOONCRAFTER_MODEL_PATH", "")
DYNAMICRAFTER_MODEL_PATH = os.getenv("DYNAMICRAFTER_MODEL_PATH", "")
SVD_MODEL_PATH = os.getenv("SVD_MODEL_PATH", "stabilityai/stable-video-diffusion-img2vid-xt")
RIFE_MODEL_PATH = os.getenv("RIFE_MODEL_PATH", "")

# ============== 视频输出配置 ==============
DEFAULT_VIDEO_WIDTH = 180
DEFAULT_VIDEO_HEIGHT = 1920
DEFAULT_FPS = 24
DEFAULT_VIDEO_FORMAT = "mp4"
DEFAULT_VIDEO_CODEC = "libx264"

# ============== 字幕配置 ==============
DEFAULT_SUBTITLE_EFFECT = "karaoke"
DEFAULT_FONT = "Microsoft YaHei"
DEFAULT_FONT_SIZE = 48
DEFAULT_FONT_COLOR = "#FFFFFF"
DEFAULT_OUTLINE_COLOR = "#000000"
DEFAULT_OUTLINE_WIDTH = 2

# ============== 提示词配置 ==============
PROMPT_STYLE_PRESETS = {
    "anime": "anime style, high quality, detailed, vibrant colors, beautiful lighting",
    "realistic": "photorealistic, high quality, detailed, cinematic lighting, 8k",
    "abstract": "abstract art, colorful, artistic, creative, surreal",
    "cyberpunk": "cyberpunk style, neon lights, futuristic, dark atmosphere, high tech",
}
DEFAULT_STYLE = "anime"

SYSTEM_PROMPT_TEMPLATE = """你是一个专业的动漫插画提示词专家。根据中文歌词生成适合 FLUX.1 模型的英文图片提示词。

要求：
1. 风格：{style_description}
2. 构图：适合竖屏 9:16 比例，主体居中
3. 连贯性：与上一句画面保持视觉连续性
4. 情感：准确传达歌词的情感氛围
5. 细节：包含场景、人物、光影、氛围等描述
6. 输出：仅输出英文提示词，不要任何解释或前缀
"""

DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, ugly, bad anatomy, watermark, text, signature"
