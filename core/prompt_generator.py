"""
LLM 提示词生成器 - 根据歌词生成图片提示词
"""

import sys
import json
from pathlib import Path
from typing import List
import config


# 开启 OpenAI / httpx 的 HTTP 请求日志
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger("httpx").setLevel(logging.DEBUG)
# logging.getLogger("openai").setLevel(logging.DEBUG)
# logging.getLogger("httpcore").setLevel(logging.DEBUG)

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class PromptGenerator:
    """使用 LLM 根据歌词生成图片提示词"""

    def __init__(self, provider: str = None, api_key: str = None):
        """
        初始化提示词生成器

        Args:
            provider: LLM 提供商 ("openai" 或 "claude")
            api_key: API 密钥
        """
        self.provider = "openai"
        self.api_key = api_key
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化 LLM 客户端"""
        from openai import OpenAI

        base_url = config.OPENAI_BASE_URL
        api_key = self.api_key or config.OPENAI_API_KEY

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30,
        )
        self.model = config.OPENAI_MODEL
        print(f"init openai client base_url: {base_url}, api_key: {api_key} model: {self.model}")

    def generate_prompt(
        self,
        lyric: str,
        style: str = None,
        previous_prompt: str = None,
        song_context: str = None,
    ) -> str:
        """
        为单句歌词生成图片提示词

        Args:
            lyric: 歌词文本
            style: 风格预设名称
            previous_prompt: 上一句的提示词（用于保持连贯性）
            song_context: 歌曲整体背景/氛围

        Returns:
            生成的英文图片提示词
        """
        style = style or config.DEFAULT_STYLE
        style_desc = config.PROMPT_STYLE_PRESETS.get(style, config.PROMPT_STYLE_PRESETS["anime"])

        system_prompt = config.SYSTEM_PROMPT_TEMPLATE.format(style_description=style_desc)

        user_message = f"歌词：{lyric}"
        if previous_prompt:
            user_message += f"\n上一句画面描述：{previous_prompt}"
        if song_context:
            user_message += f"\n歌曲整体氛围：{song_context}"

        print(f"generate prompt: {user_message} model: {self.model} system_prompt: {system_prompt}")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=False,
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def generate_prompts_batch(
        self,
        lyrics: List[str],
        style: str = None,
        song_context: str = None,
        progress_callback=None,
        checkpoint_path: str = None,
    ) -> List[str]:
        """
        批量生成提示词

        Args:
            lyrics: 歌词列表
            style: 风格预设
            song_context: 歌曲背景
            progress_callback: 进度回调函数 (current, total)

        Returns:
            提示词列表
        """
        checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
        prompts: List[str] = []
        if checkpoint_file and checkpoint_file.exists():
            try:
                prompts = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            except Exception:
                prompts = []

        previous_prompt = prompts[-1] if prompts else None

        start_index = len(prompts)
        if progress_callback and start_index > 0:
            progress_callback(start_index, len(lyrics))

        for i, lyric in enumerate(lyrics[start_index:], start=start_index):
            prompt = self.generate_prompt(
                lyric=lyric,
                style=style,
                previous_prompt=previous_prompt,
                song_context=song_context,
            )
            prompts.append(prompt)
            previous_prompt = prompt

            if checkpoint_file:
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_file.write_text(
                    json.dumps(prompts, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            if progress_callback:
                progress_callback(i + 1, len(lyrics))

        return prompts
