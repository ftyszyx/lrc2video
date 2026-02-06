"""
字幕渲染模块 - 支持多种艺术效果
"""

import sys
from pathlib import Path
from typing import List

import pysubs2
from pysubs2 import SSAFile, SSAEvent, SSAStyle

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from core.lrc_parser import LyricLine


class SubtitleRenderer:
    """字幕渲染器"""

    EFFECTS = ["karaoke", "fade", "scroll", "typewriter"]

    def __init__(self):
        self.width = config.DEFAULT_VIDEO_WIDTH
        self.height = config.DEFAULT_VIDEO_HEIGHT

    def render(
        self,
        lyrics: List[LyricLine],
        effect: str = None,
        output_path: str = None,
        font: str = None,
        font_size: int = None,
        font_color: str = None,
        outline_color: str = None,
    ) -> Path:
        """
        渲染字幕文件

        Args:
            lyrics: 歌词数据列表
            effect: 字幕效果
            output_path: 输出路径
            font: 字体
            font_size: 字体大小
            font_color: 字体颜色
            outline_color: 描边颜色

        Returns:
            ASS 字幕文件路径
        """
        effect = effect or config.DEFAULT_SUBTITLE_EFFECT
        output_path = (
            Path(output_path) if output_path else config.CACHE_DIR / "subtitle.ass"
        )

        # 创建 ASS 文件
        subs = SSAFile()

        # 设置样式
        style = self._create_style(
            font=font or config.DEFAULT_FONT,
            font_size=font_size or config.DEFAULT_FONT_SIZE,
            font_color=font_color or config.DEFAULT_FONT_COLOR,
            outline_color=outline_color or config.DEFAULT_OUTLINE_COLOR,
        )
        subs.styles["Default"] = style

        # 根据效果生成字幕事件
        if effect == "karaoke":
            events = self._render_karaoke(lyrics)
        elif effect == "fade":
            events = self._render_fade(lyrics)
        elif effect == "scroll":
            events = self._render_scroll(lyrics)
        elif effect == "typewriter":
            events = self._render_typewriter(lyrics)
        else:
            events = self._render_simple(lyrics)

        subs.events = events

        # 保存文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subs.save(str(output_path))

        return output_path

    def _create_style(
        self, font: str, font_size: int, font_color: str, outline_color: str
    ) -> SSAStyle:
        """创建字幕样式"""
        style = SSAStyle()
        style.fontname = font
        style.fontsize = font_size
        style.primarycolor = pysubs2.Color(*self._hex_to_rgb(font_color), 0)
        style.outlinecolor = pysubs2.Color(*self._hex_to_rgb(outline_color), 0)
        style.outline = config.DEFAULT_OUTLINE_WIDTH
        style.shadow = 2
        style.alignment = 2  # 底部居中
        style.marginv = 80  # 底部边距
        return style

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """十六进制颜色转 RGB"""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def _render_simple(self, lyrics: List[LyricLine]) -> List[SSAEvent]:
        """简单字幕（无特效）"""
        events = []
        for line in lyrics:
            event = SSAEvent()
            event.start = int(line.start * 1000)
            event.end = int(line.end * 1000)
            event.text = line.text
            events.append(event)
        return events

    def _render_karaoke(self, lyrics: List[LyricLine]) -> List[SSAEvent]:
        """卡拉OK效果 - 逐字高亮"""
        events = []
        for line in lyrics:
            event = SSAEvent()
            event.start = int(line.start * 1000)
            event.end = int(line.end * 1000)

            # 计算每个字的时长
            text_len = len(line.text) if line.text else 1
            char_duration = int(line.duration * 100 / text_len)  # 单位: 10ms

            # 构建卡拉OK标签
            karaoke_text = ""
            for char in line.text:
                karaoke_text += f"{{\\k{char_duration}}}{char}"

            event.text = karaoke_text
            events.append(event)
        return events

    def _render_fade(self, lyrics: List[LyricLine]) -> List[SSAEvent]:
        """渐入渐出效果"""
        events = []
        fade_duration = 300  # 300ms 渐变

        for line in lyrics:
            event = SSAEvent()
            event.start = int(line.start * 1000)
            event.end = int(line.end * 1000)

            # 添加渐入渐出效果
            event.text = f"{{\\fad({fade_duration},{fade_duration})}}{line.text}"
            events.append(event)
        return events

    def _render_scroll(self, lyrics: List[LyricLine]) -> List[SSAEvent]:
        """滚动字幕效果"""
        events = []

        for line in lyrics:
            event = SSAEvent()
            event.start = int(line.start * 1000)
            event.end = int(line.end * 1000)

            # 从底部滚动到中间
            start_y = self.height
            end_y = self.height // 2

            event.text = f"{{\\move({self.width // 2},{start_y},{self.width // 2},{end_y})}}{line.text}"
            events.append(event)
        return events

    def _render_typewriter(self, lyrics: List[LyricLine]) -> List[SSAEvent]:
        """打字机效果 - 逐字出现"""
        events = []

        for line in lyrics:
            text_len = len(line.text) if line.text else 1
            char_duration = line.duration / text_len

            # 为每个字符创建一个事件
            for i, char in enumerate(line.text):
                event = SSAEvent()
                event.start = int((line.start + i * char_duration) * 1000)
                event.end = int(line.end * 1000)

                # 显示到当前字符为止的所有文本
                event.text = line.text[: i + 1]
                events.append(event)

        return events
