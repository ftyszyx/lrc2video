"""
LRC 歌词文件解析器
"""

import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class LyricLine:
    """单行歌词数据"""

    index: int
    start: float  # 开始时间（秒）
    end: float  # 结束时间（秒）
    duration: float  # 持续时间（秒）
    text: str  # 歌词文本


@dataclass
class LRCData:
    """LRC 文件解析结果"""

    title: str
    artist: Optional[str]
    album: Optional[str]
    total_duration: float
    lyrics: List[LyricLine]


class LRCParser:
    """LRC 文件解析器"""

    # 时间戳正则: [mm:ss.xx] 或 [mm:ss:xx]
    TIME_PATTERN = re.compile(r"\[(\d{2}):(\d{2})[.:](\d{2,3})\]")
    # 元数据正则: [tag:value]
    META_PATTERN = re.compile(r"\[(\w+):(.+)\]")

    def __init__(self):
        pass

    def parse(self, lrc_path: str) -> LRCData:
        """
        解析 LRC 文件

        Args:
            lrc_path: LRC 文件路径

        Returns:
            LRCData: 解析后的歌词数据
        """
        path = Path(lrc_path)
        if not path.exists():
            raise FileNotFoundError(f"LRC 文件不存在: {lrc_path}")

        # 从文件名提取歌曲名
        title = path.stem

        # 读取文件内容
        content = path.read_text(encoding="utf-8")

        # 解析元数据和歌词
        metadata = self._parse_metadata(content)
        raw_lyrics = self._parse_lyrics(content)

        # 计算每行的结束时间和持续时间
        lyrics = self._calculate_durations(raw_lyrics)

        # 计算总时长
        total_duration = lyrics[-1].end if lyrics else 0.0

        return LRCData(
            title=metadata.get("ti", title),
            artist=metadata.get("ar"),
            album=metadata.get("al"),
            total_duration=total_duration,
            lyrics=lyrics,
        )

    def _parse_metadata(self, content: str) -> Dict[str, str]:
        """解析元数据标签"""
        metadata = {}
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # 检查是否是元数据行（不包含时间戳）
            if not self.TIME_PATTERN.search(line):
                match = self.META_PATTERN.match(line)
                if match:
                    tag, value = match.groups()
                    metadata[tag.lower()] = value.strip()

        return metadata

    def _parse_lyrics(self, content: str) -> List[tuple]:
        """解析歌词行，返回 (时间, 文本) 列表"""
        lyrics = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # 查找所有时间戳
            times = self.TIME_PATTERN.findall(line)
            if not times:
                continue

            # 提取歌词文本（移除所有时间戳）
            text = self.TIME_PATTERN.sub("", line).strip()
            if not text:
                continue

            # 每个时间戳对应同一段歌词
            for time_match in times:
                minutes, seconds, milliseconds = time_match
                # 处理毫秒（可能是2位或3位）
                ms = int(milliseconds)
                if len(milliseconds) == 2:
                    ms *= 10

                time_seconds = int(minutes) * 60 + int(seconds) + ms / 1000
                lyrics.append((time_seconds, text))

        # 按时间排序
        lyrics.sort(key=lambda x: x[0])

        return lyrics

    def _calculate_durations(self, raw_lyrics: List[tuple]) -> List[LyricLine]:
        """计算每行歌词的持续时间"""
        lyrics = []

        for i, (start, text) in enumerate(raw_lyrics):
            # 结束时间为下一行的开始时间，最后一行默认加5秒
            if i < len(raw_lyrics) - 1:
                end = raw_lyrics[i + 1][0]
            else:
                end = start + 5.0

            duration = end - start

            lyrics.append(
                LyricLine(index=i, start=start, end=end, duration=duration, text=text)
            )

        return lyrics

    def to_dict(self, lrc_data: LRCData) -> dict:
        """转换为字典格式"""
        return {
            "title": lrc_data.title,
            "artist": lrc_data.artist,
            "album": lrc_data.album,
            "total_duration": lrc_data.total_duration,
            "lyrics": [
                {
                    "index": line.index,
                    "start": line.start,
                    "end": line.end,
                    "duration": line.duration,
                    "text": line.text,
                }
                for line in lrc_data.lyrics
            ],
        }
