"""
视频合成模块 - 拼接视频片段并烧录字幕
"""

import sys
import subprocess
from pathlib import Path
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class VideoAssembler:
    """视频合成器"""

    def __init__(self):
        self.ffmpeg_path = "ffmpeg"

    def concat_videos(self, video_paths: List[Path], output_path: Path = None) -> Path:
        """
        拼接多个视频片段

        Args:
            video_paths: 视频片段路径列表
            output_path: 输出路径

        Returns:
            拼接后的视频路径
        """
        output_path = output_path or config.CACHE_DIR / "concat.mp4"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建文件列表
        list_file = config.CACHE_DIR / "concat_list.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for video_path in video_paths:
                # 使用正斜杠并转义
                path_str = str(video_path.absolute()).replace("\\", "/")
                f.write(f"file '{path_str}'\n")

        # 使用 FFmpeg concat demuxer 拼接
        cmd = [
            self.ffmpeg_path,
            "-y",  # 覆盖输出
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg 拼接失败: {e.stderr}")
            raise

        return output_path

    def burn_subtitles(
        self, video_path: Path, subtitle_path: Path, output_path: Path = None
    ) -> Path:
        """
        将字幕烧录到视频

        Args:
            video_path: 视频路径
            subtitle_path: 字幕文件路径
            output_path: 输出路径

        Returns:
            带字幕的视频路径
        """
        output_path = output_path or config.OUTPUT_DIR / "output.mp4"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 处理 Windows 路径中的特殊字符
        subtitle_str = (
            str(subtitle_path.absolute()).replace("\\", "/").replace(":", "\\:")
        )

        # 使用 FFmpeg 烧录字幕
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"ass='{subtitle_str}'",
            "-c:v",
            config.DEFAULT_VIDEO_CODEC,
            "-preset",
            "medium",
            "-crf",
            "23",
            "-c:a",
            "copy",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg 字幕烧录失败: {e.stderr}")
            raise

        return output_path

    def assemble(
        self,
        video_paths: List[Path],
        subtitle_path: Path,
        output_name: str,
        output_dir: Path = None,
    ) -> Path:
        """
        完整的视频组装流程

        Args:
            video_paths: 视频片段列表
            subtitle_path: 字幕文件路径
            output_name: 输出文件名（不含扩展名）
            output_dir: 输出目录

        Returns:
            最终视频路径
        """
        output_dir = Path(output_dir) if output_dir else config.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 拼接视频
        print("正在拼接视频片段...")
        concat_video = self.concat_videos(video_paths)

        # 2. 烧录字幕
        print("正在烧录字幕...")
        final_output = output_dir / f"{output_name}.mp4"
        self.burn_subtitles(concat_video, subtitle_path, final_output)

        print(f"视频生成完成: {final_output}")
        return final_output
