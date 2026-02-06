"""
FFmpeg 工具函数
"""

import subprocess
import shutil
from pathlib import Path
from typing import List


def check_ffmpeg() -> bool:
    """检查 FFmpeg 是否可用"""
    return shutil.which("ffmpeg") is not None


def get_video_duration(video_path: str) -> float:
    """获取视频时长"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def concat_videos(video_paths: List[Path], output_path: Path) -> Path:
    """拼接视频"""
    list_file = output_path.parent / "concat_list.txt"

    with open(list_file, "w", encoding="utf-8") as f:
        for path in video_paths:
            # 使用正斜杠
            path_str = str(path.absolute()).replace("\\", "/")
            f.write(f"file '{path_str}'\n")

    cmd = [
        "ffmpeg",
        "-y",
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
    finally:
        if list_file.exists():
            list_file.unlink()  # 删除临时文件

    return output_path


def burn_subtitles(video_path: Path, subtitle_path: Path, output_path: Path) -> Path:
    """烧录字幕到视频"""
    # 处理 Windows 路径中的特殊字符
    subtitle_str = str(subtitle_path.absolute()).replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"ass='{subtitle_str}'",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def add_audio(video_path: Path, audio_path: Path, output_path: Path) -> Path:
    """为视频添加音频"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """从视频中提取音频"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def resize_video(video_path: Path, output_path: Path, width: int, height: int) -> Path:
    """调整视频尺寸"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"scale={width}:{height}",
        "-c:a",
        "copy",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path
