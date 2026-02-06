"""
文件操作工具函数
"""

import sys
import shutil
from pathlib import Path
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def ensure_dir(path: Path) -> Path:
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_cache():
    """清理缓存目录"""
    cache_dir = config.CACHE_DIR
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def get_output_path(name: str, ext: str = "mp4") -> Path:
    """获取输出文件路径"""
    ensure_dir(config.OUTPUT_DIR)
    return config.OUTPUT_DIR / f"{name}.{ext}"


def list_files(directory: Path, pattern: str = "*") -> List[Path]:
    """列出目录中的文件"""
    directory = Path(directory)
    return sorted(directory.glob(pattern))


def safe_filename(name: str) -> str:
    """转换为安全的文件名"""
    # 移除或替换不安全字符
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        name = name.replace(char, "_")
    return name.strip()


def get_file_size(path: Path) -> int:
    """获取文件大小（字节）"""
    path = Path(path)
    if path.exists():
        return path.stat().st_size
    return 0


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def copy_file(src: Path, dst: Path) -> Path:
    """复制文件"""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def move_file(src: Path, dst: Path) -> Path:
    """移动文件"""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def delete_file(path: Path) -> bool:
    """删除文件"""
    path = Path(path)
    if path.exists():
        path.unlink()
        return True
    return False


def delete_dir(path: Path) -> bool:
    """删除目录"""
    path = Path(path)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        return True
    return False
