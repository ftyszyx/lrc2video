# 🎵 LRC2Video - 歌词视频生成器

将 LRC 歌词文件自动转换为带艺术字幕的动漫风格视频。

## ✨ 功能特点

- 📝 **LRC 解析**: 自动解析 LRC 歌词文件，提取歌词和时间轴
- 🤖 **AI 提示词生成**: 使用 GPT-4/Claude 根据歌词生成图片提示词
- 🎨 **FLUX.1 文生图**: 本地运行 FLUX.1 生成高质量动漫风格图片
- 🎬 **前后帧视频生成**: 支持多种模型（ToonCrafter/DynamiCrafter/SVD/RIFE）
- 📺 **艺术字幕**: 多种字幕效果（卡拉OK/渐入渐出/滚动/打字机）
- 🖥️ **Web 界面**: 简洁易用的 Gradio 界面

## 🔧 系统要求

- Python 3.10+
- NVIDIA GPU (建议 16GB+ 显存)
- FFmpeg
- CUDA 11.8+
- uv (推荐) 或 pip

## 📦 安装

### 1. 安装 uv（推荐）

uv 是一个极速的 Python 包管理器，比 pip 快 10-100 倍。

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux/Mac:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**或使用 pip 安装:**
```bash
pip install uv
```

### 2. 克隆项目

```bash
git clone https://github.com/yourusername/lrc2video.git
cd lrc2video
```

### 3. 创建虚拟环境并安装依赖

**使用 uv（推荐）:**
```bash
# 创建虚拟环境
uv venv --python 3.12

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

**安装 CUDA 版 PyTorch（需要 NVIDIA GPU）:**

`pyproject.toml` 里只声明了 `torch` 依赖，但 CUDA 版需要从 PyTorch 官方源安装对应的 wheel。请在激活虚拟环境后执行（示例以 cu121 为例）：

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
```

安装完成后可以验证：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 4. 安装 FFmpeg

**Windows:**
```bash
# 使用 chocolatey
choco install ffmpeg

# 使用 winget
winget install FFmpeg

# 或手动下载: https://ffmpeg.org/download.html
```

**Linux:**
```bash
sudo apt install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

### 5. 配置环境变量

```bash
# 复制示例配置文件
cp .env.example .env  # Linux/Mac
copy .env.example .env  # Windows
```

编辑 `.env` 文件，填入你的 API Key：

```env
# LLM API 配置
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini

# 或使用 Claude
# LLM_PROVIDER=claude
# CLAUDE_API_KEY=your_claude_api_key
```

### 6. (可选) 安装视频生成模型

**ToonCrafter (推荐动漫风格):**
```bash
git clone https://github.com/ToonCrafter/ToonCrafter.git
cd ToonCrafter && uv pip install -e .
```

**DynamiCrafter:**
```bash
git clone https://github.com/Doubiiu/DynamiCrafter.git
cd DynamiCrafter && uv pip install -e .
```

**RIFE (快速帧插值):**
```bash
git clone https://github.com/hzwer/ECCV2022-RIFE.git
cd ECCV2022-RIFE && uv pip install -e .
```

## 🚀 使用方法

### 启动 Web 界面

**使用 uv:**
```bash
uv run python app.py
```

**或激活环境后直接运行:**
```bash
python app.py
```

然后在浏览器打开 `http://localhost:7860`

### 命令行使用

```python
from core import LRCParser, PromptGenerator, ImageGenerator, VideoGenerator
from core import SubtitleRenderer, VideoAssembler

# 1. 解析 LRC
parser = LRCParser()
lrc_data = parser.parse("examples/demo.lrc")

# 2. 生成提示词
prompt_gen = PromptGenerator(provider="openai", api_key="your_key")
prompts = prompt_gen.generate_prompts_batch(
    lyrics=[line.text for line in lrc_data.lyrics],
    style="anime"
)

# 3. 生成图片
image_gen = ImageGenerator()
image_paths = image_gen.generate_batch(prompts)

# 4. 生成视频片段
video_gen = VideoGenerator(model_type="tooncrafter")
durations = [line.duration for line in lrc_data.lyrics[:-1]]
video_paths = video_gen.generate_batch(image_paths, durations)

# 5. 渲染字幕
subtitle_renderer = SubtitleRenderer()
subtitle_path = subtitle_renderer.render(lrc_data.lyrics, effect="karaoke")

# 6. 合成最终视频
assembler = VideoAssembler()
final_video = assembler.assemble(video_paths, subtitle_path, "output_video")
```

## 📁 项目结构

```
lrc2video/
├── app.py                      # Gradio Web 主入口
├── config.py                   # 配置文件
├── pyproject.toml              # 项目配置 (uv/pip)
├── README.md                   # 项目说明
├── .env.example                # 环境变量示例
│
├── core/                       # 核心处理模块
│   ├── __init__.py
│   ├── lrc_parser.py          # LRC 解析
│   ├── prompt_generator.py    # LLM 提示词生成
│   ├── image_generator.py     # FLUX.1 文生图
│   ├── video_generator.py     # 前后帧生成视频
│   ├── subtitle_renderer.py   # 字幕渲染
│   └── video_assembler.py     # 视频合成
│
├── models/                     # 模型封装
│   ├── __init__.py
│   ├── flux_wrapper.py
│   ├── tooncrafter_wrapper.py
│   ├── dynamicrafter_wrapper.py
│   ├── svd_wrapper.py
│   └── rife_wrapper.py
│
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── ffmpeg_utils.py
│   └── file_utils.py
│
├── templates/                  # 字幕模板
├── output/                     # 输出目录
├── cache/                      # 缓存目录
└── examples/                   # 示例文件
    └── demo.lrc
```

## ⚡ uv 常用命令

| 命令 | 说明 |
|------|------|
| `uv venv` | 创建虚拟环境 |
| `uv pip install -e .` | 安装当前项目（可编辑模式） |
| `uv pip install <package>` | 安装包 |
| `uv pip list` | 列出已安装的包 |
| `uv run python app.py` | 在虚拟环境中运行 |
| `uv pip compile pyproject.toml -o requirements.txt` | 生成 requirements.txt |

## 🎨 支持的风格

| 风格 | 说明 |
|------|------|
| `anime` | 动漫插画风格（默认） |
| `realistic` | 真实照片风格 |
| `abstract` | 抽象艺术风格 |
| `cyberpunk` | 赛博朋克风格 |

## 🎬 支持的视频模型

| 模型 | 说明 | 显存需求 |
|------|------|----------|
| `tooncrafter` | 动漫风格最佳，推荐 | ~10GB |
| `dynamicrafter` | 通用效果好 | ~12GB |
| `svd` | Stability AI 官方 | ~16GB |
| `rife` | 快速帧插值，速度最快 | ~2GB |

## 📝 支持的字幕效果

| 效果 | 说明 |
|------|------|
| `karaoke` | 卡拉OK逐字高亮 |
| `fade` | 渐入渐出 |
| `scroll` | 滚动字幕 |
| `typewriter` | 打字机效果 |

## ⚠️ 注意事项

1. **显存管理**: 程序会自动在不同阶段加载/卸载模型以节省显存
2. **首次运行**: 首次运行会自动下载模型，需要较长时间
3. **API 费用**: 使用 OpenAI/Claude API 会产生费用
4. **视频时长**: 较长的歌曲会需要更多处理时间

## 🐛 常见问题

### Q: 显存不足怎么办？
A: 尝试使用 RIFE 模型，它只需要约 2GB 显存。

### Q: 生成速度很慢？
A: 可以减少推理步数，或使用 RIFE 快速帧插值。

### Q: 字幕显示不正确？
A: 确保 LRC 文件编码为 UTF-8。

### Q: uv 安装依赖失败？
A: 尝试使用以下命令：
```bash
uv pip install -e . --no-cache
```

### Q: 如何更新依赖？
A: 使用 uv 更新：
```bash
uv pip install -e . --upgrade
```

## 📄 License

MIT License

## 🙏 致谢

- [uv](https://github.com/astral-sh/uv) - 极速 Python 包管理器
- [FLUX.1](https://github.com/black-forest-labs/flux) - 文生图模型
- [ToonCrafter](https://github.com/ToonCrafter/ToonCrafter) - 动漫视频生成
- [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter) - 视频生成
- [RIFE](https://github.com/hzwer/ECCV2022-RIFE) - 帧插值
- [Gradio](https://gradio.app/) - Web 界面
