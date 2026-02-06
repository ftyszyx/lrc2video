"""
lrc2video - Gradio Web 界面
将 LRC 歌词文件自动转换为带艺术字幕的动漫风格视频
"""
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional
import gradio as gr

from pydantic import BaseModel, ConfigDict, ValidationError

import config
from core import (
    LRCParser,
    PromptGenerator,
    ImageGenerator,
    VideoGenerator,
    SubtitleRenderer,
    VideoAssembler,
)
from utils.file_utils import safe_filename


class JobParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    style: str
    video_model: str
    subtitle_effect: str


class JobState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    params: JobParams
    prompts_path: Optional[str] = None
    images_dir: Optional[str] = None
    videos_dir: Optional[str] = None
    subtitle_path: Optional[str] = None
    final_video: Optional[str] = None


def _get_lrc_sig(lrc_file) -> str:
    lrc_path = getattr(lrc_file, "name", None) or str(lrc_file)
    p = Path(lrc_path)
    stat = p.stat()
    return f"{p.name}|{stat.st_size}"


def compute_job_id(lrc_file) -> str:
    payload = {"lrc": _get_lrc_sig(lrc_file)}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    print(f"compute_job_id: {raw}")
    jobid = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    print(f"jobid: {jobid}")
    return jobid


def load_job_state(path: Path, current_params: JobParams) -> JobState:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}

    if not isinstance(data, dict):
        return JobState(params=current_params)

    data.setdefault("params", current_params.model_dump())
    try:
        return JobState.model_validate(data)
    except ValidationError:
        return JobState(params=current_params)


def save_job_state(path: Path, state: JobState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def invalidate_paths(
    state: JobState,
    *,
    prompts: bool,
    images: bool,
    videos: bool,
    subtitle: bool,
    final: bool,
) -> None:
    def _invalidate(field: str, is_dir: bool) -> None:
        value = getattr(state, field)
        if not value:
            return
        p = Path(value)
        if p.exists():
            if is_dir:
                shutil.rmtree(p)
            else:
                p.unlink()
        setattr(state, field, None)

    if prompts:
        _invalidate("prompts_path", is_dir=False)
    if images:
        _invalidate("images_dir", is_dir=True)
    if videos:
        _invalidate("videos_dir", is_dir=True)
    if subtitle:
        _invalidate("subtitle_path", is_dir=False)
    if final:
        _invalidate("final_video", is_dir=False)


def apply_param_mismatch_policy(
    state: JobState,
    *,
    current_params: JobParams,
    resume: bool,
) -> bool:
    if state.params == current_params:
        return False

    if resume:
        if state.params.style != current_params.style:
            invalidate_paths(state, prompts=True, images=True, videos=True, subtitle=True, final=True)
        elif state.params.video_model != current_params.video_model:
            invalidate_paths(state, prompts=False, images=False, videos=True, subtitle=True, final=True)
        elif state.params.subtitle_effect != current_params.subtitle_effect:
            invalidate_paths(state, prompts=False, images=False, videos=False, subtitle=True, final=True)
    else:
        invalidate_paths(state, prompts=True, images=True, videos=True, subtitle=True, final=True)

    state.params = current_params
    return True


class LRC2VideoApp:
    """LRC 转视频应用"""

    def __init__(self):
        self.lrc_parser = LRCParser()
        self.prompt_generator = None
        self.image_generator = None
        self.video_generator = None
        self.subtitle_renderer = SubtitleRenderer()
        self.video_assembler = VideoAssembler()

    def process(
        self,
        lrc_file,
        style: str,
        video_model: str,
        subtitle_effect: str,
        llm_api_key: str,
        resume: bool,
        progress=gr.Progress(),
    ) -> str:
        """
        主处理流程
        Args:
            lrc_file: 上传的 LRC 文件
            style: 视频风格
            video_model: 视频生成模型
            subtitle_effect: 字幕效果
            llm_provider: LLM 提供商
            llm_api_key: LLM API 密钥
            progress: Gradio 进度条

        Returns:
            生成的视频路径
        """
        _progress = progress

        def progress_log(fraction, desc=""):
            print(f"Progress: {fraction * 100:.2f}% - {desc}")
            _progress(fraction, desc=desc)

        job_id = compute_job_id(lrc_file)
        job_dir = config.CACHE_DIR / "jobs" / job_id
        state_path = job_dir / "state.json"

        if not resume and job_dir.exists():
            shutil.rmtree(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)
        current_params = JobParams(style=style, video_model=video_model, subtitle_effect=subtitle_effect)
        state = load_job_state(state_path, current_params)
        if apply_param_mismatch_policy(state, current_params=current_params, resume=resume):
            save_job_state(state_path, state)
        original_cache_dir = config.CACHE_DIR
        config.CACHE_DIR = job_dir
        try:
            # 1. 解析 LRC 文件
            progress_log(0.05, desc="正在解析 LRC 文件...")
            if isinstance(lrc_file, dict) and "path" in lrc_file:
                lrc_path = lrc_file["path"]
            else:
                lrc_path = lrc_file.name
            lrc_data = self.lrc_parser.parse(lrc_path)
            lyrics = lrc_data.lyrics
            song_title = safe_filename(lrc_data.title)

            progress_log(0.1, desc=f"解析完成，共 {len(lyrics)} 句歌词")

            # 2. 生成图片提示词
            progress_log(0.15, desc="正在生成图片提示词...")
            self.prompt_generator = PromptGenerator(api_key=llm_api_key if llm_api_key else None)

            lyric_texts = [line.text for line in lyrics]
            prompts_path = Path(state.prompts_path or (job_dir / "prompts.json"))
            prompts = self.prompt_generator.generate_prompts_batch(
                lyrics=lyric_texts,
                style=style,
                song_context=f"歌曲《{lrc_data.title}》",
                progress_callback=lambda cur, total: progress_log(0.15 + 0.15 * (cur / total), desc=f"生成提示词 {cur}/{total}"),
                checkpoint_path=str(prompts_path),
            )
            state.prompts_path = str(prompts_path)
            save_job_state(state_path, state)

            # 3. 生成图片
            progress_log(0.3, desc="正在生成图片...")
            self.image_generator = ImageGenerator()

            images_dir = Path(state.images_dir or (job_dir / "images"))
            existing_images = sorted(images_dir.glob("frame_*.png")) if images_dir.exists() else []
            if len(existing_images) == len(prompts) and len(existing_images) > 0:
                image_paths = existing_images
            else:
                image_paths = self.image_generator.generate_batch(
                    prompts=prompts,
                    output_dir=str(images_dir),
                    progress_callback=lambda cur, total: progress_log(0.3 + 0.3 * (cur / total), desc=f"生成图片 {cur}/{total}"),
                )
                state.images_dir = str(images_dir)
                save_job_state(state_path, state)

            # 卸载图片模型释放显存
            self.image_generator.unload_model()

            # 4. 生成视频片段
            progress_log(0.6, desc="正在生成视频片段...")
            self.video_generator = VideoGenerator(model_type=video_model)

            durations = [line.duration for line in lyrics[:-1]]  # 最后一帧不需要

            videos_dir = Path(state.videos_dir or (job_dir / "videos"))
            existing_videos = sorted(videos_dir.glob("segment_*.mp4")) if videos_dir.exists() else []
            expected_segments = max(len(image_paths) - 1, 0)
            if len(existing_videos) == expected_segments and expected_segments > 0:
                video_paths = existing_videos
            else:
                video_paths = self.video_generator.generate_batch(
                    image_paths=image_paths,
                    durations=durations,
                    output_dir=str(videos_dir),
                    progress_callback=lambda cur, total: progress_log(0.6 + 0.25 * (cur / total), desc=f"生成视频片段 {cur}/{total}"),
                )
                state.videos_dir = str(videos_dir)
                save_job_state(state_path, state)

            # 卸载视频模型
            self.video_generator.unload_model()

            # 5. 渲染字幕
            progress_log(0.85, desc="正在渲染字幕...")
            subtitle_path = Path(state.subtitle_path or (job_dir / "subtitle.ass"))
            if not subtitle_path.exists():
                subtitle_path = self.subtitle_renderer.render(lyrics=lyrics, effect=subtitle_effect, output_path=str(subtitle_path))
                state.subtitle_path = str(subtitle_path)
                save_job_state(state_path, state)

            # 6. 合成最终视频
            progress_log(0.9, desc="正在合成最终视频...")
            final_video = Path(state.final_video or (config.OUTPUT_DIR / f"{song_title}.mp4"))
            if not final_video.exists():
                final_video = self.video_assembler.assemble(
                    video_paths=video_paths,
                    subtitle_path=subtitle_path,
                    output_name=song_title,
                )
                state.final_video = str(final_video)
                save_job_state(state_path, state)

            progress_log(1.0, desc="完成！")

            return str(final_video)

        except Exception as e:
            raise gr.Error(f"处理失败: {str(e)}")
        finally:
            config.CACHE_DIR = original_cache_dir


def create_ui():
    """创建 Gradio 界面"""
    app = LRC2VideoApp()

    with gr.Blocks(title="🎵 LRC2Video - 歌词视频生成器", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
        # 🎵 LRC2Video - 歌词视频生成器
        
        上传 LRC 歌词文件，自动生成带艺术字幕的动漫风格视频
        
        ### 使用说明
        1. 上传 LRC 歌词文件（文件名将作为歌曲名）
        2. 选择视频风格、生成模型和字幕效果
        3. 配置 LLM API（用于生成图片提示词）
        4. 点击"开始生成"按钮
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # 输入区域
                lrc_file = gr.File(label="📁 上传 LRC 文件", file_types=[".lrc"], type="filepath")

                with gr.Row():
                    style = gr.Dropdown(
                        label="🎨 视频风格",
                        choices=list(config.PROMPT_STYLE_PRESETS.keys()),
                        value=config.DEFAULT_STYLE,
                        info="选择生成图片的风格",
                    )

                    video_model = gr.Dropdown(
                        label="🎬 视频模型",
                        choices=VideoGenerator.SUPPORTED_MODELS,
                        value=config.DEFAULT_VIDEO_MODEL,
                        info="选择前后帧生成视频的模型",
                    )

                subtitle_effect = gr.Dropdown(
                    label="📝 字幕效果",
                    choices=SubtitleRenderer.EFFECTS,
                    value=config.DEFAULT_SUBTITLE_EFFECT,
                    info="选择歌词字幕的显示效果",
                )

                with gr.Accordion("⚙️ LLM API 设置", open=True):
                    llm_api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        placeholder="输入你的 API Key（留空则使用环境变量）",
                        info="如果已在 .env 中配置，可以留空",
                    )

                resume = gr.Checkbox(
                    label="失败后继续（复用上次缓存）",
                    value=True,
                )

                generate_btn = gr.Button("🚀 开始生成", variant="primary", size="lg")

            with gr.Column(scale=1):
                # 输出区域
                output_video = gr.Video(label="🎥 生成的视频", interactive=False)

                gr.Markdown(
                    """
                ### 📌 注意事项
                - 首次运行会下载 AI 模型，需要较长时间
                - 生成过程中请勿关闭页面
                - 视频生成完成后可直接下载
                - 如遇到显存不足，请尝试使用 RIFE 模型
                """
                )

        # 绑定事件
        generate_btn.click(  # pylint: disable=no-member
            fn=app.process,
            inputs=[
                lrc_file,
                style,
                video_model,
                subtitle_effect,
                llm_api_key,
                resume,
            ],
            outputs=[output_video],
        )

        # 示例
        gr.Markdown(
            """
        ---
        ### 🎯 支持的功能
        
        | 功能 | 说明 |
        |------|------|
        | **视频风格** | 动漫、写实、抽象、赛博朋克 |
        | **视频模型** | ToonCrafter、DynamiCrafter、SVD、RIFE |
        | **字幕效果** | 卡拉OK、渐入渐出、滚动、打字机 |
        | **输出格式** | MP4 (1080x1920 竖屏) |
        """
        )

    return demo


if __name__ == "__main__":
    # 检查 FFmpeg
    from utils.ffmpeg_utils import check_ffmpeg

    if not check_ffmpeg():
        print("⚠️ 警告: FFmpeg 未安装，视频合成功能将不可用")
        print("请安装 FFmpeg: https://ffmpeg.org/download.html")

    # 启动应用
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
