import os
import re
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st
from openai import OpenAI


APP_TITLE = "Creativeeus"


# -----------------------------
# OpenAI
# -----------------------------
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Add your API key in Windows Environment Variables, then reopen Command Prompt."
        )
    return OpenAI(api_key=api_key)


# -----------------------------
# Helpers
# -----------------------------
def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    return re.sub(r"_+", "_", text).strip("_") or "output"


def ensure_work_dir() -> str:
    work_dir = os.path.join(tempfile.gettempdir(), "creativeeus_app")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def format_ass_time(seconds: float) -> str:
    cs = int(round(seconds * 100))
    hours = cs // 360000
    cs %= 360000
    minutes = cs // 6000
    cs %= 6000
    secs = cs // 100
    cs %= 100
    return f"{hours}:{minutes:02}:{secs:02}.{cs:02}"


def run_ffmpeg_extract_audio(video_path: str, audio_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        audio_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{result.stderr}")


def render_final_video(video_path: str, ass_path: str, output_path: str) -> None:
    ass_filename = os.path.basename(ass_path)
    ass_folder = os.path.dirname(ass_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"ass={ass_filename}",
        "-c:a",
        "copy",
        output_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ass_folder,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg video render failed:\n{result.stderr}")


# -----------------------------
# Transcription
# -----------------------------
def transcribe_audio(
    audio_path: str,
    model_size: str,
    source_language: Optional[str],
    vad_filter: bool,
) -> List[Dict]:
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: faster-whisper. Install it with: pip install faster-whisper"
        ) from exc

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, _ = model.transcribe(
        audio_path,
        language=None if source_language == "auto" else source_language,
        task="transcribe",
        word_timestamps=False,
        vad_filter=vad_filter,
        beam_size=5,
    )

    rows: List[Dict] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        rows.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            }
        )

    if not rows:
        raise RuntimeError("No speech detected in this video.")
    return rows


# -----------------------------
# Caption chunking
# -----------------------------
def split_long_text(text: str, max_words: int = 6) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    current: List[str] = []
    for word in words:
        current.append(word)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


def split_segments_for_captions(segments: List[Dict], max_words: int = 6) -> List[Dict]:
    out: List[Dict] = []
    for seg in segments:
        parts = split_long_text(seg["text"], max_words=max_words)
        if not parts:
            continue

        total_duration = max(0.5, seg["end"] - seg["start"])
        part_duration = total_duration / len(parts)

        for i, part in enumerate(parts):
            start = seg["start"] + i * part_duration
            end = seg["start"] + (i + 1) * part_duration
            out.append({"start": start, "end": end, "text": part.strip()})
    return out


# -----------------------------
# AI Hinglish rewrite
# -----------------------------
def rewrite_to_hinglish(lines: List[Dict]) -> List[Dict]:
    client = get_openai_client()
    rewritten: List[Dict] = []

    system_prompt = (
        "You rewrite subtitle lines into natural Hinglish for Indian creator videos. "
        "Use Roman Hindi only, not Devanagari. Keep English creator/editing words in English. "
        "Make lines short, readable, and natural for reels and shorts. Do not add emojis. "
        "Do not change meaning. Use common Hinglish spellings like kyun, kaise, tumhara, samajh, bata raha hoon. "
        "Return only the rewritten subtitle line."
    )

    for item in lines:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Rewrite this subtitle into natural Hinglish:\n\n{item['text']}",
                },
            ],
        )
        text = (response.output_text or "").strip()
        if not text:
            text = item["text"]

        rewritten.append({"start": item["start"], "end": item["end"], "text": text})

    return rewritten


# -----------------------------
# Subtitle builders
# -----------------------------
def build_srt(lines: List[Dict]) -> str:
    blocks = []
    for idx, line in enumerate(lines, start=1):
        blocks.append(
            f"{idx}\n"
            f"{format_srt_time(line['start'])} --> {format_srt_time(line['end'])}\n"
            f"{line['text']}\n"
        )
    return "\n".join(blocks)


def build_ass(lines: List[Dict], font_size: int, y_margin: int, animation_style: str) -> str:
    if animation_style == "Highlight Punch":
        primary = "&H00FFFFFF"
        secondary = "&H0000CCFF"
        outline = "&H00101010"
        back = "&H46000000"
        border = 4
        shadow = 0
        style_name = "Punch"
    elif animation_style == "Bold Reels":
        primary = "&H00FFFFFF"
        secondary = "&H0000D7FF"
        outline = "&H00111111"
        back = "&H50000000"
        border = 4
        shadow = 0
        style_name = "Reels"
    elif animation_style == "Glow Pop":
        primary = "&H00FFFFFF"
        secondary = "&H0078E8FF"
        outline = "&H00202220"
        back = "&H46000000"
        border = 3
        shadow = 1
        style_name = "Glow"
    else:
        primary = "&H00F9F9F9"
        secondary = "&H00A7E6BE"
        outline = "&H001A1A1A"
        back = "&H50000000"
        border = 2
        shadow = 0
        style_name = "Clean"

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: {style_name},Arial,{font_size},{primary},{secondary},{outline},{back},1,0,0,0,100,100,0,0,1,{border},{shadow},2,60,60,{y_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for line in lines:
        clean_text = line["text"].replace("\n", " ").replace("{", "").replace("}", "").strip()
        words = clean_text.split()
        duration = max(0.2, line["end"] - line["start"])

        if animation_style == "Highlight Punch":
            anim = "{\\an2\\fad(40,80)\\blur0.5\\bord4\\shad0\\fscx86\\fscy86\\t(0,90,\\fscx110\\fscy110)\\t(90,180,\\fscx100\\fscy100)}"
            if len(words) >= 2:
                emphasis_index = min(len(words) - 1, max(0, len(words) // 2))
                styled = []
                for i, word in enumerate(words):
                    if i == emphasis_index:
                        styled.append("{\\c&H0000CCFF&\\3c&H000000&}" + word.upper() + "{\\c}")
                    else:
                        styled.append(word.upper())
                text = " ".join(styled)
            else:
                text = clean_text.upper()
        elif animation_style == "Bold Reels":
            anim = "{\\an2\\fad(60,100)\\blur0.6\\bord4\\shad0\\fsp1\\fscx88\\fscy88\\t(0,120,\\fscx108\\fscy108)\\t(120,220,\\fscx100\\fscy100)}"
            text = clean_text.upper()
        elif animation_style == "Glow Pop":
            anim = "{\\an2\\fad(50,120)\\blur1.6\\bord3\\shad1\\fscx90\\fscy90\\t(0,140,\\fscx105\\fscy105)\\t(140,240,\\fscx100\\fscy100)}"
            text = clean_text
        else:
            anim = "{\\an2\\fad(60,90)\\blur0.3\\bord2\\shad0\\fscx96\\fscy96\\t(0,120,\\fscx100\\fscy100)}"
            text = clean_text

        if duration < 0.7:
            anim = anim.replace("\\fad(60,100)", "\\fad(20,40)")
            anim = anim.replace("\\fad(50,120)", "\\fad(20,50)")
            anim = anim.replace("\\fad(40,80)", "\\fad(15,35)")
            anim = anim.replace("\\fad(60,90)", "\\fad(20,40)")

        events.append(
            f"Dialogue: 0,{format_ass_time(line['start'])},{format_ass_time(line['end'])},{style_name},,0,0,0,,{anim}{text}"
        )

    return header + "\n".join(events) + "\n"


# -----------------------------
# App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.markdown(
        """
        <style>
        .stApp { background: #f4f4f1; }
        section[data-testid="stSidebar"] { background: #315846; border-right: 1px solid rgba(255,255,255,0.08); }
        section[data-testid="stSidebar"] * { color: #f4f4ef !important; }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1220px; }
        .hero-wrap { background: linear-gradient(135deg, #335a47 0%, #3e6b56 100%); padding: 28px 34px 30px 34px; margin-bottom: 22px; color: white; }
        .top-nav { position: sticky; top: 0; z-index: 999; backdrop-filter: blur(10px); background: rgba(51, 90, 71, 0.88); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 14px 18px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 34px; }
        .brand { font-size: 1.6rem; font-weight: 800; color: #ffffff; letter-spacing: -0.03em; }
        .nav-links { display: flex; gap: 22px; color: rgba(255,255,255,0.82); font-size: 0.92rem; flex-wrap: wrap; }
        .nav-links a { display: inline-flex; align-items: center; padding: 6px 2px; transition: all 0.18s ease; color: rgba(255,255,255,0.82); text-decoration: none; position: relative; }
        .nav-links a:hover { color: #ffffff; transform: translateY(-1px); }
        .nav-links a.active { color: white; position: relative; font-weight: 700; }
        .nav-links a.active::after { content: ""; position: absolute; left: 0; bottom: -6px; width: 100%; height: 3px; background: #f2c94c; border-radius: 999px; }
        .hero-grid { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 22px; align-items: center; }
        .hero-title { font-size: 3.3rem; line-height: 1.05; font-weight: 800; margin-bottom: 16px; color: #ffffff; max-width: 560px; }
        .hero-sub { color: rgba(255,255,255,0.82); font-size: 1rem; line-height: 1.7; max-width: 500px; margin-bottom: 20px; }
        .hero-btn-row { display: flex; gap: 12px; flex-wrap: wrap; }
        .hero-pill-primary { display: inline-block; background: #f2c94c; color: #162820; padding: 11px 18px; border-radius: 999px; font-weight: 800; font-size: 0.92rem; }
        .hero-pill-secondary { display: inline-block; border: 1px solid rgba(255,255,255,0.35); color: #ffffff; padding: 11px 18px; border-radius: 999px; font-weight: 700; font-size: 0.92rem; }
        .hero-card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); border-radius: 26px; min-height: 280px; padding: 24px; display: flex; flex-direction: column; justify-content: center; }
        .hero-card-title { font-size: 1.9rem; font-weight: 800; color: #ffffff; margin-bottom: 8px; }
        .hero-card-sub { color: rgba(255,255,255,0.78); line-height: 1.7; font-size: 0.98rem; }
        .hero-mini-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
        .hero-mini-box { background: rgba(255,255,255,0.08); border-radius: 18px; padding: 14px; }
        .hero-mini-title { font-weight: 800; font-size: 1.12rem; color: white; }
        .hero-mini-sub { font-size: 0.9rem; color: rgba(255,255,255,0.75); margin-top: 4px; }
        .section-shell { background: #f4f4f1; padding-top: 10px; }
        .section-heading { font-size: 2rem; font-weight: 800; color: #1f2a23; margin-bottom: 8px; }
        .section-sub { color: #6c746f; line-height: 1.7; max-width: 620px; margin-bottom: 20px; }
        .soft-card { background: #ffffff; border-radius: 22px; padding: 18px; border: 1px solid #e9e7e0; box-shadow: 0 8px 24px rgba(0,0,0,0.03); }
        .metric-grid, .feature-icon-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 10px; margin-bottom: 20px; }
        .feature-icon-grid { grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 18px; margin-bottom: 14px; }
        .metric-card, .feature-icon-card { background: #ffffff; border: 1px solid #e8e6df; border-radius: 18px; padding: 18px; }
        .feature-icon-card { min-height: 120px; }
        .metric-title, .feature-icon-title { font-size: 1.1rem; font-weight: 800; color: #223128; margin-bottom: 6px; }
        .metric-text, .feature-icon-sub { color: #6d746f; font-size: 0.93rem; line-height: 1.6; }
        .feature-icon { width: 42px; height: 42px; border-radius: 50%; background: #edf3ee; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; margin-bottom: 10px; }
        .preview-line { background: #ffffff; border: 1px solid #ebe8e2; border-radius: 16px; padding: 14px 16px; margin-bottom: 10px; }
        .time-tag { color: #6d746f; font-size: 0.84rem; font-weight: 700; margin-bottom: 4px; }
        .caption-text { color: #19241d; font-size: 1.08rem; font-weight: 800; line-height: 1.5; }
        .slider-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 14px 14px 6px 14px;
            margin-top: 8px;
            margin-bottom: 12px;
        }
        .slider-card-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
            color: #f2f4ef;
            margin-bottom: 4px;
        }
        .slider-card-sub {
            font-size: 0.84rem;
            color: rgba(255,255,255,0.72);
            margin-bottom: 2px;
            line-height: 1.5;
        }
        .section-anchor-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.12em; color: #7d847f; margin-bottom: 6px; font-weight: 700; }
        .stButton > button { background: #2f5a46; color: white; border-radius: 999px; border: none; padding: 0.75rem 1.2rem; font-weight: 800; }
        .stDownloadButton > button { background: #ffffff; color: #294b3b; border-radius: 999px; border: 1px solid #d8d5cc; padding: 0.75rem 1.2rem; font-weight: 800; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { background: #ffffff; border: 1px solid #e7e4db; border-radius: 12px; padding: 10px 16px; }
        .footer-note { color: #6f7570; text-align: center; margin-top: 32px; font-size: 0.92rem; }
        @media (max-width: 900px) { .hero-grid, .metric-grid, .hero-mini-grid, .feature-icon-grid { grid-template-columns: 1fr; } .hero-title { font-size: 2.3rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div id="home"></div>
        <div class="hero-wrap">
            <div class="top-nav">
                <div class="brand">Creativeeus.</div>
                <div class="nav-links">
                    <a href="#home" class="active">Home</a>
                    <a href="#studio">Studio</a>
                    <a href="#captions">Captions</a>
                    <a href="#exports">Exports</a>
                    <a href="#about">About</a>
                </div>
            </div>
            <div class="hero-grid">
                <div>
                    <div class="hero-title">Reels Caption Studio For Modern Creators</div>
                    <div class="hero-sub">Generate accurate Hinglish captions, preview clean lines, and export final videos from one stylish workspace built for short-form creators.</div>
                    <div class="hero-btn-row">
                        <span class="hero-pill-primary">Generate Captions</span>
                        <span class="hero-pill-secondary">Explore Workflow</span>
                    </div>
                </div>
                <div class="hero-card">
                    <div class="hero-card-title">Creator-first workflow</div>
                    <div class="hero-card-sub">Built for reels, shorts, edits, talking-head videos, tutorials, and high-speed content production with a premium green editorial feel.</div>
                    <div class="hero-mini-grid">
                        <div class="hero-mini-box">
                            <div class="hero-mini-title">4 Styles</div>
                            <div class="hero-mini-sub">Highlight Punch, Bold Reels, Glow Pop, Clean Studio</div>
                        </div>
                        <div class="hero-mini-box">
                            <div class="hero-mini-title">Direct Export</div>
                            <div class="hero-mini-sub">Render final video inside Creativeeus</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "generated" not in st.session_state:
        st.session_state.generated = False
    if "video_path" not in st.session_state:
        st.session_state.video_path = ""
    if "base_name" not in st.session_state:
        st.session_state.base_name = "output"
    if "lines" not in st.session_state:
        st.session_state.lines = []
    if "srt_text" not in st.session_state:
        st.session_state.srt_text = ""
    if "ass_text" not in st.session_state:
        st.session_state.ass_text = ""

    with st.sidebar:
        st.markdown('<div style="font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase; color:rgba(255,255,255,0.7); margin-bottom:6px;">Exclusively for Ankit and Yash</div>', unsafe_allow_html=True)
        st.markdown("## Creativeeus")
        st.caption("Creator control panel")
        st.markdown("### Transcription")
        model_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium"], index=2)
        source_language = st.selectbox("Source language", ["auto", "hi", "en"], index=0)
        vad_filter = st.checkbox("Use VAD filter", value=True)

        st.markdown("### Caption Style")

        st.markdown('<div class="slider-card"><div class="slider-card-title">Max words per caption</div><div class="slider-card-sub">Use lower values for faster, punchier reels subtitles.</div></div>', unsafe_allow_html=True)
        max_words = st.slider("Max words per caption", 3, 10, 6, label_visibility="collapsed")

        st.markdown('<div class="slider-card"><div class="slider-card-title">Font size</div><div class="slider-card-sub">Increase this for bold mobile-first subtitle styling.</div></div>', unsafe_allow_html=True)
        font_size = st.slider("Font size", 28, 96, 58, label_visibility="collapsed")

        st.markdown('<div class="slider-card"><div class="slider-card-title">Bottom margin</div><div class="slider-card-sub">Move captions higher or lower on the video frame.</div></div>', unsafe_allow_html=True)
        y_margin = st.slider("Bottom margin", 20, 220, 70, label_visibility="collapsed")

        st.markdown('<div class="slider-card"><div class="slider-card-title">Animation style</div><div class="slider-card-sub">Choose the subtitle motion style for the final render.</div></div>', unsafe_allow_html=True)
        animation_style = st.selectbox("Animation style", ["Highlight Punch", "Bold Reels", "Glow Pop", "Clean Studio"], index=0, label_visibility="collapsed")

        st.markdown("### Creativeeus Use Cases")
        st.caption("Reels • YouTube Shorts • Talking Head Videos • Tutorial Edits")

    st.markdown('<div id="studio"></div><div class="section-shell">', unsafe_allow_html=True)
    st.markdown('<div class="section-anchor-title">Studio</div><div class="section-heading">Crafted for excellent video captions.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload your clip, generate more natural Hinglish captions, preview subtitle lines, and export your finished video from a cleaner brand-ready interface.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="metric-grid">
            <div class="metric-card"><div class="metric-title">Fast Workflow</div><div class="metric-text">Upload, caption, preview, and export in one place without switching tools.</div></div>
            <div class="metric-card"><div class="metric-title">Reels Ready</div><div class="metric-text">Short readable lines designed for creator content and modern vertical video edits.</div></div>
            <div class="metric-card"><div class="metric-title">Direct Export</div><div class="metric-text">Render your final captioned video from the same workspace after reviewing the text.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your video", type=["mp4", "mov", "mkv", "avi", "webm"])
    if uploaded is not None:
        st.video(uploaded)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="feature-icon-grid">
            <div class="feature-icon-card"><div class="feature-icon">🎬</div><div class="feature-icon-title">Reels First</div><div class="feature-icon-sub">Built around short-form creator workflows and quick caption turnaround.</div></div>
            <div class="feature-icon-card"><div class="feature-icon">📝</div><div class="feature-icon-title">Natural Hinglish</div><div class="feature-icon-sub">Cleaner Roman Hindi captions with better readability for Indian audiences.</div></div>
            <div class="feature-icon-card"><div class="feature-icon">✨</div><div class="feature-icon-title">Animated Styles</div><div class="feature-icon-sub">Choose highlight punch, bold reels, glow pop, or clean studio motion.</div></div>
            <div class="feature-icon-card"><div class="feature-icon">⬇️</div><div class="feature-icon-title">Direct Export</div><div class="feature-icon-sub">Render your final captioned video from the same dashboard without extra steps.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded is not None and st.button("Generate Accurate Hinglish Captions", type="primary"):
        try:
            work_dir = ensure_work_dir()
            base_name = slugify(Path(uploaded.name).stem)
            video_path = os.path.join(work_dir, uploaded.name)
            audio_path = os.path.join(work_dir, f"{base_name}.wav")

            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner("Extracting audio..."):
                run_ffmpeg_extract_audio(video_path, audio_path)

            with st.spinner("Transcribing speech..."):
                segments = transcribe_audio(audio_path=audio_path, model_size=model_size, source_language=source_language, vad_filter=vad_filter)

            with st.spinner("Splitting captions..."):
                lines = split_segments_for_captions(segments, max_words=max_words)

            with st.spinner("Rewriting into natural Hinglish..."):
                lines = rewrite_to_hinglish(lines)

            with st.spinner("Building subtitle files..."):
                srt_text = build_srt(lines)
                ass_text = build_ass(lines, font_size=font_size, y_margin=y_margin, animation_style=animation_style)

            st.session_state.generated = True
            st.session_state.video_path = video_path
            st.session_state.base_name = base_name
            st.session_state.lines = lines
            st.session_state.srt_text = srt_text
            st.session_state.ass_text = ass_text
            st.success("Creativeeus captions generated successfully.")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.generated:
        st.markdown('<div id="captions"></div><div class="section-anchor-title">Captions</div><div class="section-heading" style="margin-top:24px;">Why creators choose Creativeeus</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Review subtitle lines in a cleaner layout, download subtitle files, and render final videos from the same creator dashboard.</div>', unsafe_allow_html=True)
        preview_tab, export_tab = st.tabs(["Preview", "Exports"])

        with preview_tab:
            for item in st.session_state.lines[:30]:
                st.markdown(f'''<div class="preview-line"><div class="time-tag">{format_srt_time(item['start'])} → {format_srt_time(item['end'])}</div><div class="caption-text">{item['text']}</div></div>''', unsafe_allow_html=True)

        with export_tab:
            st.markdown('<div id="exports"></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(label="Download SRT", data=st.session_state.srt_text, file_name=f"{st.session_state.base_name}_hinglish.srt", mime="application/x-subrip")
            with c2:
                st.download_button(label="Download ASS", data=st.session_state.ass_text, file_name=f"{st.session_state.base_name}_hinglish.ass", mime="text/plain")

            st.markdown('<div class="section-anchor-title">Exports</div>', unsafe_allow_html=True)
            st.markdown("### Final Video Export")
            if st.button("Render Final Video"):
                try:
                    work_dir = ensure_work_dir()
                    ass_path = os.path.join(work_dir, f"{st.session_state.base_name}_hinglish.ass")
                    output_path = os.path.join(work_dir, f"{st.session_state.base_name}_hinglish_final.mp4")
                    with open(ass_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state.ass_text)
                    with st.spinner("Rendering final video..."):
                        render_final_video(st.session_state.video_path, ass_path, output_path)
                    with open(output_path, "rb") as f:
                        video_bytes = f.read()
                    st.success("Final video rendered successfully.")
                    st.download_button(label="Download Final Captioned Video", data=video_bytes, file_name=f"{st.session_state.base_name}_hinglish_final.mp4", mime="video/mp4")
                except Exception as exc:
                    st.error(str(exc))

    st.markdown('<div id="about"></div><div class="section-anchor-title" style="text-align:center;">About</div><div class="footer-note">Creativeeus • Premium green creator interface inspired by editorial product landing pages</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
