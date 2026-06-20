"""
====================================================
  Speech to Text with Automated Content Filtering
====================================================
  Requirements:
    py -3.12 -m pip install streamlit SpeechRecognition sounddevice scipy better-profanity

  How to run:
    py -3.12 -m streamlit run speech_to_text_filter.py
"""

import io
import time
import threading
import numpy as np
import streamlit as st
import speech_recognition as sr
import sounddevice as sd
import scipy.io.wavfile as wav
from better_profanity import profanity

profanity.load_censor_words()

NEGATIVE_WORDS = [
    "hate", "stupid", "idiot", "dumb", "ugly", "loser",
    "worthless", "useless", "horrible", "terrible", "awful",
    "disgusting", "pathetic", "moron", "jerk", "trash",
    "lame", "freak", "creep", "coward", "failure", "miserable",
    "hopeless", "furious", "rage", "despise", "dreadful",
    "horrific", "wretched", "vile", "nasty", "rotten","crazy"
]

SAMPLE_RATE = 16000

# ── Global recording state ──────────────────────────────────────
class _State:
    chunks: list = []
    active: bool = False
    lock: threading.Lock = threading.Lock()
    thread: threading.Thread | None = None

if "_state" not in st.session_state:
    st.session_state["_state"] = _State()

_state: _State = st.session_state["_state"]


def _record_worker():
    """Background thread: streams mic audio into _state.chunks."""

    def callback(indata, frames, time_info, status):
        if _state.active:
            with _state.lock:
                _state.chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=4096,
        callback=callback,
    ):
        while _state.active:
            time.sleep(0.05)


def start_recording():
    with _state.lock:
        _state.chunks = []
    _state.active = True
    _state.thread = threading.Thread(target=_record_worker, daemon=True)
    _state.thread.start()


def stop_recording_and_transcribe() -> str | None:
    _state.active = False
    if _state.thread:
        _state.thread.join(timeout=3)

    with _state.lock:
        chunks = list(_state.chunks)

    if not chunks:
        return None

    audio_np = np.concatenate(chunks, axis=0)
    buf = io.BytesIO()
    wav.write(buf, SAMPLE_RATE, audio_np)
    buf.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(buf) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)


# ── Text filtering ──────────────────────────────────────────────

def censor_word(word: str) -> str:
    clean = word.strip(".,!?;:'\"")
    if len(clean) <= 1:
        return "***"
    return clean[0] + "***"


def filter_text(text: str) -> tuple[str, list[str]]:
    words = text.split()
    filtered_words, blocked_found = [], []
    for word in words:
        clean = word.lower().strip(".,!?;:'\"")
        if profanity.contains_profanity(clean) or clean in NEGATIVE_WORDS:
            filtered_words.append(censor_word(word))
            blocked_found.append(word)
        else:
            filtered_words.append(word)
    return " ".join(filtered_words), blocked_found


# ── Page config ─────────────────────────────────────────────────
st.set_page_config(page_title="Speech Filter", page_icon="🎙️", layout="centered")

# ── Session state ───────────────────────────────────────────────
for key, default in [
    ("history", []),
    ("total_words", 0),
    ("total_blocked", 0),
    ("recording", False),
    ("start_time", None),
    ("dark_mode", True),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Theme tokens ────────────────────────────────────────────────
dm = st.session_state.dark_mode

if dm:
    bg              = "#0e1117"
    card_bg         = "#1e1e2e"
    text            = "#e0e0e0"
    subtext         = "#aaaaaa"
    border          = "#444444"
    badge_bg        = "#3d1a1a"
    badge_color     = "#ff6b6b"
    primary_bg      = "#1a3a1a"
    primary_text    = "#66bb6a"
    primary_border  = "#448844"
    primary_hover   = "#1e4a1e"
    danger_bg       = "#3a1a1a"
    danger_text     = "#ff6b6b"
    danger_border   = "#ff4444"
    danger_hover    = "#4a2020"
    toggle_bg       = "#4a4a4a66"
    toggle_border   = "#8a8a8a"
    toggle_text     = "#ffffff"
    toggle_label    = "Light Mode"
else:
    bg              = "#ffffff"
    card_bg         = "#f4f6fb"
    text            = "#1a1a2e"
    subtext         = "#555577"
    border          = "#d0d4e8"
    badge_bg        = "#ffe0e0"
    badge_color     = "#c0392b"
    primary_bg      = "#e8f5e9"
    primary_text    = "#1b5e20"
    primary_border  = "#66bb6a"
    primary_hover   = "#d0ead0"
    danger_bg       = "#fff0f0"
    danger_text     = "#b71c1c"
    danger_border   = "#e57373"
    danger_hover    = "#ffe0e0"
    toggle_bg       = "#6a6a6a66"
    toggle_border   = "#b0b0b0"
    toggle_text     = "#000000"
    toggle_label    = "Dark Mode"


# ── Global CSS ──────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, .stApp,
section[data-testid="stMain"] > div {{
    font-family: 'DM Sans', sans-serif !important;
    background-color: {bg} !important;
    color: {text} !important;
}}

[data-testid="stAppViewContainer"] .block-container {{
    padding-top: 0.25rem !important;
}}

.block-container {{
    padding-top: 0.25rem !important;
}}

.stApp h1, .stApp h2, .stApp h3 {{
    color: {text} !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}}
.stApp .stMarkdown p,
.stApp .stMarkdown li,
.stApp .stMarkdown strong {{
    color: {text} !important;
}}
[data-testid="stCaptionContainer"] p {{
    color: {subtext} !important;
}}

[data-testid="stMetricLabel"] p  {{ color: {subtext} !important; font-size: 0.85rem !important; }}
[data-testid="stMetricValue"]    {{ color: {text}    !important; font-weight: 700 !important; }}

hr {{ border-color: {border} !important; opacity: 1 !important; }}

[data-testid="stExpander"] summary {{
    background-color: {card_bg} !important;
    color: {text} !important;
    border-radius: 8px !important;
    border: 1px solid {border} !important;
    font-weight: 600 !important;
}}
[data-testid="stExpander"] summary:hover {{
    border-color: {primary_border} !important;
}}
[data-testid="stExpander"] > div:last-child {{
    background-color: {card_bg} !important;
    border: 1px solid {border} !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}}

/* ── ALL buttons — shared base ── */
.stButton > button {{
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.2rem !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
    cursor: pointer !important;
}}

/* Primary buttons (Start / Stop) — green */
.stButton > button[kind="primary"] {{
    background-color: {primary_bg}    !important;
    color:            {primary_text}  !important;
    border:           1.5px solid {primary_border} !important;
    box-shadow: none !important;
    outline: none !important;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primary"]:focus,
.stButton > button[kind="primary"]:active {{
    background-color: {primary_hover}   !important;
    color:            {primary_text}    !important;
    border-color:     {primary_border}  !important;
    outline: none !important;
    box-shadow: 0 0 0 2px {primary_border}44 !important;
}}

/* Secondary buttons */
.stButton > button[kind="secondary"] {{
    background-color: {toggle_bg}    !important;
    color:            #ffffff       !important;
    border:           1.5px solid {toggle_border} !important;
    box-shadow: none !important;
    outline: none !important;
}}
.stButton > button[kind="secondary"]:hover,
.stButton > button[kind="secondary"]:focus {{
    background-color: {toggle_bg}     !important;
    color:            #ffffff        !important;
    border-color:     {toggle_border} !important;
    outline: none !important;
}}

/* Theme toggle: lower, right-aligned, and isolated from secondary danger styles */
.st-key-theme_toggle {{
    display: flex !important;
    justify-content: flex-end !important;
    margin-top: 3.25rem !important;
    margin-bottom: 2rem !important;
    transform: translateX(260px);
}}

.st-key-theme_toggle button {{
    background-color: {toggle_bg} !important;
    color: {toggle_text} !important;
    border: 1.5px solid {toggle_border} !important;
    border-radius: 7px !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    min-height: 0 !important;
    line-height: 1.1 !important;
    padding: 0.42rem 0.75rem !important;
    letter-spacing: 0 !important;
    width: auto !important;
    min-width: 98px !important;
    white-space: nowrap !important;
    box-shadow: none !important;
}}

.st-key-theme_toggle button *,
.st-key-theme_toggle button p {{
    color: {toggle_text} !important;
}}

.st-key-theme_toggle button:hover,
.st-key-theme_toggle button:focus {{
    background-color: {toggle_bg} !important;
    color: {toggle_text} !important;
    border-color: {toggle_border} !important;
    outline: none !important;
    box-shadow: none !important;
}}

/* ── Result boxes ── */
.result-box {{
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 0.97rem;
    margin-bottom: 8px;
    border: 1px solid {border};
    background-color: {card_bg};
    color: {text};
    min-height: 48px;
    word-wrap: break-word;
    line-height: 1.75;
}}

/* ── Blocked badges ── */
.blocked-badge {{
    display: inline-block;
    background: {badge_bg};
    color: {badge_color};
    border-radius: 6px;
    padding: 3px 12px;
    margin: 2px;
    font-size: 0.85rem;
    font-weight: 600;
}}

/* ── Recording timer ── */
.timer-box {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #e53935;
    text-align: center;
    letter-spacing: 3px;
    padding: 10px 0;
}}
.rec-dot {{
    display: inline-block;
    width: 11px; height: 11px;
    border-radius: 50%;
    background: #e53935;
    margin-right: 8px;
    vertical-align: middle;
    animation: blink 1s infinite;
}}
@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.15; }}
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ── Theme toggle — right-aligned, pill-style Streamlit button ───
# Use columns to push the button to the right without floating/fixed positioning
_, toggle_col = st.columns([7, 3])
with toggle_col:
    if st.button(toggle_label, key="theme_toggle", type="secondary"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()


# ── Header ──────────────────────────────────────────────────────
st.title("Speech to Text")
st.caption("With Automated Content Filtering")
st.divider()

# ── Stats ────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Total Words", st.session_state.total_words)
c2.metric("Blocked Words", st.session_state.total_blocked)
pct = (
    round(
        (st.session_state.total_words - st.session_state.total_blocked)
        / st.session_state.total_words
        * 100
    )
    if st.session_state.total_words > 0
    else 100
)
c3.metric("Clean %", f"{pct}%")

st.divider()

# ── Recording UI ─────────────────────────────────────────────────
if not st.session_state.recording:
    if st.button("Start Recording", use_container_width=True, type="primary"):
        start_recording()
        st.session_state.recording = True
        st.session_state.start_time = time.time()
        st.rerun()

else:
    timer_placeholder = st.empty()
    stop_pressed = st.button("Stop and Process", use_container_width=True, type="primary")

    if not stop_pressed:
        for _ in range(3600):
            if not st.session_state.recording:
                break
            elapsed = int(time.time() - st.session_state.start_time)
            mins, secs = elapsed // 60, elapsed % 60
            timer_placeholder.markdown(
                f'<div class="timer-box">'
                f'<span class="rec-dot"></span> Recording &nbsp; {mins:02d}:{secs:02d}'
                f'</div>',
                unsafe_allow_html=True,
            )
            time.sleep(1)
    else:
        st.session_state.recording = False
        st.session_state.start_time = None

        with st.spinner("Processing your speech..."):
            try:
                text = stop_recording_and_transcribe()
                if text is None:
                    st.warning("No audio captured. Please try again.")
                    st.stop()
            except sr.UnknownValueError:
                st.warning("Could not understand the audio. Please speak clearly and try again.")
                st.stop()
            except sr.RequestError as e:
                st.error(f"API error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        filtered, blocked = filter_text(text)
        word_count = len(text.split())
        blocked_count = len(blocked)
        st.session_state.total_words += word_count
        st.session_state.total_blocked += blocked_count
        st.session_state.history.insert(0, {
            "original": text,
            "filtered": filtered,
            "blocked": blocked,
            "word_count": word_count,
            "blocked_count": blocked_count,
        })
        st.rerun()

# ── Results ──────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.subheader("Results")
    for i, entry in enumerate(st.session_state.history):
        label = (
            f"Recording {len(st.session_state.history) - i}  —  "
            f"{entry['word_count']} words, {entry['blocked_count']} blocked"
        )
        with st.expander(label, expanded=(i == 0)):
            st.markdown("**Original:**")
            st.markdown(
                f'<div class="result-box">{entry["original"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Filtered:**")
            st.markdown(
                f'<div class="result-box">{entry["filtered"]}</div>',
                unsafe_allow_html=True,
            )
            if entry["blocked"]:
                badges = " ".join(
                    f'<span class="blocked-badge">{w}</span>' for w in entry["blocked"]
                )
                st.markdown(f"**Blocked words:** {badges}", unsafe_allow_html=True)

    if st.button("Clear History", type="secondary"):
        st.session_state.history = []
        st.session_state.total_words = 0
        st.session_state.total_blocked = 0
        st.rerun()
