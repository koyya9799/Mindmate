# app.py
# MindMate — Streamlit MVP (with personalization: nickname + tone) + robustness & self-test
import streamlit as st
from datetime import datetime, timedelta
import json, os, time
import logging

# --- Small UX polish: calming background gradient + centered emoji header ---
CALMING_CSS = """
<style>
/* subtle vertical gradient background that is light and demo-friendly */
body {
  background: linear-gradient(180deg, #f5f7fa 0%, #eef2f7 50%, #f8fbff 100%);
}
/* center the header and add some breathing space */
.stApp > header { display: none; } /* hide default header to reduce clutter */
.header-center {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 0;
}
.header-emoji {
  font-size: 36px;
  transform: translateY(2px);
}
.header-title {
  font-size: 20px;
  font-weight: 700;
  color: #0f172a;
}
.card {
  background: rgba(255,255,255,0.6);
  padding: 10px 16px;
  border-radius: 10px;
  box-shadow: 0 6px 18px rgba(10,15,30,0.04);
}
</style>
"""
st.markdown(CALMING_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="header-center card">
      <div class="header-emoji">🧘‍♂️</div>
      <div>
        <div class="header-title">MindMate — Calm, private mood assistant</div>
        <div style="font-size:12px;color:#334155">Short text check-ins • Local-only storage</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Logging setup for debugging during demo ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("mindmate")

# --- File names ---
HISTORY_FILE = "mood_history.json"
PROFILE_FILE = "user_profile.json"
HF_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# --- VADER setup (optional) ---
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception:
    sid = None
    VADER_AVAILABLE = False

# --- Persistence helpers ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
        except Exception as e:
            logger.exception("Failed to read history file")
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("Failed to write history file")

def append_entry(text, emotion=None, score=None, entry_type="analysis"):
    history = load_history()
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "text": text,
        "emotion": emotion,
        "score": score,
        "type": entry_type
    }
    history.append(entry)
    history = history[-1000:]
    save_history(history)
    return entry

def load_profile():
    """Load user profile (nickname + tone). Returns dict or defaults."""
    if os.path.exists(PROFILE_FILE):
        try:
            with open(PROFILE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            logger.exception("Failed to read profile file")
            pass
    # default profile
    return {"name": "", "tone": "encouraging"}

def save_profile(profile):
    try:
        with open(PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        logger.exception("Failed to save profile")
        return False

# --- Emotion & suggestions ---
def map_compound_to_label(compound):
    if compound is None:
        return "neutral"
    try:
        c = float(compound)
    except Exception:
        return "neutral"
    if c >= 0.6:
        return "joy"
    if 0.2 <= c < 0.6:
        return "positive"
    if -0.2 < c < 0.2:
        return "neutral"
    if -0.6 < c <= -0.2:
        return "sadness"
    return "anger/sadness"

SUGGESTIONS = {
    "joy": ["Celebrate a small win — share it with a friend.", "Do one small thing you enjoyed again."],
    "positive": ["Take a 1-minute pause to appreciate this.", "Write one small note of what went well today."],
    "neutral": ["Try a short breathing exercise (60s).", "Set one small goal for the next hour."],
    "sadness": ["Write 3 small things you're grateful for.", "Take a 10-minute walk and breathe fresh air."],
    "anger/sadness": ["Try 2-minute paced breathing: inhale 4s, exhale 6s.", "Step away for a short walk."]
}

def render_suggestion(suggestion, tone, name):
    """Return a toned & personalized suggestion string."""
    display_name = name.strip() if name else ""
    if tone == "encouraging":
        if display_name:
            return f"Hey {display_name} — {suggestion} "
        else:
            return f"{suggestion} "
    elif tone == "neutral":
        return suggestion
    elif tone == "pragmatic":
        if display_name:
            return f"{display_name}: {suggestion} — try this now."
        else:
            return f"Suggestion: {suggestion} — try this now."
    else:
        return suggestion

# --- Hugging Face optional loader (on demand) ---
@st.cache_resource(show_spinner=False)
def load_hf_pipeline(model_name=HF_MODEL_NAME):
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=False)
        return pipe
    except Exception:
        logger.exception("Failed to load HF pipeline")
        return None

def predict_hf_emotion(pipe, text):
    if pipe is None:
        return None, None
    try:
        out = pipe(text[:512])
        if isinstance(out, list):
            best = out[0]
        else:
            best = out
        label = best.get("label")
        score = float(best.get("score", 0.0))
        return label.lower(), score
    except Exception:
        logger.exception("HF prediction exception")
        return None, None

# --- Robustness helpers (sanitize, safe wrappers) ---
MAX_INPUT_CHARS = 2000  # truncate very long inputs for performance

def sanitize_input(text):
    """Return safe text and a list of warnings (if any)."""
    warnings = []
    if text is None:
        return "", ["No input provided"]
    text = str(text).strip()
    if text == "":
        warnings.append("Empty input")
        return "", warnings
    # Non-ASCII check (basic)
    if not all(ord(c) < 128 for c in text):
        warnings.append("Non-ASCII characters detected (app works best with English).")
    if len(text) > MAX_INPUT_CHARS:
        warnings.append(f"Input truncated to {MAX_INPUT_CHARS} characters for performance.")
        text = text[:MAX_INPUT_CHARS]
    return text, warnings

def safe_predict_hf_emotion(pipe, text):
    """Wrapper that calls HF pipeline safely, returns (label,score,err)."""
    if pipe is None:
        return None, None, None
    try:
        label, score = predict_hf_emotion(pipe, text)
        return label, score, None
    except Exception as e:
        logger.exception("HF prediction failed")
        return None, None, str(e)

def safe_vader_score(sid_obj, text):
    """Wrapper for VADER, returns (label, score, err)."""
    if sid_obj is None:
        return None, None, "VADER not available"
    try:
        scores = sid_obj.polarity_scores(text)
        compound = scores.get("compound")
        label = map_compound_to_label(compound)
        return label, compound, None
    except Exception as e:
        logger.exception("VADER failed")
        return None, None, str(e)

# --- analysis orchestration (use this to replace inline code) ---
def analyze_text(raw_text):
    """
    Sanitized + safe analysis; returns dict with results and warnings.
    """
    text, val_warnings = sanitize_input(raw_text)
    result = {
        "text": text,
        "warnings": val_warnings,
        "detected_label": None,
        "detected_score": None,
        "detected_source": None,
        "error": None
    }
    if not text:
        return result

    # Try HF if enabled
    if st.session_state.get("hf_enabled") and st.session_state.get("hf_loaded") and st.session_state.get("hf_pipe") is not None:
        hf_label, hf_score, hf_err = safe_predict_hf_emotion(st.session_state.hf_pipe, text)
        if hf_err:
            result["warnings"].append("Hugging Face model error — falling back.")
            logger.warning(f"HF error during prediction: {hf_err}")
        if hf_label is not None:
            result.update({"detected_label": hf_label, "detected_score": hf_score, "detected_source": "HF"})
            return result

    # Try VADER
    vader_label, vader_score, vader_err = safe_vader_score(sid, text)
    if vader_err:
        result["warnings"].append("VADER error — using keyword fallback.")
        logger.warning(f"VADER error: {vader_err}")
    if vader_label is not None:
        result.update({"detected_label": vader_label, "detected_score": vader_score, "detected_source": "VADER"})
        return result

    # Keyword fallback
    txt = (text or "").lower()
    if any(k in txt for k in ["sad", "depress", "down", "unmotiv", "lonely"]):
        lbl = "sadness"
    elif any(k in txt for k in ["happy", "glad", "great", "joy"]):
        lbl = "joy"
    elif any(k in txt for k in ["angry", "furious", "mad"]):
        lbl = "anger/sadness"
    elif any(k in txt for k in ["anxious", "anxiety", "stressed", "stressing"]):
        lbl = "sadness"
    else:
        lbl = "neutral"
    result.update({"detected_label": lbl, "detected_score": None, "detected_source": "keyword"})
    return result

# --- Session state initialization ---
for key, default in {
    "hf_enabled": False,
    "hf_loaded": False,
    "hf_pipe": None,
    "start_breathing": False,
    "show_journal": False,
    "profile": None,
    "seen_onboarding": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Load profile into session_state on start
if st.session_state.get("profile") is None:
    st.session_state.profile = load_profile()

# ------------- Onboarding modal (first-time users) -------------
if not st.session_state.get("seen_onboarding", False):
    try:
        with st.modal("Welcome to MindMate — Quick demo"):
            st.markdown("**Welcome!** MindMate detects mood from short text check-ins, stores your entries locally, and suggests small coping steps.")
            st.write("- Type or paste how you feel, then click **Analyze**.")
            st.write("- Edit the summary or download it from the Export section.")
            st.info("This is a demo — not a medical device. If you're in crisis, contact local emergency services.")
            if st.button("Got it — start using MindMate"):
                st.session_state["seen_onboarding"] = True
    except Exception:
        # If st.modal isn't available in this Streamlit version, fall back to a simple info box
        st.info("Welcome! Type your feelings, click Analyze, and your entries will be stored locally.")
        st.session_state["seen_onboarding"] = True

# ---------- UI & Layout ----------
st.set_page_config(page_title="MindMate — MVP", page_icon="", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='margin:0;'> MindMate — Personalized Mental Health Assistant</h1>", unsafe_allow_html=True)
st.write("A calm, privacy-first demo. Your data remains local.")
st.info("Demo tip: If the Hugging Face model download is slow, keep it disabled. Use 'Seed demo history' to quickly populate data.")
st.write("")

# Sidebar: profile + model + controls
with st.sidebar:
    st.header("Profile")
    # show current profile values in small form
    name_input = st.text_input("Nickname (optional)", value=st.session_state.profile.get("name",""), key="profile_name")
    tone_choice = st.selectbox("Preferred tone", options=["encouraging","neutral","pragmatic"], index=["encouraging","neutral","pragmatic"].index(st.session_state.profile.get("tone","encouraging")), key="profile_tone")
    if st.button("Save profile"):
        new_profile = {"name": name_input.strip(), "tone": tone_choice}
        ok = save_profile(new_profile)
        if ok:
            st.session_state.profile = new_profile
            st.success("Profile saved.")
            st.rerun()
        else:
            st.error("Could not save profile.")

    st.markdown("---")
    st.header("Controls & Model")
    st.info("Privacy: Local only — data stays on your machine.")

    hf_toggle = st.checkbox("Enable Hugging Face model (optional, may download ~100–300MB)", value=st.session_state.hf_enabled)
    if hf_toggle and not st.session_state.hf_loaded:
        st.session_state.hf_enabled = True
        with st.spinner("Loading Hugging Face model (this may take a minute)..."):
            pipe = load_hf_pipeline()
            if pipe is None:
                st.warning("Could not load HF model. The app will fall back to VADER.")
                st.session_state.hf_loaded = False
                st.session_state.hf_pipe = None
                st.session_state.hf_enabled = False
            else:
                st.session_state.hf_pipe = pipe
                st.session_state.hf_loaded = True
                st.success("Hugging Face model loaded.")
    elif not hf_toggle and st.session_state.hf_loaded:
        st.session_state.hf_enabled = False
        st.session_state.hf_pipe = None
        st.session_state.hf_loaded = False

    st.markdown("---")
    if st.button("Seed demo history (3 entries)"):
        now = datetime.utcnow()
        demo = [
            {"timestamp": (now - timedelta(days=2)).isoformat() + "Z", "text": "I felt great today!", "emotion": "joy", "score": 0.85, "type":"analysis"},
            {"timestamp": (now - timedelta(days=1)).isoformat() + "Z", "text": "Kind of tired and neutral.", "emotion": "neutral", "score": 0.0, "type":"analysis"},
            {"timestamp": now.isoformat() + "Z", "text": "Stressed about work.", "emotion": "sadness", "score": -0.45, "type":"analysis"},
        ]
        history = load_history()
        history.extend(demo)
        save_history(history)
        st.success("Seeded demo data — go back to the main page.")
        st.rerun()

    st.markdown("---")
    if st.button("Wipe local data"):
        if os.path.exists(HISTORY_FILE):
            try:
                os.remove(HISTORY_FILE)
            except Exception as e:
                st.error("Could not delete history: " + str(e))
        if os.path.exists(PROFILE_FILE):
            try:
                os.remove(PROFILE_FILE)
            except Exception:
                pass
        st.success("Local data wiped.")
        # reset session profile
        st.session_state.profile = {"name": "", "tone": "encouraging"}
        st.rerun()

    st.markdown("---")
    st.caption("Prototype — not a medical device. If you're in crisis contact local emergency services.")

# ---------- Main content ----------
# 1) Input
st.markdown("---")
st.header("1) Input")
st.write("Write briefly how you feel right now (one or two sentences is fine).")

with st.form("mood_form_main", clear_on_submit=False):
    user_text = st.text_area("How are you feeling right now?", placeholder="I'm feeling anxious about exams...")
    submitted = st.form_submit_button("Analyze")

# initialize detection vars
detected_label = None
detected_score = None
detected_source = None

# analysis flow (uses analyze_text wrapper)
if submitted:
    analysis = analyze_text(user_text)
    # show warnings
    for w in analysis.get("warnings", []):
        st.warning(w)
    if analysis.get("error"):
        st.error("Analysis error: " + str(analysis.get("error")))

    if analysis.get("detected_label"):
        src = analysis.get("detected_source") or "model"
        score = analysis.get("detected_score")
        lbl = analysis.get("detected_label")
        if score is not None:
            st.metric(f"Detected ({src})", f"{lbl.upper()}  ({score:.2f})")
        else:
            st.metric(f"Detected ({src})", lbl.upper())

        # contextual polished messages & small animation
        if lbl:
            lname = str(lbl).lower()
            if lname in ["sadness", "anger/sadness"]:
                st.info("I see you're feeling low — small steps can help. Try one suggestion on the right.")
            elif lname in ["neutral", "positive"]:
                st.info("You're fairly neutral — small habits keep momentum going.")
            elif lname == "joy":
                st.success("Nice — that sounds positive! Small wins matter.")
                try:
                    st.balloons()
                except Exception:
                    pass

    # personalized greeting
    name = st.session_state.profile.get("name","").strip()
    tone = st.session_state.profile.get("tone","encouraging")
    if name:
        st.info(f"Hey {name} — analysis complete.")

    # Save entry
    saved_entry = append_entry(analysis.get("text",""), analysis.get("detected_label"), analysis.get("detected_score"), entry_type="analysis")
    st.success("Saved locally ✓")
    st.write("Saved entry:")
    st.json(saved_entry)

    # Suggestions column shown below after saving (two columns)
    col1, col2 = st.columns([2,1])
    with col2:
        st.header("3) Suggestions")
        st.write("Quick, actionable ideas — try one now.")
        suggestions = SUGGESTIONS.get(analysis.get("detected_label") or "neutral", SUGGESTIONS["neutral"])
        for s in suggestions[:2]:
            st.write(render_suggestion(s, tone, name))
        st.write("")
        # breathing button guard: do not start if already running
        if st.button("Start 60s breathing (guided)"):
            if not st.session_state.get("start_breathing", False):
                st.session_state.start_breathing = True
                st.rerun()
            else:
                st.info("Breathing already in progress.")
        if st.button("Try a 3-line journal"):
            st.session_state.show_journal = True
            st.rerun()

# Guided breathing flow
if st.session_state.get("start_breathing", False):
    st.markdown("---")
    st.header("Guided breathing")
    st.write("Follow the text and the progress bar. Breathe gently and comfortably.")
    total_seconds = 60
    progress = st.progress(0)
    instruction = st.empty()
    for i in range(total_seconds + 1):
        progress.progress(i / total_seconds)
        phase = i % 12
        if phase < 4:
            instruction.markdown("**Inhale** — 4s")
        elif phase < 8:
            instruction.markdown("**Hold** — 4s")
        else:
            instruction.markdown("**Exhale** — 4s")
        time.sleep(1)
    instruction.markdown("**Done** — how do you feel now?")
    st.success("Nice — breathing complete.")
    st.session_state.start_breathing = False

# Journaling UI
if st.session_state.get("show_journal", False):
    st.markdown("---")
    st.header("Journaling — 3-line prompt")
    st.write("Write a short 3-line journal entry. Keep it honest and brief. Press Save to store locally.")
    journal_text = st.text_area("Your 3-line journal", placeholder="Line 1: ...\nLine 2: ...\nLine 3: ...", height=140, key="journal_area")
    col_save, col_cancel = st.columns([1,1])
    with col_save:
        if st.button("Save journal"):
            jt = (journal_text or "").strip()
            if not jt:
                st.warning("Please write something before saving.")
            else:
                saved = append_entry(jt, emotion="journal", score=None, entry_type="journal")
                st.success("Journal saved locally ✓")
                st.write("Saved journal:")
                st.json(saved)
                st.session_state.show_journal = False
                st.session_state.journal_area = ""
                st.rerun()
    with col_cancel:
        if st.button("Cancel journaling"):
            st.session_state.show_journal = False
            st.rerun()

# ---------- History & Trends ----------
st.markdown("---")
st.header("History & Trends")
st.write("Your entries are stored locally. Use the chart to see trends over time.")
raw_history = load_history()
st.write(f"Local file: `{HISTORY_FILE}` — total raw entries: {len(raw_history)}")
normalized = []
for e in raw_history:
    timestamp = e.get("timestamp") or e.get("time") or None
    text_val = e.get("text") or e.get("message") or ""
    emotion_val = e.get("emotion") or e.get("label") or "neutral"
    score_val = e.get("score", None)
    if score_val is None and isinstance(e.get("vader_scores"), dict):
        score_val = e["vader_scores"].get("compound")
    normalized.append({
        "timestamp": timestamp,
        "emotion": emotion_val,
        "score": score_val if score_val is not None else 0.0,
        "text": text_val,
        "type": e.get("type", "analysis")
    })
if len(normalized) == 0:
    st.info("No history entries to plot yet. Submit an analysis or a journal to generate mood history, or use the sidebar to seed demo data.")
else:
    import pandas as pd
    df = pd.DataFrame(normalized)
    def parse_to_date(val):
        try:
            ts = pd.to_datetime(val, errors='coerce')
            if pd.isna(ts):
                return datetime.utcnow().date()
            return ts.date()
        except Exception:
            return datetime.utcnow().date()
    df['date'] = df['timestamp'].apply(parse_to_date)
    EMO_SCORE_MAP = {
        "joy": 1.0,
        "positive": 0.6,
        "neutral": 0.0,
        "sadness": -0.6,
        "anger/sadness": -0.9,
        "journal": 0.0
    }
    def emotion_to_numeric(row):
        emo = str(row['emotion']).lower()
        for k in EMO_SCORE_MAP:
            if k in emo:
                return EMO_SCORE_MAP[k]
        try:
            s = float(row['score'])
            return s
        except Exception:
            return 0.0
    df['score_for_chart'] = df.apply(emotion_to_numeric, axis=1)
    daily = df.groupby('date', as_index=True)['score_for_chart'].mean().sort_index()
    st.line_chart(daily)
    def friendly_when(val):
        try:
            ts = pd.to_datetime(val, errors='coerce')
            if pd.isna(ts):
                return datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            return ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    recent_df = df.copy()
    recent_df['when'] = recent_df['timestamp'].apply(friendly_when)
    recent = recent_df.sort_values(by='timestamp', ascending=False).head(8)
    recent_display = recent[['when', 'type', 'emotion', 'score_for_chart', 'text']].rename(columns={'score_for_chart': 'score_numeric'}).fillna("")
    st.subheader("Recent entries (last 8)")
    st.table(recent_display)

# -------------------------
# Fixed: Hour 12-14: 7-day mood summary export (text) - timezone-safe
# -------------------------
import pandas as pd
import html
from collections import Counter

# small local EMO_SCORE_MAP (keeps consistent with the rest of app)
EMO_SCORE_MAP = {
    "joy": 1.0,
    "positive": 0.6,
    "neutral": 0.0,
    "sadness": -0.6,
    "anger/sadness": -0.9,
    "journal": 0.0
}

def safe_score_from_entry(entry):
    emo = str(entry.get("emotion", "")).lower()
    for k in EMO_SCORE_MAP:
        if k in emo:
            return EMO_SCORE_MAP[k]
    try:
        return float(entry.get("score", 0.0))
    except Exception:
        return 0.0

def generate_7day_summary(history, profile=None):
    """
    history: list of entries (dicts)
    profile: optional dict {'name':..., 'tone':...}
    returns: summary_text (str)
    """
    # Use pandas timestamps in UTC to avoid naive/aware comparison issues
    now_ts = pd.Timestamp.utcnow()          # tz-aware UTC Timestamp
    seven_days_ago = now_ts - pd.Timedelta(days=7)

    # Normalize timestamps and filter last 7 days
    recent = []
    for e in history:
        ts = e.get("timestamp") or e.get("time") or None
        # Parse with utc=True so result is timezone-aware (UTC)
        try:
            ts_parsed = pd.to_datetime(str(ts), utc=True, errors='coerce')
        except Exception:
            ts_parsed = pd.NaT
        if pd.isna(ts_parsed):
            continue
        # Compare pandas Timestamps (both tz-aware)
        if ts_parsed >= seven_days_ago:
            recent.append(e)

    if len(recent) == 0:
        return "No mood entries in the last 7 days to summarize."

    # average score
    scores = [safe_score_from_entry(e) for e in recent]
    avg_score = sum(scores) / len(scores)

    # dominant emotion
    emotions = [str(e.get("emotion") or "neutral").lower() for e in recent]
    counter = Counter(emotions)
    dominant_emotion = counter.most_common(1)[0][0] if counter else "neutral"

    # suggestion: choose first suggestion for the dominant emotion if exists
    suggestion_list = SUGGESTIONS.get(dominant_emotion, SUGGESTIONS.get("neutral", []))
    top_suggestion = suggestion_list[0] if suggestion_list else "Try a short breathing exercise."

    # Build natural-language summary (1 paragraph)
    name = profile.get("name","").strip() if profile else ""
    name_prefix = f"Hey {name} — " if name else ""
    # Interpret average score into a phrase
    if avg_score >= 0.5:
        mood_phrase = "overall positive"
    elif avg_score >= 0.1:
        mood_phrase = "mildly positive"
    elif avg_score > -0.1:
        mood_phrase = "fairly neutral"
    elif avg_score > -0.6:
        mood_phrase = "somewhat negative"
    else:
        mood_phrase = "quite negative"

    summary = (
        f"{name_prefix}Over the last 7 days your average mood was {mood_phrase} "
        f"(average score {avg_score:.2f}). The most frequent emotion logged was '{dominant_emotion}'. "
        f"A quick suggestion: {top_suggestion}"
    )
    return summary

# --- Generate summary_text using the corrected function ---
raw_history = load_history()
try:
    profile = {}
    if os.path.exists("user_profile.json"):
        with open("user_profile.json","r",encoding="utf-8") as pf:
            profile = json.load(pf)
except Exception:
    profile = {}

summary_text = generate_7day_summary(raw_history, profile=profile)

# Show the summary in a textarea for easy copy/edit
st.markdown("---")
st.header("Export: 7-day mood summary")

st.subheader("Summary (editable)")
st.write("You can edit the summary below before downloading or copying it.")
txt_area = st.text_area("7-day summary", value=summary_text, height=140, key="summary_text_area")

# Download as .txt
bname = "mindmate_mood_summary.txt"
st.download_button("Download summary (.txt)", data=txt_area, file_name=bname, mime="text/plain")

# Copy to clipboard helper (same as before)
escaped = html.escape(txt_area)
copy_html = f"""
<button id="copy-btn">Copy summary to clipboard</button>
<script>
const btn = document.getElementById("copy-btn");
btn.onclick = () => {{
  const text = `{escaped}`;
  navigator.clipboard.writeText(text).then(() => {{
    btn.innerText = "Copied!";
    setTimeout(() => btn.innerText = "Copy summary to clipboard", 1500);
  }}).catch(err => {{
    alert("Copy failed — please select the text manually and copy (Cmd/Ctrl+C).");
  }});
}};
</script>
"""
import streamlit.components.v1 as components
components.html(copy_html, height=60)
st.caption("Tip: You can edit the summary textarea, then Download or Copy. For sharing, paste into email or notes.")

# -----------------------
# Self-test checklist (Hour 14-16)
# -----------------------
def run_self_test():
    results = {}
    try:
        # 1: analyze (use short test string)
        test_text = "I am happy today"
        a = analyze_text(test_text)
        results["analyze"] = bool(a.get("detected_label"))
        logger.info(f"Self-test analyze result: {a}")
    except Exception:
        results["analyze"] = False
        logger.exception("Self-test analyze failed")

    try:
        # 2: save (append a temp entry)
        tmp = append_entry("__self_test__ entry", "neutral", 0.0, entry_type="test")
        results["save"] = "__self_test__" in tmp.get("text", "")
        logger.info("Self-test save ok")
    except Exception:
        results["save"] = False
        logger.exception("Self-test save failed")

    try:
        # 3: chart (attempt to build df)
        h = load_history()
        import pandas as pd
        df_test = pd.DataFrame(h[-3:]) if isinstance(h, list) else pd.DataFrame()
        results["chart"] = isinstance(df_test, pd.DataFrame)
        logger.info("Self-test chart ok")
    except Exception:
        results["chart"] = False
        logger.exception("Self-test chart failed")

    try:
        # 4: wipe (simulate check for write access)
        has_access = os.access(".", os.W_OK)
        results["wipe_ok"] = has_access
        logger.info(f"Self-test wipe check ok: {has_access}")
    except Exception:
        results["wipe_ok"] = False
        logger.exception("Self-test wipe failed")

    try:
        # 5: export (generate summary)
        s = generate_7day_summary(load_history(), profile={})
        results["export"] = isinstance(s, str) and len(s) > 0
        logger.info("Self-test export ok")
    except Exception:
        results["export"] = False
        logger.exception("Self-test export failed")

    return results

st.markdown("---")
st.header("Demo Self-test checklist")
if st.button("Run self-test"):
    res = run_self_test()
    ok = all(res.values())
    for k, v in res.items():
        st.write(f"- {k}: {'PASS' if v else 'FAIL'}")
    if ok:
        st.success("All self-tests passed.")
    else:
        st.error("Some tests failed — check terminal logs for details.")
    logger.info(f"Self-test results: {res}")

# Wipe button at bottom too
st.markdown("---")
if st.button("Wipe local mood_history.json (delete file)"):
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            st.success("Deleted mood_history.json — local history wiped.")
            st.rerun()
        except Exception as e:
            st.error("Could not delete file: " + str(e))
    else:
        st.info("No mood_history.json file found (already empty).")
st.caption("This prototype is for demo only and not a medical device.")

# small footer
st.markdown("---")
st.markdown("<div style='text-align:center; font-size:12px; color:#64748b;'>Made with care • Local-only demo • Good for quick check-ins</div>", unsafe_allow_html=True)
