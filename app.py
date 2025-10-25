# app.py
# MindMate — Streamlit MVP (with guided breathing + journaling)
import streamlit as st
from datetime import datetime, timedelta
import json, os, time

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

# --- History file ---
HISTORY_FILE = "mood_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
        except Exception:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

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

# ---------- UI & Layout ----------
st.set_page_config(
    page_title="MindMate — MVP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='margin:0;'>🧠 MindMate — Personalized Mental Health Assistant</h1>", unsafe_allow_html=True)
st.write("A calm, privacy-first demo that detects emotion from short text and suggests quick coping steps.")
st.write("")

# initialize journaling visibility in session state
if "show_journal" not in st.session_state:
    st.session_state.show_journal = False

# Sidebar: privacy, quick actions, and small profile
with st.sidebar:
    st.header("⚙️ Controls")
    st.info("Privacy: Local only — all data stays on your machine (`mood_history.json`).")
    if st.button("Wipe local data (delete mood_history.json)"):
        if os.path.exists(HISTORY_FILE):
            try:
                os.remove(HISTORY_FILE)
                st.success("Local data wiped.")
                st.experimental_rerun()
            except Exception as e:
                st.error("Could not delete: " + str(e))
        else:
            st.info("No local data found.")
    st.markdown("---")
    st.header("🔎 Demo helpers")
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
        st.experimental_rerun()
    st.markdown("---")
    st.subheader("Quick tools")
    if st.button("Start 60s breathing (sidebar)"):
        # small UX: set a flag so main UI shows breathing immediately
        st.session_state.start_breathing = True
        st.experimental_rerun()
    st.markdown("---")
    st.caption("Prototype — not a medical device. If you're in crisis contact local emergency services.")

# ---------- Main content ----------
# 1) Input
st.header("1) Input")
st.write("Write briefly how you feel right now (one or two sentences is fine).")

# Use a unique form key
with st.form("mood_form_main", clear_on_submit=False):
    user_text = st.text_area("How are you feeling right now?", placeholder="I'm feeling anxious about exams...")
    submitted = st.form_submit_button("Analyze")

# We'll only run analysis if user submitted
detected_label = None
vader_compound = None
if submitted:
    text = (user_text or "").strip()
    if not text:
        st.warning("Please type something before analyzing.")
    else:
        # 2) Analysis (left column) and 3) Suggestions (right column)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("2) Analysis")
            # VADER analysis if available
            if VADER_AVAILABLE and sid is not None:
                try:
                    scores = sid.polarity_scores(text)
                    vader_compound = scores.get("compound", 0.0)
                    detected_label = map_compound_to_label(vader_compound)
                    st.metric("Detected emotion", f"{detected_label.upper()}  ({vader_compound:.2f})")
                    st.write("Raw scores:", scores)
                except Exception as e:
                    detected_label = "neutral"
                    st.info("VADER analysis failed — using fallback.")
                    st.write("Analysis error:", str(e))
            else:
                # keyword fallback
                txt = text.lower()
                if any(k in txt for k in ["sad", "depress", "down", "unmotiv", "lonely"]):
                    detected_label = "sadness"
                elif any(k in txt for k in ["happy", "glad", "great", "joy"]):
                    detected_label = "joy"
                elif any(k in txt for k in ["angry", "furious", "mad"]):
                    detected_label = "anger/sadness"
                elif any(k in txt for k in ["anxious", "anxiety", "stressed", "stressing"]):
                    detected_label = "sadness"
                else:
                    detected_label = "neutral"
                st.metric("Detected emotion", detected_label.upper())

            # Save the entry locally once analysis done
            saved_entry = append_entry(text, detected_label, vader_compound, entry_type="analysis")
            st.success("Saved locally ✓")
            st.write("Saved entry:")
            st.json(saved_entry)

        with col2:
            st.header("3) Suggestions")
            st.write("Quick, actionable ideas — try one now.")
            suggestions = SUGGESTIONS.get(detected_label or "neutral", SUGGESTIONS["neutral"])
            for s in suggestions[:2]:
                st.write("• " + s)

            st.write("")  # spacing

            # Guided breathing button (main)
            if st.button("Start 60s breathing (guided)"):
                st.session_state.start_breathing = True
                # immediate rerun to start breathing flow
                st.experimental_rerun()

            # Journaling flow toggle
            if st.button("Try a 3-line journal"):
                st.session_state.show_journal = True
                st.experimental_rerun()

# If a breathing flag is set in session state, run guided breathing
if st.session_state.get("start_breathing", False):
    st.markdown("---")
    st.header("Guided breathing")
    st.write("Follow the text and the progress bar. Try to breathe slowly and comfortably.")

    # breathing parameters (60s demo)
    total_seconds = 60
    progress = st.progress(0)
    instruction = st.empty()

    # run the breathing loop (blocks the script for demo; acceptable for short demo)
    for i in range(total_seconds + 1):
        progress.progress(i / total_seconds)
        phase = i % 12
        # small, calming instructions
        if phase < 4:
            instruction.markdown("**Inhale** — 4s")
        elif phase < 8:
            instruction.markdown("**Hold** — 4s")
        else:
            instruction.markdown("**Exhale** — 4s")
        time.sleep(1)

    instruction.markdown("**Done** — how do you feel now?")
    st.success("Nice — breathing complete.")
    # clear the flag so it doesn't run again automatically
    st.session_state.start_breathing = False

# Journaling UI
if st.session_state.get("show_journal", False):
    st.markdown("---")
    st.header("Journaling — 3-line prompt")
    st.write("Write a short 3-line journal entry. Keep it honest and brief. Press Save to store locally.")
    # Provide a small text area with placeholder guiding 3 lines
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
                # hide journal after saving, clear textarea
                st.session_state.show_journal = False
                st.session_state.journal_area = ""
                st.experimental_rerun()
    with col_cancel:
        if st.button("Cancel journaling"):
            st.session_state.show_journal = False
            # keep text if they come back
            st.experimental_rerun()

# 4) History & Trends (always visible)
st.markdown("---")
st.header("4) History & Trends")
st.write("Your entries are stored locally. Use the chart to see trends over time.")

# Load raw history and normalize
raw_history = load_history()
st.write(f"Local file: `{HISTORY_FILE}` — total raw entries: {len(raw_history)}")

# Normalize entries
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
    st.write("No history entries to plot yet. Submit an analysis or a journal to generate mood history, or use the sidebar to seed demo data.")
else:
    import pandas as pd

    df = pd.DataFrame(normalized)

    # helper to convert various timestamp formats to python.date
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
        "journal": 0.0  # journals count as neutral for the chart
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

    # group by date and compute mean
    daily = df.groupby('date', as_index=True)['score_for_chart'].mean().sort_index()

    # show line chart
    st.line_chart(daily)

    # recent entries (last 8) — include entry type label
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

# Wipe local data button (redundant with sidebar but useful)
st.markdown("---")
if st.button("Wipe local mood_history.json (delete file)"):
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
            st.success("Deleted mood_history.json — local history wiped.")
            st.experimental_rerun()
        except Exception as e:
            st.error("Could not delete file: " + str(e))
    else:
        st.info("No mood_history.json file found (already empty).")

st.caption("This prototype is for demo only and not a medical device.")
