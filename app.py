# app.py
import streamlit as st
from datetime import datetime
import json, os, time

# --- VADER setup ---
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception:
    sid = None
    VADER_AVAILABLE = False

# --- Storage ---
HISTORY_FILE = "mood_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# --- Mapping: compound -> simple emotion label ---
def map_compound_to_label(compound):
    """
    Map VADER compound score (-1..1) to simple labels.
    Tweak thresholds as needed.
    """
    if compound >= 0.6:
        return "joy"
    if 0.2 <= compound < 0.6:
        return "positive"
    if -0.2 < compound < 0.2:
        return "neutral"
    if -0.6 < compound <= -0.2:
        return "sadness"
    return "anger/sadness"

# --- Suggestions: short one-liners for quick demo ---
SUGGESTIONS = {
    "joy": [
        "Celebrate a small win — tell a friend or jot it down.",
        "Keep this momentum — do one action you enjoyed again."
    ],
    "positive": [
        "Nice — take a 1-minute pause to appreciate it.",
        "Try a short note of what you did well today."
    ],
    "neutral": [
        "Would you like a short breathing exercise (90s)?",
        "Set one small goal for the next hour."
    ],
    "sadness": [
        "Write down 3 small things you’re grateful for right now.",
        "Take a 10-minute walk and breathe fresh air."
    ],
    "anger/sadness": [
        "Try a 2-minute breathing cycle: 4s inhale, 6s exhale (repeat).",
        "Step away for a walk or splash cold water on your face."
    ]
}

# --- Streamlit UI ---
st.set_page_config(page_title="MindMate — MVP", page_icon="🧠")
st.title("MindMate — MVP — Emotion Label + Suggestions")

st.markdown("Type what you're feeling and press **Analyze**. The app shows a simple emotion label and two quick suggestions. Privacy: entries saved locally.")

with st.form("mood_form"):
    user_text = st.text_area("How are you feeling right now?", placeholder="I'm feeling anxious about exams...")
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not user_text.strip():
        st.warning("Please type something before analyzing.")
    else:
        # --- analyze with VADER if available ---
        if VADER_AVAILABLE and sid is not None:
            scores = sid.polarity_scores(user_text)
            compound = scores.get("compound", 0.0)
            label = map_compound_to_label(compound)
            st.subheader("Detected emotion")
            st.markdown(f"**{label.upper()}**  — (compound = {compound:.2f})")
            st.write("VADER raw scores:", scores)
        else:
            # very simple fallback: keyword mapping
            txt = user_text.lower()
            if any(k in txt for k in ["sad", "depress", "down", "unmotiv"]):
                label = "sadness"
            elif any(k in txt for k in ["happy", "glad", "great"]):
                label = "joy"
            elif any(k in txt for k in ["angry", "furious", "mad"]):
                label = "anger/sadness"
            else:
                label = "neutral"
            st.subheader("Detected emotion (fallback)")
            st.markdown(f"**{label.upper()}**")

        # --- show top 2 suggestions ---
        st.subheader("Top suggestions")
        suggestions = SUGGESTIONS.get(label, SUGGESTIONS["neutral"])
        # show only top 2
        for s in suggestions[:2]:
            st.write("- " + s)

        # --- Start breathing button ---
        st.markdown("**Quick intervention**")
        if st.button("Start 90s breathing guide"):
            # Simple guided breathing: show progress bar and pacing text
            progress = st.progress(0)
            instruction = st.empty()
            total_seconds = 90
            for i in range(total_seconds + 1):
                progress.progress(i / total_seconds)
                # cycle phases roughly: inhale(4s) hold(2s) exhale(6s) repeated
                phase = i % 12
                if phase < 4:
                    instruction.text(f"Inhale — 4s ({4-phase}s left)")
                elif phase < 6:
                    instruction.text("Hold — 2s")
                else:
                    instruction.text(f"Exhale — 6s ({12-phase}s left)")
                time.sleep(1)
            instruction.text("Done — how do you feel now?")
            st.success("Nice! Try analyzing again to see if your detected emotion changes.")

        # --- Save to local history ---
        history = load_history()
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "text": user_text,
            "label": label,
            "vader_scores": scores if VADER_AVAILABLE and sid is not None else None
        }
        history.append(entry)
        history = history[-200:]
        save_history(history)
        st.info("Saved this entry locally to mood_history.json")

# --- show recent history ---
st.markdown("---")
st.subheader("Recent entries")
history = load_history()
if history:
    import pandas as pd
    df = pd.DataFrame(history)
    df['date'] = pd.to_datetime(df['timestamp']).dt.strftime("%Y-%m-%d %H:%M")
    st.write(df[['date', 'label', 'text']].tail(8))
else:
    st.write("No history yet — submit an analysis to populate this list.")
