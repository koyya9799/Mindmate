# app.py
import streamlit as st
from datetime import datetime
import json, os

# --- VADER setup ---
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception as e:
    sid = None
    VADER_AVAILABLE = False

# --- Simple keyword fallback if VADER not available ---
KEYWORD_EMO_MAP = {
    "sad": "sadness",
    "depress": "sadness",
    "happy": "joy",
    "joy": "joy",
    "angry": "anger",
    "anxious": "fear",
    "anxiety": "fear",
    "stressed": "stress",
    "stress": "stress",
}

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

def map_compound_to_label(compound):
    # VADER compound -> simple label mapping for demo
    if compound >= 0.5:
        return "joy"
    if 0.1 <= compound < 0.5:
        return "positive"
    if -0.1 < compound < 0.1:
        return "neutral"
    if -0.5 < compound <= -0.1:
        return "sadness/concern"
    return "sadness/anger"

def keyword_fallback(text):
    txt = text.lower()
    for k, lab in KEYWORD_EMO_MAP.items():
        if k in txt:
            return lab
    return "neutral"

# --- Streamlit UI ---
st.set_page_config(page_title="MindMate — MVP", page_icon="🧠")
st.title("MindMate — MVP — Emotion Detection (VADER)")

st.markdown("""
Type how you're feeling and press **Analyze**.  
*This demo uses NLTK VADER (local). If VADER isn't available, a simple keyword fallback runs instead.*
""")

with st.form("mood_form"):
    user_text = st.text_area("How are you feeling right now?", placeholder="I'm feeling anxious about exams...")
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not user_text.strip():
        st.warning("Please type something before analyzing.")
    else:
        st.subheader("Raw analysis")

        # VADER path
        if VADER_AVAILABLE and sid is not None:
            scores = sid.polarity_scores(user_text)
            st.write("VADER scores:", scores)
            compound = scores.get("compound", 0.0)
            label = map_compound_to_label(compound)
            st.markdown(f"**Mapped label (from compound):** `{label}` (compound = {compound})")
        else:
            st.info("VADER not available — using simple keyword fallback.")
            label = keyword_fallback(user_text)
            scores = None
            st.markdown(f"**Fallback label:** `{label}`")

        # provide quick suggestions
        st.subheader("Suggestions (simple demo)")
        SUGGESTIONS = {
            "sadness": ["Try writing 3 things you are grateful for.", "Go for a short walk."],
            "joy": ["Celebrate a small win — share it with a friend!", "Keep a positive note for tomorrow."],
            "positive": ["Nice — keep your momentum.", "Consider a 1-minute breathing break."],
            "neutral": ["Would you like a short breathing exercise?", "Try writing one small goal for the day."],
            "sadness/concern": ["Try a grounding exercise: name 5 things you see.", "Consider journaling for 5 minutes."],
            "sadness/anger": ["Try a 2-minute breathing exercise: 4s inhale, 6s exhale.", "Take a short walk and come back later."]
        }
        # pick suggestions based on label (map similar labels)
        key = label
        if key not in SUGGESTIONS:
            if "sadness" in key:
                key = "sadness"
            elif "joy" in key or "positive" in key:
                key = "joy"
            else:
                key = "neutral"
        for s in SUGGESTIONS[key]:
            st.write("- " + s)

        # Save to history
        history = load_history()
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "text": user_text,
            "label": label,
            "vader_scores": scores
        }
        history.append(entry)
        history = history[-200:]
        save_history(history)
        st.success("Saved this entry locally to mood_history.json")

# Mood history visualization (simple)
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

# Helpful note about VADER compound thresholds
st.markdown("""
**VADER compound score guide (demo mapping):**
- `>= 0.5` → strong positive (joy)
- `0.1–0.5` → mildly positive
- `-0.1–0.1` → neutral
- `-0.5–-0.1` → mildly negative / concern
- `< -0.5` → strong negative (sadness / anger)

*This mapping is for demonstration; real mental-health deployment needs clinical validation.*
""")
