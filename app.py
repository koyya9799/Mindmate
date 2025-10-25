# app.py
import streamlit as st

st.set_page_config(page_title="MindMate — MVP", page_icon="🧠")
st.title("MindMate — MVP")
st.write("Type how you're feeling and press Analyze.")

# minimal input
text = st.text_input("How are you feeling right now?", placeholder="I'm feeling anxious about exams...")
if st.button("Analyze"):
    st.write("You typed:", text)
