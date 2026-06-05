import streamlit as st
import os
import joblib

# Page config
st.set_page_config(page_title="Fake News Detection", layout="wide")

st.title("📰 Fake News Detection App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ FIX: files are in ROOT, not models/
model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

# Check files
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

if not os.path.exists(vectorizer_path):
    st.error(f"Vectorizer file not found at: {vectorizer_path}")
    st.stop()

# Load
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.success("Model & Vectorizer loaded successfully!")

# Input
text = st.text_area("Enter News Text")

# Predict
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]

        if prediction == 1:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News Detected")