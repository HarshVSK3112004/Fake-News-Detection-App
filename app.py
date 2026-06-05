import streamlit as st
import os
import joblib

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="Fake News Detection",
    layout="wide"
)

st.title("📰 Fake News Detection App")

# ==========================
# Paths
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# ==========================
# Check files exist
# ==========================
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

if not os.path.exists(vectorizer_path):
    st.error(f"Vectorizer file not found at: {vectorizer_path}")
    st.stop()

# ==========================
# Load model + vectorizer
# ==========================
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.success("Model & Vectorizer loaded successfully!")

# ==========================
# Input
# ==========================
text = st.text_area("Enter News Text")

# ==========================
# Prediction
# ==========================
if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        # FIX: Convert text to numerical features
        text_vec = vectorizer.transform([text])

        prediction = model.predict(text_vec)[0]

        # Output
        if prediction == 1:
            st.error("🚨 Fake News Detected")
        else:
            st.success("✅ Real News Detected")