"""
Run this ONCE in your terminal:
    python diagnose.py

Paste the full output here so we can fix the app correctly.
"""
import joblib, os, numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model      = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

print("=" * 60)
print("MODEL TYPE       :", type(model).__name__)
print("MODEL CLASSES_   :", getattr(model, "classes_", "NOT FOUND"))

# Vocab check
vocab = getattr(vectorizer, "vocabulary_", {})
print("VOCAB SIZE       :", len(vocab))
print("VOCAB SAMPLE     :", list(vocab.keys())[:20])

# Test two sentences
tests = [
    "Scientists confirm vaccines are safe based on clinical trials.",
    "SHOCKING: Government controls your mind with secret chemicals!!!",
]
for t in tests:
    vec  = vectorizer.transform([t])
    pred = model.predict(vec)[0]
    nnz  = vec.nnz
    proba = model.predict_proba(vec)[0] if hasattr(model, "predict_proba") else "N/A"
    print("-" * 60)
    print("TEXT  :", t[:60])
    print("NNZ   :", nnz, "  ← should be > 0")
    print("PRED  :", pred)
    print("PROBA :", proba)

print("=" * 60)
print("TRAINING CLASS DISTRIBUTION (if available):")
if hasattr(model, "class_count_"):
    for cls, cnt in zip(model.classes_, model.class_count_):
        print(f"  Class {cls}: {int(cnt)} samples")