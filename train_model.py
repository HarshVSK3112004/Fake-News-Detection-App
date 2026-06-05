import os
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Load data ─────────────────────────────────────────────────────────────────
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 1   # 1 = Fake  ← FIXED (was 0)
true["label"] = 0   # 0 = Real  ← FIXED (was 1)

data = pd.concat([fake, true], ignore_index=True)

# Drop leaky/irrelevant columns
data = data.drop(columns=["subject", "date"], errors="ignore")

# ── Clean text ────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)   # ← FIXED regex
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Combine title + text for richer features
data["combined"] = (
    data["title"].fillna("") + " " + data["text"].fillna("")
)
data["combined"] = data["combined"].apply(clean_text)

X = data["combined"]
y = data["label"]

# ── Vectorize ─────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
)

X_vec = vectorizer.fit_transform(X)

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# ── Train ─────────────────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", n_jobs=-1)
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
print(classification_report(y_test, preds, target_names=["Real", "Fake"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model,      "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("Model & Vectorizer saved successfully.")