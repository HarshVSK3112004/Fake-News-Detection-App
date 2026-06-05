import streamlit as st
import os
import joblib
import pandas as pd
import io
import re
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("📰 Fake News Detection App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path      = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}"); st.stop()
if not os.path.exists(vectorizer_path):
    st.error(f"Vectorizer file not found at: {vectorizer_path}"); st.stop()

model      = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
st.success("Model & Vectorizer loaded successfully!")

# ── Auto-detect label mapping from model.classes_ ────────────────────────────
# Handles both (0=Real,1=Fake) and (0=Fake,1=Real) encodings automatically.
def get_label_map():
    """
    Returns a dict mapping integer class values to 'Real' or 'Fake'.
    Tries to detect from model.classes_ names; falls back to 1=Fake, 0=Real.
    """
    if not hasattr(model, "classes_"):
        return {0: "Real", 1: "Fake"}

    classes = list(model.classes_)

    # If classes are strings like 'FAKE', 'REAL', 'fake', 'real', etc.
    str_classes = [str(c).strip().lower() for c in classes]

    label_map = {}
    for idx, cls_str in enumerate(str_classes):
        if cls_str in ("fake", "1", "true"):          # 'true' = "true news" sometimes reversed
            label_map[classes[idx]] = "Fake"
        elif cls_str in ("real", "0", "false"):
            label_map[classes[idx]] = "Real"
        else:
            # Unknown string labels — use positional assumption: last class = Fake
            label_map[classes[idx]] = "Fake" if idx == len(classes) - 1 else "Real"

    return label_map

LABEL_MAP = get_label_map()

# Show debug info in sidebar
with st.sidebar:
    st.markdown("## 🔧 Model Info")
    if hasattr(model, "classes_"):
        st.write("**Model classes_:**", list(model.classes_))
    st.write("**Label mapping used:**", LABEL_MAP)
    st.caption("If predictions seem inverted, check that label mapping matches your training data.")

    st.markdown("---")
    st.markdown("### 🩺 Quick Sanity Check")
    if st.button("Run sanity check"):
        test_texts = [
            "Scientists confirm vaccines are safe and effective based on clinical trials.",
            "SHOCKING: Government puts microchips in water to control your mind!!!"
        ]
        for t in test_texts:
            v = vectorizer.transform([t])
            p = model.predict(v)[0]
            mapped = LABEL_MAP.get(p, f"Unknown({p})")
            nnz = v.nnz
            st.write(f"**Text:** _{t[:60]}..._")
            st.write(f"→ Raw pred: `{p}` → **{mapped}** | Non-zero features: `{nnz}`")
            st.markdown("---")


# ── Helpers ───────────────────────────────────────────────────────────────────

def predict_single(text: str):
    """
    Return (label_str, prob_real, prob_fake).
    Handles label inversion safely using LABEL_MAP.
    """
    vec  = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    label = LABEL_MAP.get(pred, "Unknown")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]   # aligned with model.classes_

        # Find which index corresponds to Real / Fake
        classes = list(model.classes_)
        prob_dict = {LABEL_MAP.get(c, str(c)): float(prob[i]) for i, c in enumerate(classes)}
        p_real = prob_dict.get("Real", 0.0)
        p_fake = prob_dict.get("Fake", 0.0)

        return label, p_real, p_fake

    return label, None, None


def predict_batch(texts):
    """
    Returns (labels_list, probs_array_or_None).
    probs_array columns are aligned to [P(Real), P(Fake)].
    """
    vecs  = vectorizer.transform(texts)
    preds = model.predict(vecs)
    labels = [LABEL_MAP.get(p, "Unknown") for p in preds]

    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(vecs)   # shape (n, n_classes)
        classes   = list(model.classes_)

        # Build aligned prob columns
        real_idx = next((i for i, c in enumerate(classes) if LABEL_MAP.get(c) == "Real"), None)
        fake_idx = next((i for i, c in enumerate(classes) if LABEL_MAP.get(c) == "Fake"), None)

        p_real = raw_probs[:, real_idx] if real_idx is not None else 1 - raw_probs[:, 0]
        p_fake = raw_probs[:, fake_idx] if fake_idx is not None else raw_probs[:, 0]

        return labels, p_real, p_fake

    return labels, None, None


def prob_bar_chart(prob_real: float, prob_fake: float):
    """Horizontal Plotly bar for Real / Fake probabilities."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[prob_real * 100], y=["Probability"],
        orientation="h", name="Real",
        marker_color="#22c55e",
        text=[f"Real {prob_real*100:.1f}%"], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=[prob_fake * 100], y=["Probability"],
        orientation="h", name="Fake",
        marker_color="#ef4444",
        text=[f"Fake {prob_fake*100:.1f}%"], textposition="inside",
    ))
    fig.update_layout(
        barmode="stack",
        height=100,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(range=[0, 100], showticklabels=False),
        yaxis=dict(showticklabels=False),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def extract_title_author(text: str):
    """
    Heuristic extraction of title and author from raw news text.

    Rules:
      - Title  : first non-empty line (≤ 200 chars) OR text before the first '.'
      - Author : looks for patterns like
          "By John Smith", "Author: Jane Doe", "Written by ...",
          "Reporter: ...", "- John Smith" at line start
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    # Title: first line, capped at 200 chars
    title = lines[0][:200] if lines else "—"

    # Author: scan all lines for common by-line patterns
    author_patterns = [
        r"^[Bb]y\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        r"^[Aa]uthor[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        r"^[Ww]ritten\s+[Bb]y\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        r"^[Rr]eporter[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        r"^[-–]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s*$",
    ]
    author = "—"
    full_text = "\n".join(lines)
    for pattern in author_patterns:
        m = re.search(pattern, full_text, re.MULTILINE)
        if m:
            author = m.group(1)
            break

    return title, author


def build_pdf_report(result_df: pd.DataFrame, summary: dict) -> bytes:
    """
    Build a simple PDF report using fpdf2.
    Falls back to a plain-text byte report if fpdf2 is unavailable.
    """
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Header
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Fake News Detection — Prediction Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.ln(4)

        # Summary box
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Summary", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for k, v in summary.items():
            pdf.cell(0, 6, f"  {k}: {v}", ln=True)
        pdf.ln(4)

        # Table header
        pdf.set_font("Helvetica", "B", 10)
        col_w = [10, 100, 20, 30]   # #, Text, Label, Confidence
        headers = ["#", "Text (truncated)", "Label", "Confidence (%)"]
        for w, h in zip(col_w, headers):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()

        # Table rows (max 200)
        pdf.set_font("Helvetica", "", 8)
        for i, row in result_df.head(200).iterrows():
            text_snippet = str(row.get("text_snippet", row.iloc[0]))[:80]
            label = str(row.get("Predicted", ""))
            conf  = str(row.get("Confidence (%)", "N/A"))
            pdf.cell(col_w[0], 6, str(i + 1), border=1)
            pdf.cell(col_w[1], 6, text_snippet, border=1)
            pdf.cell(col_w[2], 6, label, border=1)
            pdf.cell(col_w[3], 6, conf, border=1)
            pdf.ln()

        return pdf.output()

    except ImportError:
        # Fallback: plain-text report
        lines = ["Fake News Detection — Prediction Report",
                 f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "",
                 "=== Summary ==="]
        for k, v in summary.items():
            lines.append(f"{k}: {v}")
        lines += ["", "=== Predictions (first 200 rows) ===",
                  "\t".join(["#", "Predicted", "Confidence (%)"])]
        for i, row in result_df.head(200).iterrows():
            lines.append(f"{i+1}\t{row.get('Predicted','')}\t{row.get('Confidence (%)','N/A')}")
        return "\n".join(lines).encode()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Dataset Testing"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    text_input = st.text_area("Enter News Text", height=200)

    if st.button("Predict", key="single_predict"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            # ── Vectorizer sanity check ────────────────────────────────────
            vec_check = vectorizer.transform([text_input])
            if vec_check.nnz == 0:
                st.warning(
                    "⚠️ The vectorizer produced an all-zero vector for this text. "
                    "The vocabulary may not match your input language or preprocessing. "
                    "Prediction may be unreliable."
                )

            label, p_real, p_fake = predict_single(text_input)

            # ── Title / Author extraction ──────────────────────────────────
            title, author = extract_title_author(text_input)
            st.markdown("---")
            c1, c2 = st.columns(2)
            c1.markdown(f"**📌 Detected Title**\n\n> {title}")
            c2.markdown(f"**✍️ Detected Author**\n\n> {author}")
            st.markdown("---")

            # ── Verdict ────────────────────────────────────────────────────
            if label == "Fake":
                st.error("🚨 Fake News Detected")
            else:
                st.success("✅ Real News Detected")

            # ── Probability bar chart ──────────────────────────────────────
            if p_real is not None:
                st.markdown("**Fake / Real Probability**")
                st.plotly_chart(
                    prob_bar_chart(p_real, p_fake),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
                st.caption(f"Real: {p_real*100:.2f}%  |  Fake: {p_fake*100:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Dataset Testing
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Batch / Dataset Testing")
    st.markdown(
        "Upload a **CSV or Excel** file. Select the text column and optionally a "
        "true-label column (`0` = Real, `1` = Fake)."
    )

    uploaded_file = st.file_uploader(
        "Upload dataset", type=["csv", "xlsx", "xls"], key="dataset_upload"
    )

    if uploaded_file:
        try:
            df = (pd.read_excel(uploaded_file)
                  if uploaded_file.name.endswith((".xlsx", ".xls"))
                  else pd.read_csv(uploaded_file))
        except Exception as e:
            st.error(f"Could not read file: {e}"); st.stop()

        st.write(f"**Loaded {len(df):,} rows × {len(df.columns)} columns**")
        st.dataframe(df.head(5), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            text_col = st.selectbox("Select the **text / news** column",
                                    options=df.columns.tolist(), key="text_col")
        with col2:
            label_col = st.selectbox("Select the **true label** column (optional)",
                                     options=["(none)"] + df.columns.tolist(),
                                     key="label_col")

        if st.button("▶ Run Predictions on Dataset", key="batch_predict"):
            with st.spinner("Running predictions…"):
                texts  = df[text_col].fillna("").astype(str).tolist()
                labels, p_real_arr, p_fake_arr = predict_batch(texts)

                result_df = df.copy()
                result_df["Predicted"]    = labels
                result_df["text_snippet"] = df[text_col].fillna("").astype(str).str[:80]

                # Title / Author extraction per row
                extracted = df[text_col].fillna("").astype(str).apply(
                    lambda t: pd.Series(extract_title_author(t),
                                        index=["Extracted_Title", "Extracted_Author"])
                )
                result_df = pd.concat([result_df, extracted], axis=1)

                has_proba = p_real_arr is not None
                if has_proba:
                    result_df["Prob_Real (%)"]  = (p_real_arr * 100).round(2)
                    result_df["Prob_Fake (%)"]  = (p_fake_arr * 100).round(2)
                    import numpy as np
                    result_df["Confidence (%)"] = (np.maximum(p_real_arr, p_fake_arr) * 100).round(2)

            # ── Summary metrics ────────────────────────────────────────────
            n_fake = int((result_df["Predicted"] == "Fake").sum())
            n_real = int((result_df["Predicted"] == "Real").sum())
            total  = len(result_df)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Articles", f"{total:,}")
            m2.metric("🚨 Predicted Fake", f"{n_fake:,}", f"{n_fake/total*100:.1f}%")
            m3.metric("✅ Predicted Real", f"{n_real:,}", f"{n_real/total*100:.1f}%")

            # ── Probability Distribution Chart ─────────────────────────────
            if has_proba:
                st.markdown("---")
                st.subheader("📊 Fake / Real Probability Distribution")
                import plotly.express as px

                chart_df = pd.DataFrame({
                    "Prob_Fake (%)": result_df["Prob_Fake (%)"],
                    "Label": result_df["Predicted"],
                })
                fig_hist = px.histogram(
                    chart_df, x="Prob_Fake (%)", color="Label",
                    nbins=40, barmode="overlay",
                    color_discrete_map={"Fake": "#ef4444", "Real": "#22c55e"},
                    labels={"Prob_Fake (%)": "P(Fake) %", "count": "Articles"},
                    title="Distribution of Fake-News Probability Scores",
                    opacity=0.75,
                )
                fig_hist.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_hist, use_container_width=True)

                # Overall stacked bar (single-row summary)
                st.markdown("**Overall Fake vs Real Split**")
                st.plotly_chart(
                    prob_bar_chart(n_real / total, n_fake / total),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            # ── Accuracy (optional true labels) ───────────────────────────
            if label_col != "(none)":
                try:
                    from sklearn.metrics import classification_report, confusion_matrix

                    true_labels_raw = df[label_col].astype(str).str.strip().str.lower()

                    # Normalize true labels to "Real"/"Fake" strings
                    def normalize_true_label(v):
                        if v in ("1", "fake", "true"):
                            return "Fake"
                        elif v in ("0", "real", "false"):
                            return "Real"
                        return v  # pass through unknown

                    true_labels_str = true_labels_raw.apply(normalize_true_label)

                    pred_labels = result_df["Predicted"]
                    accuracy    = (true_labels_str.values == pred_labels.values).mean() * 100

                    report = classification_report(
                        true_labels_str, pred_labels,
                        labels=["Real", "Fake"],
                        target_names=["Real", "Fake"],
                        output_dict=True,
                        zero_division=0,
                    )
                    cm = confusion_matrix(true_labels_str, pred_labels, labels=["Real", "Fake"])

                    st.markdown("---")
                    st.subheader("📈 Evaluation Metrics")
                    a1, a2, a3, a4 = st.columns(4)
                    a1.metric("Accuracy",          f"{accuracy:.2f}%")
                    a2.metric("Precision (Fake)",  f"{report['Fake']['precision']*100:.2f}%")
                    a3.metric("Recall (Fake)",      f"{report['Fake']['recall']*100:.2f}%")
                    a4.metric("F1 (Fake)",          f"{report['Fake']['f1-score']*100:.2f}%")

                    cm_df = pd.DataFrame(cm,
                        index=["Actual Real", "Actual Fake"],
                        columns=["Pred Real", "Pred Fake"])
                    st.markdown("**Confusion Matrix**")
                    st.dataframe(cm_df, use_container_width=False)

                except Exception as e:
                    st.warning(f"Could not compute accuracy: {e}")

            # ── Results preview ────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🔎 Results Preview")

            filter_option = st.radio("Show", ["All", "Fake only", "Real only"],
                                     horizontal=True, key="filter_radio")
            preview_df = result_df.copy()
            if filter_option == "Fake only":
                preview_df = preview_df[preview_df["Predicted"] == "Fake"]
            elif filter_option == "Real only":
                preview_df = preview_df[preview_df["Predicted"] == "Real"]

            display_cols = [text_col, "Extracted_Title", "Extracted_Author", "Predicted"]
            if has_proba:
                display_cols += ["Prob_Real (%)", "Prob_Fake (%)", "Confidence (%)"]
            if label_col != "(none)":
                display_cols.append(label_col)

            st.dataframe(
                preview_df[display_cols].reset_index(drop=True),
                use_container_width=True, height=350,
            )

            # ── Downloads ─────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("⬇ Download Results")

            export_df = result_df.drop(columns=["text_snippet"], errors="ignore")

            # CSV
            csv_buf = io.StringIO()
            export_df.to_csv(csv_buf, index=False)

            dl1, dl2 = st.columns(2)

            with dl1:
                st.download_button(
                    label="📥 Download predictions as CSV",
                    data=csv_buf.getvalue(),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

            # PDF / text report
            summary = {
                "Total articles":  total,
                "Predicted Fake":  f"{n_fake} ({n_fake/total*100:.1f}%)",
                "Predicted Real":  f"{n_real} ({n_real/total*100:.1f}%)",
                "Generated":       datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            if label_col != "(none)":
                try:
                    summary["Accuracy"] = f"{accuracy:.2f}%"
                except Exception:
                    pass

            report_bytes = build_pdf_report(result_df, summary)

            try:
                from fpdf import FPDF
                report_name = "prediction_report.pdf"
                report_mime = "application/pdf"
            except ImportError:
                report_name = "prediction_report.txt"
                report_mime = "text/plain"

            with dl2:
                st.download_button(
                    label="📄 Download prediction report (PDF)",
                    data=report_bytes,
                    file_name=report_name,
                    mime=report_mime,
                )