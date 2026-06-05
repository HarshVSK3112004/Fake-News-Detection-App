import streamlit as st
import os
import joblib
import pandas as pd
import io

# Page config
st.set_page_config(page_title="Fake News Detection", layout="wide")

st.title("📰 Fake News Detection App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

if not os.path.exists(vectorizer_path):
    st.error(f"Vectorizer file not found at: {vectorizer_path}")
    st.stop()

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

st.success("Model & Vectorizer loaded successfully!")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📊 Dataset Testing"])

# ── Tab 1 : Single Prediction (unchanged) ────────────────────────────────────
with tab1:
    text = st.text_area("Enter News Text", height=200)

    if st.button("Predict", key="single_predict"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            text_vec = vectorizer.transform([text])
            prediction = model.predict(text_vec)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(text_vec)[0]
                confidence = max(prob) * 100
            else:
                confidence = None

            if prediction == 1:
                st.error("🚨 Fake News Detected")
            else:
                st.success("✅ Real News Detected")

            if confidence is not None:
                st.info(f"Confidence Score: {confidence:.2f}%")

# ── Tab 2 : Dataset Testing ───────────────────────────────────────────────────
with tab2:
    st.subheader("Batch / Dataset Testing")
    st.markdown(
        "Upload a **CSV or Excel** file. Select the column that contains the news text, "
        "and (optionally) a column with true labels (`0` = Real, `1` = Fake). "
        "The app will run predictions on every row and let you download the results."
    )

    uploaded_file = st.file_uploader(
        "Upload dataset", type=["csv", "xlsx", "xls"], key="dataset_upload"
    )

    if uploaded_file:
        # ── Load file ──────────────────────────────────────────────────────
        try:
            if uploaded_file.name.endswith((".xlsx", ".xls")):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()

        st.write(f"**Loaded {len(df):,} rows × {len(df.columns)} columns**")
        st.dataframe(df.head(5), use_container_width=True)

        # ── Column selection ───────────────────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            text_col = st.selectbox(
                "Select the **text / news** column",
                options=df.columns.tolist(),
                key="text_col",
            )
        with col2:
            label_options = ["(none)"] + df.columns.tolist()
            label_col = st.selectbox(
                "Select the **true label** column (optional)",
                options=label_options,
                key="label_col",
            )

        # ── Run predictions ────────────────────────────────────────────────
        if st.button("▶ Run Predictions on Dataset", key="batch_predict"):
            with st.spinner("Running predictions…"):
                texts = df[text_col].fillna("").astype(str).tolist()
                vecs = vectorizer.transform(texts)
                preds = model.predict(vecs)

                pred_labels = ["Fake" if p == 1 else "Real" for p in preds]
                result_df = df.copy()
                result_df["Predicted"] = pred_labels
                result_df["Predicted_Code"] = preds

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vecs)
                    result_df["Confidence (%)"] = (probs.max(axis=1) * 100).round(2)

            # ── Summary metrics ────────────────────────────────────────────
            n_fake = int((preds == 1).sum())
            n_real = int((preds == 0).sum())
            total  = len(preds)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Articles", f"{total:,}")
            m2.metric("🚨 Predicted Fake", f"{n_fake:,}", f"{n_fake/total*100:.1f}%")
            m3.metric("✅ Predicted Real", f"{n_real:,}", f"{n_real/total*100:.1f}%")

            # ── Accuracy (if true labels provided) ─────────────────────────
            if label_col != "(none)":
                try:
                    true_labels = df[label_col].astype(int)
                    correct = (true_labels.values == preds).sum()
                    accuracy = correct / total * 100

                    from sklearn.metrics import classification_report, confusion_matrix
                    import numpy as np

                    st.markdown("---")
                    st.subheader("📈 Evaluation Metrics")

                    a1, a2, a3 = st.columns(3)
                    a1.metric("Accuracy", f"{accuracy:.2f}%")

                    report = classification_report(
                        true_labels, preds,
                        target_names=["Real", "Fake"],
                        output_dict=True,
                    )
                    a2.metric("Precision (Fake)", f"{report['Fake']['precision']*100:.2f}%")
                    a3.metric("Recall (Fake)",    f"{report['Fake']['recall']*100:.2f}%")

                    # Confusion matrix as a small table
                    cm = confusion_matrix(true_labels, preds)
                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual Real", "Actual Fake"],
                        columns=["Predicted Real", "Predicted Fake"],
                    )
                    st.markdown("**Confusion Matrix**")
                    st.dataframe(cm_df, use_container_width=False)

                except Exception as e:
                    st.warning(f"Could not compute accuracy: {e}")

            # ── Preview results ────────────────────────────────────────────
            st.markdown("---")
            st.subheader("🔎 Results Preview")

            # Filter widget
            filter_option = st.radio(
                "Show", ["All", "Fake only", "Real only"],
                horizontal=True, key="filter_radio"
            )
            preview_df = result_df.copy()
            if filter_option == "Fake only":
                preview_df = preview_df[preview_df["Predicted_Code"] == 1]
            elif filter_option == "Real only":
                preview_df = preview_df[preview_df["Predicted_Code"] == 0]

            display_cols = [text_col, "Predicted"]
            if "Confidence (%)" in preview_df.columns:
                display_cols.append("Confidence (%)")
            if label_col != "(none)":
                display_cols.append(label_col)

            st.dataframe(
                preview_df[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=350,
            )

            # ── Download ───────────────────────────────────────────────────
            st.markdown("---")
            st.subheader("⬇ Download Results")

            # Drop helper column before export
            export_df = result_df.drop(columns=["Predicted_Code"])

            csv_buf = io.StringIO()
            export_df.to_csv(csv_buf, index=False)

            st.download_button(
                label="Download predictions as CSV",
                data=csv_buf.getvalue(),
                file_name="predictions.csv",
                mime="text/csv",
            )