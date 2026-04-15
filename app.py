import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sentiment Intelligence System", page_icon="🧠", layout="centered")

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        with open('complement_nb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf
    except FileNotFoundError:
        st.error("⚠️ Model or Vectorizer files missing! Ensure the .pkl files are in the same folder as app.py.")
        return None, None

# --- UI DESIGN ---
st.title("🧠 Sentiment Intelligence System")
st.subheader("Hybrid Machine Learning & Linguistic Analysis")
st.markdown("---")

if 'review_text' not in st.session_state:
    st.session_state.review_text = ''

def clear_review():
    st.session_state.review_text = ''

review_text = st.text_area(
    "Enter Customer Review:",
    value=st.session_state.review_text,
    key="review_text",
    placeholder="Type something like 'The delivery was not slow at all'...",
    height=150,
)
col1, col2 = st.columns([3, 1])
with col1:
    analyze_btn = st.button("🚀 Run Analysis")
with col2:
    st.button("🧹 Clear", on_click=clear_review)

if analyze_btn and review_text.strip():
    model, tfidf = load_assets()
    
    if model and tfidf:
        clean_text = review_text.lower()
        
        # 1. HEURISTIC SHIELD (The Manual Override for Demo Safety)
        manual_override = None
        # Specific patterns Naive Bayes struggles with due to statistical bias
        pos_patterns = ["not slow", "not bad", "not expensive", "not difficult", "not a waste"]
        neg_patterns = ["not good", "not great", "never buy", "waste of money", "too slow"]
        
        if any(p in clean_text for p in pos_patterns):
            manual_override = 1
        elif any(p in clean_text for p in neg_patterns):
            manual_override = 0

        # 2. PREPROCESS TEXT
        clean_text = review_text.lower()

        # 3. PATTERN OVERRIDE LOGIC
        if manual_override is not None:
            prediction = manual_override
            decision_type = "Heuristic Override (Pattern Match)"
            text_features = None
            expected_features = getattr(model, 'n_features_in_', None)
        else:
            text_vector = tfidf.transform([clean_text])
            expected_features = getattr(model, 'n_features_in_', None)
            text_features = text_vector.shape[1]

            if expected_features is not None and expected_features != text_features:
                st.error(f"The loaded model expects {expected_features} features, but TF-IDF only provides {text_features}.")
                st.error("Please use a model trained with TF-IDF-only input or update the model file.")
                prediction = None
                decision_type = "Feature mismatch"
            else:
                final_input = text_vector
                prediction = model.predict(final_input)[0]
                decision_type = "Hybrid Machine Learning (ComplementNB + TF-IDF)"

        # 5. DISPLAY RESULTS
        st.markdown("### Analysis Result:")
        if prediction is None:
            st.error("⚠️ Unable to predict because the model input shape did not match.")
        elif prediction == 1:
            st.success("😊 **POSITIVE SENTIMENT DETECTED**")
        else:
            st.error("😞 **NEGATIVE SENTIMENT DETECTED**")

        # 6. TECHNICAL DATA FOR THE PANEL
        with st.expander("📊 View Technical Decision Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Text Feature Count:** {text_features if text_features is not None else 'N/A'}")
                if expected_features is not None:
                    st.write(f"**Model Feature Count:** {expected_features}")
            with col2:
                st.write(f"**Manual Override Used:** {manual_override is not None}")
                st.write(f"**Decision Engine:** {decision_type}")
            if expected_features is not None and text_features is not None and expected_features != text_features:
                st.warning("Model and TF-IDF feature counts do not match. Re-train the model or load a TF-IDF-only model.")

st.markdown("---")
st.caption("Project: Optimized for E-Commerce & Product Feedback")