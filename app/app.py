"""
app/app.py — Streamlit Web Application
────────────────────────────────────────────────────────────────────────────────
Fake News Detection Web UI

Run with:
    streamlit run app/app.py

Features:
  - Paste any article text for instant fake/real classification
  - Choose between fast (TF-IDF + LR) and accurate (BERT) mode
  - Shows confidence score and word-level explanation
  - Displays VADER sentiment breakdown
  - Example articles for quick testing
────────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import pickle
import logging

import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import clean_text, combine_title_text
from src.features import extract_sentiment_features, extract_linguistic_features
import pandas as pd

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shield | Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Fix for sidebar toggle button disappearing: Do NOT hide the whole header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Modern Typography & Smooth Scrolling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Premium Hero Section with Animations (Dark Mode Compatible) */
    .hero-container {
        padding: 4rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%);
        border-radius: 24px;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        animation: floatIn 1s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .main-title {
        font-size: 4rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
        color: var(--text-color) !important;
        margin-bottom: 1rem !important;
        line-height: 1.1;
    }
    .main-title span {
        background: linear-gradient(to right, #4f46e5, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-title {
        font-size: 1.25rem;
        color: var(--text-color);
        opacity: 0.8;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Model Selection Radio Buttons - Premium Makeover */
    div[role="radiogroup"] {
        background-color: transparent;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        margin-top: 1rem;
    }
    div[role="radiogroup"]:hover {
        border-color: #6366f1;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.2);
        transform: translateY(-2px);
    }
    div[role="radiogroup"] label {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        padding: 0.75rem 0.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    div[role="radiogroup"] label:hover {
        background-color: rgba(99, 102, 241, 0.05);
        transform: translateX(6px);
        color: #4f46e5 !important;
    }

    /* Verdict Cards with Glowing Accents */
    .verdict-card {
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        animation: scaleUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
        background-color: transparent;
        border: 1px solid #e2e8f0;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.05);
    }
    .verdict-fake {
        border-top: 8px solid #ef4444;
    }
    .verdict-real {
        border-top: 8px solid #10b981;
    }
    .verdict-uncertain {
        border-top: 8px solid #f59e0b;
    }
    .verdict-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .fake-text { color: #ef4444; }
    .real-text { color: #10b981; }
    .uncertain-text { color: #f59e0b; }
    .verdict-desc {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    /* Interactive Buttons - Enlarged */
    .stButton>button {
        border-radius: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        font-size: 1.1rem !important;
        padding: 1rem 2rem !important;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        border: none;
        padding: 1.2rem 2.5rem !important;
        color: white;
        font-size: 1.2rem !important;
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(79, 70, 229, 0.4);
    }
    

    /* Interactive Text Areas */
    .stTextArea textarea, .stTextInput input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        font-size: 1rem;
        background-color: transparent;
    }
    .stTextArea textarea {
        resize: none !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1);
    }

    /* Keyframe Animations */
    @keyframes floatIn {
        0% { opacity: 0; transform: translateY(30px) scale(0.98); }
        100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    @keyframes scaleUp {
        0% { opacity: 0; transform: scale(0.95) translateY(20px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }

    /* Metrics Refinement */
    [data-testid="stMetricValue"] {
        font-weight: 800;
        color: var(--text-color);
        font-size: 2.5rem;
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_fast_model():
    """Load TF-IDF vectorizer + Logistic Regression (fast mode)."""
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    try:
        with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        model = joblib.load(os.path.join(model_dir, "logisticregression_tfidf.pkl"))
        return vectorizer, model
    except FileNotFoundError:
        return None, None


@st.cache_resource
def load_bert_model():
    """Load fine-tuned DistilBERT (accurate mode)."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models", "bert_finetuned")
        if not os.path.exists(model_dir):
            return None, None, None
        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model     = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()
        return tokenizer, model, device
    except Exception:
        return None, None, None


# ── Prediction Functions ──────────────────────────────────────────────────────

def predict_fast(text: str, vectorizer, model):
    """TF-IDF + Logistic Regression prediction. Returns (label, prob_fake)."""
    cleaned  = clean_text(text)
    vec      = vectorizer.transform([cleaned])
    label    = model.predict(vec)[0]
    prob     = model.predict_proba(vec)[0]
    return label, prob[1]  # label (0=REAL,1=FAKE), P(FAKE)


def predict_bert(text: str, tokenizer, model, device):
    """DistilBERT prediction. Returns (label, prob_fake)."""
    import torch
    from torch.nn.functional import softmax

    cleaned  = clean_text(text)
    inputs   = tokenizer(
        cleaned, return_tensors="pt",
        max_length=256, padding="max_length",
        truncation=True
    )
    inputs   = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=-1).cpu().numpy()[0]
    label = int(probs.argmax())
    return label, probs[1]


def get_top_fake_words(text: str, vectorizer, model, n: int = 10):
    """
    Identify the top words in the input text that push toward FAKE classification.
    Uses Logistic Regression coefficients × TF-IDF values.
    """
    try:
        cleaned  = clean_text(text)
        vec      = vectorizer.transform([cleaned])
        coef     = model.coef_[0]
        tfidf_vals = vec.toarray()[0]
        scores   = coef * tfidf_vals
        feature_names = vectorizer.get_feature_names_out()

        top_fake_idx = np.argsort(scores)[-n:][::-1]
        top_real_idx = np.argsort(scores)[:n]

        fake_words = [(feature_names[i], float(scores[i])) for i in top_fake_idx if scores[i] > 0]
        real_words = [(feature_names[i], abs(float(scores[i]))) for i in top_real_idx if scores[i] < 0]
        return fake_words, real_words
    except Exception:
        return [], []


# ── UI ────────────────────────────────────────────────────────────────────────

# Hero Section
st.markdown("""
<div class="hero-container">
    <h1 class="main-title">Shield <span>| Fake News Detector</span></h1>
    <p class="sub-title">Advanced NLP & Machine Learning algorithms for instant misinformation detection.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>🛡️ Shield Settings</h2>", unsafe_allow_html=True)
    
    mode = st.radio(
        "🧠 Detection Engine",
        options=["⚡ Fast (TF-IDF + LR)", "🎯 Accurate (DistilBERT)"],
        help="Fast mode is instantaneous. Accurate mode requires the trained BERT model."
    )

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("📚 Quick Test Articles")
    st.write("Click a button below to load an example article into the text area.")
    example_fake = st.button("Load Fake News Example 🚨", use_container_width=True)
    example_real = st.button("Load Real News Example ✅", use_container_width=True)
    
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.caption("Developed for educational purposes. Always verify news with multiple sources.")

# Example texts
FAKE_EXAMPLE = """BREAKING: Scientists CONFIRM that 5G towers are spreading COVID-19!!! 
The mainstream media is HIDING the truth from you. George Soros is funding a secret 
agenda to inject microchips into every person through the COVID vaccine. Share this 
before it gets DELETED!!! Our government has been LYING to us for years. 
Wake up sheeple! The elite globalists are destroying our freedoms. 
This is NOT a drill — your family is in DANGER!"""

REAL_EXAMPLE = """The Federal Reserve raised interest rates by 0.25 percentage points 
on Wednesday, the tenth consecutive increase since March 2022, as policymakers 
continue their effort to bring inflation back toward their 2% target. 
Fed Chair Jerome Powell said in a news conference that the central bank remains 
committed to restoring price stability, noting that while inflation has eased 
from its peak, it remains well above the committee's long-run goal."""

# Main input area
st.markdown("### 📰 Input Article")
st.write("Paste the text of the article you want to verify below.")

default_text = ""
if example_fake:
    default_text = FAKE_EXAMPLE
elif example_real:
    default_text = REAL_EXAMPLE

col_input, col_stats = st.columns([3, 1])

with col_input:
    title_input = st.text_input("Headline (Optional):", placeholder="e.g., Politicians Caught in Scandal...")
    text_input  = st.text_area(
        "Article Content:",
        value=default_text,
        height=200,
        placeholder="Paste the full article text here...",
    )
    analyze_btn = st.button("🔍 Analyze Authenticity", type="primary", use_container_width=True)

with col_stats:
    if text_input:
        word_count = len(text_input.split())
        char_count = len(text_input)
        sent_count = len([s for s in text_input.split(".") if s.strip()])
        st.markdown("<div style='padding: 1rem; background: rgba(128,128,128,0.1); border-radius: 10px; height: 100%;'>", unsafe_allow_html=True)
        st.metric("Words", f"{word_count:,}")
        st.metric("Characters", f"{char_count:,}")
        st.metric("Sentences", f"{sent_count:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding: 2rem 1rem; text-align: center; background: rgba(128,128,128,0.1); border-radius: 10px; opacity: 0.7;'>
            <i>Waiting for text...</i>
        </div>
        """, unsafe_allow_html=True)

# Analysis Section
if analyze_btn:
    if not text_input or len(text_input.strip()) < 50:
        st.error("⚠️ Please enter at least 50 characters of text to analyze.")
    else:
        combined_text = combine_title_text(title_input, text_input)

        with st.spinner("🤖 Processing with AI models..."):
            # Load models
            vectorizer, lr_model = load_fast_model()

            if "DistilBERT" in mode:
                bert_tokenizer, bert_model, bert_device = load_bert_model()
                if bert_model is None:
                    st.warning("⚠️ BERT model not found. Running in Fast Mode (TF-IDF + LR) instead.")
                    mode = "⚡ Fast (TF-IDF + LR)"

            # Get prediction
            if vectorizer is None:
                st.error("❌ Fast model not found. Please train models first.")
                st.stop()

            if "DistilBERT" in mode and bert_model is not None:
                label, prob_fake = predict_bert(combined_text, bert_tokenizer, bert_model, bert_device)
                model_used = "DistilBERT"
            else:
                label, prob_fake = predict_fast(combined_text, vectorizer, lr_model)
                model_used = "Logistic Regression (TF-IDF)"

            prob_real = 1 - prob_fake

        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        
        # Determine verdict category
        if 0.35 < prob_fake < 0.65:
            certainty = "uncertain"
        elif label == 1:
            certainty = "fake"
        else:
            certainty = "real"

        # Organize results into tabs for a cleaner UI
        tab_verdict, tab_words, tab_sentiment, tab_linguistics = st.tabs([
            "🎯 Verdict & Confidence", 
            "🔑 Word Analysis", 
            "💬 Sentiment", 
            "📐 Linguistics"
        ])

        with tab_verdict:
            # ── Beautiful Verdict Card ──────────────────────────────────────────
            if certainty == "fake":
                st.markdown(f"""
                <div class="verdict-card verdict-fake">
                    <div class="verdict-title fake-text">🚨 Likely Fake News</div>
                    <div class="verdict-desc">Our AI detected significant patterns associated with misinformation.</div>
                    <p style="font-size: 1.2rem;">Confidence: <strong class="fake-text">{prob_fake*100:.1f}%</strong> probability</p>
                    <p style="color: #999; font-size: 0.9rem;">Powered by {model_used}</p>
                </div>""", unsafe_allow_html=True)
            elif certainty == "real":
                st.markdown(f"""
                <div class="verdict-card verdict-real">
                    <div class="verdict-title real-text">✅ Likely Real News</div>
                    <div class="verdict-desc">The content aligns with patterns found in factual reporting.</div>
                    <p style="font-size: 1.2rem;">Confidence: <strong class="real-text">{prob_real*100:.1f}%</strong> probability</p>
                    <p style="color: #999; font-size: 0.9rem;">Powered by {model_used}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-card verdict-uncertain">
                    <div class="verdict-title uncertain-text">⚠️ Uncertain</div>
                    <div class="verdict-desc">This article is borderline. It could be opinion, satire, or highly ambiguous.</div>
                    <p style="font-size: 1.2rem;">Fake Score: <strong class="uncertain-text">{prob_fake*100:.1f}%</strong></p>
                    <p style="color: #999; font-size: 0.9rem;">Powered by {model_used}</p>
                </div>""", unsafe_allow_html=True)

            # Visual charts side-by-side
            c1, c2 = st.columns(2)
            with c1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_fake * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Misinformation Probability", "font": {"size": 18, "family": "Inter"}},
                    number={"suffix": "%", "font": {"weight": "bold"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": "#ff4757" if label == 1 else "#2ed573"},
                        "steps": [
                            {"range": [0, 35],  "color": "#e6ffec"},
                            {"range": [35, 65], "color": "#fff7db"},
                            {"range": [65, 100],"color": "#fff5f5"},
                        ],
                        "threshold": {
                            "line": {"color": "#333", "width": 3},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), font=dict(family="Inter"))
                st.plotly_chart(fig_gauge, use_container_width=True)
            with c2:
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=["Factual", "Fake"],
                        y=[prob_real * 100, prob_fake * 100],
                        marker_color=["#2ed573", "#ff4757"],
                        text=[f"{prob_real*100:.1f}%", f"{prob_fake*100:.1f}%"],
                        textposition="auto",
                        textfont=dict(size=14, color="white", family="Inter", weight="bold")
                    )
                ])
                fig_bar.update_layout(
                    title=dict(text="Class Probabilities", font=dict(size=18, family="Inter")),
                    yaxis=dict(range=[0, 100], title="Probability (%)"),
                    height=280,
                    margin=dict(l=20, r=20, t=50, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter")
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab_words:
            if "DistilBERT" not in mode:
                st.markdown("### Model Interpretability")
                st.write("These are the most impactful words found in your text that heavily influenced the model's decision.")
                fake_words, real_words = get_top_fake_words(combined_text, vectorizer, lr_model)
                
                cw1, cw2 = st.columns(2)
                with cw1:
                    st.markdown("<div style='background: #fff5f5; padding: 1.5rem; border-radius: 12px; border-top: 4px solid #ff4757;'>", unsafe_allow_html=True)
                    st.markdown("<h4 style='color: #ff4757; margin-top: 0;'>🔴 Fake Indicators</h4>", unsafe_allow_html=True)
                    if fake_words:
                        for word, score in fake_words[:8]:
                            st.markdown(f"**`{word}`** <span style='float:right; color:#888;'>+{score:.2f}</span>", unsafe_allow_html=True)
                    else:
                        st.write("No strong fake indicators found.")
                    st.markdown("</div>", unsafe_allow_html=True)
                with cw2:
                    st.markdown("<div style='background: #f4fff8; padding: 1.5rem; border-radius: 12px; border-top: 4px solid #2ed573;'>", unsafe_allow_html=True)
                    st.markdown("<h4 style='color: #2ed573; margin-top: 0;'>🟢 Real Indicators</h4>", unsafe_allow_html=True)
                    if real_words:
                        for word, score in real_words[:8]:
                            st.markdown(f"**`{word}`** <span style='float:right; color:#888;'>+{score:.2f}</span>", unsafe_allow_html=True)
                    else:
                        st.write("No strong real indicators found.")
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("ℹ️ Word-level interpretability is currently only available in Fast Mode (Logistic Regression). BERT processes text as deep contextual embeddings rather than isolated words.")

        with tab_sentiment:
            st.markdown("### VADER Sentiment Breakdown")
            st.write("Fake news often uses highly charged, extreme emotional language (fear-mongering or overly positive hype).")
            
            sent_df  = extract_sentiment_features(pd.Series([text_input]))
            sent_row = sent_df.iloc[0]

            cs1, cs2, cs3, cs4 = st.columns(4)
            cs1.metric("Negative", f"{sent_row['vader_neg']:.2f}")
            cs2.metric("Neutral",  f"{sent_row['vader_neu']:.2f}")
            cs3.metric("Positive", f"{sent_row['vader_pos']:.2f}")
            cs4.metric("Compound (Overall)", f"{sent_row['vader_compound']:.2f}",
                         delta="Extreme" if abs(sent_row["vader_compound"]) > 0.6 else "Moderate",
                         delta_color="inverse")

            fig_sent = px.bar(
                x=["Negative", "Neutral", "Positive"],
                y=[sent_row["vader_neg"], sent_row["vader_neu"], sent_row["vader_pos"]],
                color=["Negative", "Neutral", "Positive"],
                color_discrete_map={"Negative": "#ff4757", "Neutral": "#a4b0be", "Positive": "#2ed573"},
            )
            fig_sent.update_layout(
                height=250, 
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter"),
                yaxis_title="Proportion"
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            if abs(sent_row["vader_compound"]) > 0.6:
                st.warning(f"⚠️ **High emotional intensity detected** (Score: {sent_row['vader_compound']:.2f}). Be cautious: sensationalism is often used to trigger emotional sharing.")

        with tab_linguistics:
            st.markdown("### Structural & Linguistic Traits")
            st.write("Compare the writing style of this article against typical journalistic standards.")
            
            ling_df  = extract_linguistic_features(pd.Series([combined_text]))
            ling_row = ling_df.iloc[0]

            cl1, cl2, cl3 = st.columns(3)
            with cl1:
                st.metric("Avg Word Length", f"{ling_row['avg_word_length']:.2f} chars")
                st.metric("Exclamation Marks (!)", ling_row["exclamation_count"])
            with cl2:
                st.metric("Unique Word Ratio", f"{ling_row['unique_word_ratio']:.3f}")
                st.metric("Question Marks (?)", ling_row["question_count"])
            with cl3:
                st.metric("Avg Sentence Length", f"{ling_row['avg_sentence_length']:.1f} words")
                st.metric("Capitalization Ratio", f"{ling_row['capital_ratio']:.4f}")
            
            st.caption("Note: Fake news often features excessive exclamation marks, ALL CAPS, and lower vocabulary diversity.")
