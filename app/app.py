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
    page_icon=":material/fact_check:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling — Altitude Design System ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;600&display=swap');

    /* ── Altitude Tokens ───────────────────────────────── */
    :root {
        --color-carbon-canvas:  #181818;
        --color-obsidian:       #111111;
        --color-graphite-card:  #1f1f1f;
        --color-slate-elevated: #262626;
        --color-iron-peak:      #323232;
        --color-bone:           #eeeeee;
        --color-ash:            #e4e4e4;
        --color-fog:            #a4a19b;
        --color-smoke:          #5e5d59;
        --color-pewter:         #4b4b4b;
        --color-voltage-blue:   #2b7fff;
        --color-mid-navy:       #1a365d;
        --font-serif:   'Libre Baskerville', 'Source Serif Pro', Georgia, serif;
        --font-inter:   'Inter', system-ui, -apple-system, sans-serif;
        --font-mono:    'Fira Code', 'JetBrains Mono', monospace;
        --shadow-md:    rgba(51,51,51,0.05) 0px 2px 15px 0px, rgba(51,51,51,0.05) 0px 1px 2px -1px;
        --shadow-card:  oklab(0 0 0 / 0.2) 0px 0px 0px 1px, rgba(51,51,51,0.05) 0px 2px 15px 0px;
    }

    /* ── Base ──────────────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}

    html, body, [class*="css"], .stApp {
        font-family: var(--font-inter) !important;
        background-color: var(--color-carbon-canvas) !important;
        color: var(--color-bone);
    }
    .stApp { background-color: var(--color-carbon-canvas) !important; }
    .main   { background-color: var(--color-carbon-canvas) !important; }
    [data-testid="stAppViewContainer"] { background-color: var(--color-carbon-canvas) !important; }
    [data-testid="stHeader"] { background-color: var(--color-carbon-canvas) !important; }

    /* ── Sidebar ───────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--color-obsidian) !important;
        border-right: 1px solid var(--color-graphite-card);
    }
    [data-testid="stSidebar"] * { color: var(--color-bone) !important; }
    [data-testid="stSidebar"] .stRadio label { font-family: var(--font-inter) !important; }
    [data-testid="stSidebar"] hr { border-color: var(--color-iron-peak) !important; }

    /* ── Hero ──────────────────────────────────────────── */
    .hero-container {
        padding: 48px 32px 40px;
        border-bottom: 1px solid var(--color-iron-peak);
        margin-bottom: 32px;
    }
    .hero-title {
        font-family: var(--font-serif);
        font-size: 48px;
        font-weight: 400;
        color: var(--color-bone);
        letter-spacing: -0.025em;
        line-height: 1.15;
        margin: 0 0 12px 0;
    }
    .hero-title em {
        color: var(--color-fog);
        font-style: italic;
    }
    .hero-subtitle {
        font-family: var(--font-inter);
        font-size: 16px;
        font-weight: 400;
        color: var(--color-fog);
        line-height: 1.5;
        max-width: 600px;
        margin: 0;
    }
    .ridge-svg {
        display: block;
        width: 100%;
        margin-top: 32px;
        opacity: 0.5;
    }

    /* ── Section labels ────────────────────────────────── */
    .section-label {
        font-family: var(--font-inter);
        font-size: 10px;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--color-fog);
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--color-iron-peak);
    }

    /* ── Streamlit headings → serif ────────────────────── */
    h3, .stMarkdown h3 {
        font-family: var(--font-serif) !important;
        font-size: 28px !important;
        font-weight: 400 !important;
        color: var(--color-bone) !important;
        letter-spacing: -0.7px !important;
        line-height: 1.38 !important;
        margin-bottom: 4px !important;
    }
    h4, .stMarkdown h4 {
        font-family: var(--font-inter) !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: var(--color-fog) !important;
    }

    /* ── Radio group ───────────────────────────────────── */
    div[role="radiogroup"] {
        background-color: var(--color-graphite-card);
        padding: 12px;
        border-radius: 4px;
        border: 1px solid var(--color-iron-peak);
        margin-top: 8px;
        transition: border-color 0.15s ease;
    }
    div[role="radiogroup"]:hover { border-color: var(--color-pewter); }
    div[role="radiogroup"] label {
        font-family: var(--font-inter) !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        color: var(--color-bone) !important;
        padding: 6px 8px;
    }

    /* ── Verdict cards ─────────────────────────────────── */
    .verdict-card {
        padding: 24px;
        border-radius: 8px;
        margin: 16px 0;
        background: var(--color-graphite-card);
        border: 1px solid var(--color-iron-peak);
        box-shadow: var(--shadow-md);
    }
    .verdict-fake      { border-top: 2px solid #8b3030; }
    .verdict-real      { border-top: 2px solid #2d6e4b; }
    .verdict-uncertain { border-top: 2px solid var(--color-smoke); }

    .verdict-title {
        font-family: var(--font-serif);
        font-size: 36px;
        font-weight: 400;
        letter-spacing: -0.9px;
        line-height: 1.15;
        margin-bottom: 8px;
    }
    .fake-text      { color: #c47c7c; }
    .real-text      { color: #6aad8a; }
    .uncertain-text { color: var(--color-fog); }
    .verdict-desc {
        font-family: var(--font-inter);
        font-size: 14px;
        font-weight: 400;
        color: var(--color-fog);
        line-height: 1.5;
        margin-bottom: 16px;
    }
    .verdict-prob {
        font-family: var(--font-inter);
        font-size: 14px;
        color: var(--color-bone);
        margin-bottom: 8px;
    }
    .verdict-model {
        font-family: var(--font-inter);
        font-size: 10px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--color-smoke);
    }

    /* ── Ghost buttons ─────────────────────────────────── */
    .stButton > button {
        font-family: var(--font-inter) !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border-radius: 4px !important;
        border: 1px solid var(--color-bone) !important;
        background: transparent !important;
        color: var(--color-bone) !important;
        padding: 8px 16px !important;
        letter-spacing: 0em;
        transition: background-color 0.12s ease, border-color 0.12s ease;
        box-shadow: var(--shadow-card);
    }
    .stButton > button:hover {
        background: var(--color-slate-elevated) !important;
        border-color: var(--color-ash) !important;
    }
    .stButton > button[kind="primary"] {
        border-color: var(--color-bone) !important;
    }

    /* ── Inputs ────────────────────────────────────────── */
    .stTextArea textarea, .stTextInput input {
        font-family: var(--font-inter) !important;
        font-size: 14px !important;
        background-color: var(--color-graphite-card) !important;
        border: 1px solid var(--color-iron-peak) !important;
        border-radius: 4px !important;
        color: var(--color-bone) !important;
        padding: 12px !important;
        transition: border-color 0.12s ease, box-shadow 0.12s ease;
    }
    .stTextArea textarea {
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
        letter-spacing: 0.025em;
        resize: none !important;
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: var(--color-smoke) !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--color-voltage-blue) !important;
        box-shadow: 0 0 0 2px rgba(43,127,255,0.15) !important;
        outline: none;
    }

    /* ── Tabs ──────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--color-iron-peak) !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-inter) !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        color: var(--color-fog) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 12px 20px !important;
        transition: color 0.12s ease;
    }
    .stTabs [aria-selected="true"] {
        color: var(--color-bone) !important;
        border-bottom: 1px solid var(--color-bone) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--color-bone) !important;
    }

    /* ── Metrics ───────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        font-family: var(--font-inter) !important;
        font-size: 28px !important;
        font-weight: 600 !important;
        color: var(--color-bone) !important;
        letter-spacing: -0.7px;
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--font-inter) !important;
        font-size: 10px !important;
        font-weight: 500 !important;
        color: var(--color-fog) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: var(--font-inter) !important;
        font-size: 11px !important;
    }

    /* ── Stat card ─────────────────────────────────────── */
    .stat-card {
        background: var(--color-graphite-card);
        border: 1px solid var(--color-iron-peak);
        border-radius: 8px;
        padding: 16px;
        height: 100%;
        box-shadow: var(--shadow-md);
    }
    .stat-empty {
        background: var(--color-graphite-card);
        border: 1px solid var(--color-iron-peak);
        border-radius: 8px;
        padding: 24px 16px;
        text-align: center;
        height: 100%;
        box-shadow: var(--shadow-md);
    }
    .stat-empty-text {
        font-family: var(--font-inter);
        font-size: 13px;
        color: var(--color-smoke);
        font-style: italic;
    }

    /* ── Indicator panels (word analysis) ──────────────── */
    .indicator-panel {
        background: var(--color-graphite-card);
        border: 1px solid var(--color-iron-peak);
        border-radius: 8px;
        padding: 16px;
        box-shadow: var(--shadow-md);
    }
    .indicator-fake { border-top: 2px solid #8b3030; }
    .indicator-real { border-top: 2px solid #2d6e4b; }
    .indicator-header {
        font-family: var(--font-inter);
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    .indicator-fake .indicator-header { color: #c47c7c; }
    .indicator-real .indicator-header { color: #6aad8a; }
    .indicator-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        border-bottom: 1px solid var(--color-iron-peak);
    }
    .indicator-row:last-child { border-bottom: none; }
    .indicator-word {
        font-family: var(--font-mono);
        font-size: 12px;
        color: var(--color-bone);
        letter-spacing: 0.05em;
    }
    .indicator-score {
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--color-fog);
    }
    .indicator-empty {
        font-family: var(--font-inter);
        font-size: 13px;
        color: var(--color-smoke);
    }

    /* ── Caption / helper text ─────────────────────────── */
    .stCaption, small {
        font-family: var(--font-inter) !important;
        font-size: 10px !important;
        color: var(--color-fog) !important;
        letter-spacing: 0.025em;
    }

    /* ── Divider ───────────────────────────────────────── */
    hr { border-color: var(--color-iron-peak) !important; margin: 24px 0 !important; }

    /* ── Alerts / info boxes ───────────────────────────── */
    [data-testid="stAlert"] {
        background: var(--color-graphite-card) !important;
        border: 1px solid var(--color-iron-peak) !important;
        border-radius: 4px !important;
        font-family: var(--font-inter) !important;
        font-size: 13px !important;
        color: var(--color-bone) !important;
    }
    /* Spinner */
    .stSpinner > div { border-top-color: var(--color-voltage-blue) !important; }
</style>""", unsafe_allow_html=True)


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



# ── Example Texts ──────────────────────────────────────────────────────────────
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

if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Shield &nbsp;<em>Fake News Detector</em></h1>
    <p class="hero-subtitle">Paste an article below. The model returns a classification with a confidence score, word-level explanation, and sentiment breakdown.</p>
    <svg class="ridge-svg" viewBox="0 0 1200 40" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <polyline points="0,36 80,30 160,20 220,32 310,10 380,26 460,6 540,22 620,4 700,18 780,12 860,28 940,8 1020,22 1100,14 1200,20"
                  fill="none" stroke="#a4a19b" stroke-width="1"/>
    </svg>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(
        "<p class='section-label' style='margin-top:8px;'>Shield</p>",
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Detection Engine",
        options=["Fast (TF-IDF + LR)", "Accurate (DistilBERT)"],
        help="Fast mode is instantaneous. Accurate mode requires the trained BERT model.",
        label_visibility="collapsed",
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>Test Articles</p>", unsafe_allow_html=True)
    st.caption("Load a sample into the text area.")
    if st.button("Load Fake News Sample", use_container_width=True):
        st.session_state["article_text"] = FAKE_EXAMPLE
    if st.button("Load Real News Sample", use_container_width=True):
        st.session_state["article_text"] = REAL_EXAMPLE
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Built for educational purposes. Verify news with multiple sources.")

# Main input area
st.markdown("<p class='section-label'>Input Article</p>", unsafe_allow_html=True)

col_input, col_stats = st.columns([3, 1])

with col_input:
    title_input = st.text_input("Headline (Optional):", placeholder="e.g., Politicians Caught in Scandal...")
    text_input  = st.text_area(
        "Article Content:",
        key="article_text",
        height=200,
        placeholder="Paste the full article text here...",
    )
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

with col_stats:
    if text_input:
        word_count = len(text_input.split())
        char_count = len(text_input)
        sent_count = len([s for s in text_input.split(".") if s.strip()])
        st.markdown("<div class='stat-card'>", unsafe_allow_html=True)
        st.metric("Words", f"{word_count:,}")
        st.metric("Characters", f"{char_count:,}")
        st.metric("Sentences", f"{sent_count:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="stat-empty">
            <p class="stat-empty-text">Waiting for input</p>
        </div>
        """, unsafe_allow_html=True)

# Analysis Section
if analyze_btn:
    if not text_input or len(text_input.strip()) < 50:
        st.error("Please enter at least 50 characters of text to analyze.")
    else:
        combined_text = combine_title_text(title_input, text_input)

        with st.spinner("Analyzing..."):
            # Load models
            vectorizer, lr_model = load_fast_model()

            if "DistilBERT" in mode:
                bert_tokenizer, bert_model, bert_device = load_bert_model()
                if bert_model is None:
                    st.warning("BERT model not found. Falling back to Fast Mode (TF-IDF + LR).")
                    mode = "Fast (TF-IDF + LR)"

            # Get prediction
            if vectorizer is None:
                st.error("Fast model not found. Please train the models first.")
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
            "Verdict",
            "Word Analysis",
            "Sentiment",
            "Linguistics"
        ])

        with tab_verdict:
            if certainty == "fake":
                st.markdown(f"""
                <div class="verdict-card verdict-fake">
                    <div class="verdict-title fake-text">Likely Fake</div>
                    <div class="verdict-desc">The model found patterns consistent with misinformation in this text.</div>
                    <p class="verdict-prob">Misinformation probability &nbsp;<strong class="fake-text">{prob_fake*100:.1f}%</strong></p>
                    <p class="verdict-model">Engine &mdash; {model_used}</p>
                </div>""", unsafe_allow_html=True)
            elif certainty == "real":
                st.markdown(f"""
                <div class="verdict-card verdict-real">
                    <div class="verdict-title real-text">Likely Factual</div>
                    <div class="verdict-desc">The text matches patterns found in factual news reporting.</div>
                    <p class="verdict-prob">Factual probability &nbsp;<strong class="real-text">{prob_real*100:.1f}%</strong></p>
                    <p class="verdict-model">Engine &mdash; {model_used}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-card verdict-uncertain">
                    <div class="verdict-title uncertain-text">Uncertain</div>
                    <div class="verdict-desc">Score falls in the ambiguous range. This may be opinion, satire, or mixed content.</div>
                    <p class="verdict-prob">Misinformation score &nbsp;<strong class="uncertain-text">{prob_fake*100:.1f}%</strong></p>
                    <p class="verdict-model">Engine &mdash; {model_used}</p>
                </div>""", unsafe_allow_html=True)

            # Charts
            c1, c2 = st.columns(2)
            _dark_layout = dict(
                paper_bgcolor="#1f1f1f",
                plot_bgcolor="#1f1f1f",
                font=dict(family="Inter", color="#eeeeee"),
                margin=dict(l=20, r=20, t=50, b=20),
                height=280,
            )
            with c1:
                _bar_color = "#8b3030" if label == 1 else "#2d6e4b"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_fake * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Misinformation Probability", "font": {"size": 13, "family": "Inter", "color": "#a4a19b"}},
                    number={"suffix": "%", "font": {"size": 36, "color": "#eeeeee"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#323232", "tickfont": {"color": "#a4a19b", "size": 10}},
                        "bar": {"color": _bar_color},
                        "bgcolor": "#1f1f1f",
                        "borderwidth": 1,
                        "bordercolor": "#323232",
                        "steps": [
                            {"range": [0, 35],   "color": "#1a2a1f"},
                            {"range": [35, 65],  "color": "#1f1f1f"},
                            {"range": [65, 100], "color": "#2a1a1a"},
                        ],
                        "threshold": {
                            "line": {"color": "#5e5d59", "width": 2},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                fig_gauge.update_layout(**_dark_layout)
                st.plotly_chart(fig_gauge, use_container_width=True)
            with c2:
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=["Factual", "Fake"],
                        y=[prob_real * 100, prob_fake * 100],
                        marker_color=["#2d6e4b", "#8b3030"],
                        marker_line_color="#323232",
                        marker_line_width=1,
                        text=[f"{prob_real*100:.1f}%", f"{prob_fake*100:.1f}%"],
                        textposition="auto",
                        textfont=dict(size=13, color="#eeeeee", family="Inter"),
                    )
                ])
                fig_bar.update_layout(
                    title=dict(text="Class Probabilities", font=dict(size=13, family="Inter", color="#a4a19b")),
                    yaxis=dict(range=[0, 100], title="Probability (%)", gridcolor="#262626", tickfont=dict(color="#a4a19b", size=10)),
                    xaxis=dict(tickfont=dict(color="#eeeeee", size=12)),
                    **_dark_layout,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        with tab_words:
            if "DistilBERT" not in mode:
                st.markdown("### Model Interpretability")
                st.caption("The words below most influenced the model's decision. Scored by TF-IDF weight times logistic regression coefficient.")
                fake_words, real_words = get_top_fake_words(combined_text, vectorizer, lr_model)

                cw1, cw2 = st.columns(2)
                with cw1:
                    rows_fake = "".join(
                        f"<div class='indicator-row'><span class='indicator-word'>{w}</span><span class='indicator-score'>+{s:.2f}</span></div>"
                        for w, s in fake_words[:8]
                    ) if fake_words else "<p class='indicator-empty'>No strong indicators found.</p>"
                    st.markdown(
                        f"<div class='indicator-panel indicator-fake'>"
                        f"<div class='indicator-header'>Fake Indicators</div>"
                        f"{rows_fake}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with cw2:
                    rows_real = "".join(
                        f"<div class='indicator-row'><span class='indicator-word'>{w}</span><span class='indicator-score'>+{s:.2f}</span></div>"
                        for w, s in real_words[:8]
                    ) if real_words else "<p class='indicator-empty'>No strong indicators found.</p>"
                    st.markdown(
                        f"<div class='indicator-panel indicator-real'>"
                        f"<div class='indicator-header'>Real Indicators</div>"
                        f"{rows_real}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Word-level explanation is only available in Fast Mode (Logistic Regression). DistilBERT uses contextual embeddings rather than individual token weights.")

        with tab_sentiment:
            st.markdown("### VADER Sentiment")
            st.caption("Fake news often uses highly charged emotional language. High compound scores in either direction warrant scrutiny.")
            
            sent_df  = extract_sentiment_features(pd.Series([text_input]))
            sent_row = sent_df.iloc[0]

            cs1, cs2, cs3, cs4 = st.columns(4)
            cs1.metric("Negative", f"{sent_row['vader_neg']:.2f}")
            cs2.metric("Neutral",  f"{sent_row['vader_neu']:.2f}")
            cs3.metric("Positive", f"{sent_row['vader_pos']:.2f}")
            cs4.metric("Compound (Overall)", f"{sent_row['vader_compound']:.2f}",
                         delta="Extreme" if abs(sent_row["vader_compound"]) > 0.6 else "Moderate",
                         delta_color="inverse")

            fig_sent = go.Figure(data=[
                go.Bar(
                    x=["Negative", "Neutral", "Positive"],
                    y=[sent_row["vader_neg"], sent_row["vader_neu"], sent_row["vader_pos"]],
                    marker_color=["#8b3030", "#5e5d59", "#2d6e4b"],
                    marker_line_color="#323232",
                    marker_line_width=1,
                    text=[f"{sent_row['vader_neg']:.2f}", f"{sent_row['vader_neu']:.2f}", f"{sent_row['vader_pos']:.2f}"],
                    textposition="auto",
                    textfont=dict(size=12, color="#eeeeee", family="Inter"),
                )
            ])
            fig_sent.update_layout(
                height=250,
                showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="#1f1f1f",
                plot_bgcolor="#1f1f1f",
                font=dict(family="Inter", color="#eeeeee"),
                yaxis=dict(title="Proportion", gridcolor="#262626", tickfont=dict(color="#a4a19b", size=10)),
                xaxis=dict(tickfont=dict(color="#eeeeee", size=12)),
            )
            st.plotly_chart(fig_sent, use_container_width=True)

            if abs(sent_row["vader_compound"]) > 0.6:
                st.warning(f"High emotional intensity detected (score: {sent_row['vader_compound']:.2f}). Sensationalist language is a common indicator in misinformation.")

        with tab_linguistics:
            st.markdown("### Linguistic Traits")
            st.caption("Structural signals. Fake news often shows excessive punctuation, ALL CAPS, and reduced vocabulary diversity.")

            ling_df  = extract_linguistic_features(pd.Series([combined_text]))
            ling_row = ling_df.iloc[0]

            cl1, cl2, cl3 = st.columns(3)
            with cl1:
                st.metric("Avg Word Length", f"{ling_row['avg_word_length']:.2f} ch")
                st.metric("Exclamation Marks", ling_row["exclamation_count"])
            with cl2:
                st.metric("Unique Word Ratio", f"{ling_row['unique_word_ratio']:.3f}")
                st.metric("Question Marks", ling_row["question_count"])
            with cl3:
                st.metric("Avg Sentence Length", f"{ling_row['avg_sentence_length']:.1f} w")
                st.metric("Capitalization Ratio", f"{ling_row['capital_ratio']:.4f}")
