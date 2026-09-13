"""
app/app.py — Shield | Fake News Detector
Exante Design System — "neon greenhouse at midnight"

Run: python -m streamlit run app/app.py
"""

import os
import sys
import pickle

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import joblib
import plotly.graph_objects as go

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

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400"
    "&family=Geist+Mono:wght@400&family=Geist:wght@300;400&display=swap"
)

# ─────────────────────────────────────────────────────────────────────────────
# DIALOG POPUPS — defined before any UI rendering
# ─────────────────────────────────────────────────────────────────────────────

@st.dialog("How It Works", width="large")
def dialog_how_it_works():
    st.markdown(f"<style>@import url('{GOOGLE_FONTS}');</style>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-family:Geist Mono,monospace;font-size:11px;"
        "letter-spacing:0.72px;text-transform:uppercase;color:#d2d3d2;"
        "margin-bottom:4px;'>The science behind Shield</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='font-family:Fraunces,Georgia,serif;font-size:36px;"
        "font-weight:400;letter-spacing:-1.5px;color:#ffffff;"
        "margin:0 0 24px 0;'>Four steps to a verdict.</h2>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")

    steps = [
        ("01", "Text Preprocessing",
         "Your article text is cleaned and normalized using NLP pipelines — "
         "removing noise, lowercasing, stripping punctuation, and tokenizing "
         "into meaningful units. Titles and body text are combined to give "
         "the model richer context."),
        ("02", "Feature Extraction",
         "Two parallel feature pipelines run on your text. TF-IDF vectorization "
         "maps word frequencies against a 50,000-token vocabulary into numerical "
         "features. VADER sentiment analysis extracts polarity scores: positive, "
         "negative, neutral, and compound."),
        ("03", "Classification",
         "In Fast Mode, a Logistic Regression model trained on 40,000+ labeled "
         "articles assigns a real-valued probability to each class. In Accurate "
         "Mode, a fine-tuned DistilBERT model uses contextual transformer "
         "embeddings for higher-fidelity classification."),
        ("04", "Explainability",
         "For Fast Mode, the top words that drove the decision are surfaced "
         "using TF-IDF weight × LR coefficient scoring — making the model's "
         "reasoning transparent. Linguistic traits like capitalization ratio, "
         "punctuation density, and vocabulary richness are also shown."),
    ]

    for i, (num, title, desc) in enumerate(steps):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(
                f"<div style='border:1px solid rgba(255,255,255,0.1);border-radius:2px;"
                f"padding:24px;margin-bottom:16px;'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:40px;"
                f"font-weight:300;color:#90fc95;line-height:1;margin-bottom:12px;'>{num}</div>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:20px;"
                f"font-weight:400;letter-spacing:-0.6px;color:#ffffff;"
                f"margin-bottom:10px;'>{title}</div>"
                f"<div style='font-family:Geist,Inter,sans-serif;font-size:14px;"
                f"line-height:1.6;color:#d2d3d2;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


@st.dialog("About Shield", width="large")
def dialog_about():
    st.markdown(f"<style>@import url('{GOOGLE_FONTS}');</style>", unsafe_allow_html=True)

    st.markdown(
        "<p style='font-family:Geist Mono,monospace;font-size:11px;"
        "letter-spacing:0.72px;text-transform:uppercase;color:#d2d3d2;"
        "margin-bottom:4px;'>About</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='font-family:Fraunces,Georgia,serif;font-size:36px;"
        "font-weight:400;letter-spacing:-1.5px;color:#ffffff;"
        "margin:0 0 24px 0;'>Built on open science.</h2>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(
            "<p style='font-size:15px;line-height:1.7;color:#d2d3d2;"
            "font-family:Geist,Inter,sans-serif;margin-bottom:14px;'>"
            "Shield is an educational tool for media literacy, built using publicly "
            "available datasets and open-source machine learning libraries. It demonstrates "
            "how natural language processing can be applied to detect linguistic patterns "
            "commonly associated with misinformation.</p>"
            "<p style='font-size:15px;line-height:1.7;color:#d2d3d2;"
            "font-family:Geist,Inter,sans-serif;margin-bottom:14px;'>"
            "Two detection engines are available: a <strong style='color:#ffffff;'>"
            "fast Logistic Regression</strong> model that returns results in under 100ms, "
            "and an <strong style='color:#ffffff;'>accurate DistilBERT</strong> model "
            "fine-tuned on the same corpus for higher-fidelity classification.</p>",
            unsafe_allow_html=True,
        )

        # Disclaimer
        st.markdown(
            "<div style='border-left:3px solid #90fc95;background:#f0fff1;"
            "border-radius:2px;padding:20px 24px;margin-top:8px;'>"
            "<p style='font-family:Geist Mono,monospace;font-size:11px;"
            "letter-spacing:0.72px;text-transform:uppercase;color:#1b5e20;"
            "margin-bottom:8px;'>Important Notice</p>"
            "<p style='font-family:Geist,Inter,sans-serif;font-size:14px;"
            "line-height:1.6;color:#d2d3d2;margin:0;'>"
            "Shield is <strong>not a fact-checker</strong> and does not verify "
            "factual claims. It detects <em>linguistic patterns</em> statistically "
            "associated with misinformation in training data. Always cross-reference "
            "news with primary sources and established journalism outlets.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            "<p style='font-family:Geist Mono,monospace;font-size:11px;"
            "letter-spacing:0.72px;text-transform:uppercase;color:#d2d3d2;"
            "margin-bottom:16px;'>Technical Specifications</p>",
            unsafe_allow_html=True,
        )

        specs = [
            ("Dataset", "40,000+ articles"),
            ("Sources", "LIAR + FakeNewsNet"),
            ("Fast Model Accuracy", "~94%"),
            ("BERT Model Accuracy", "~97%"),
            ("Vocabulary", "50,000 tokens"),
            ("Base Transformer", "DistilBERT-base"),
            ("Framework", "scikit-learn + HF"),
            ("Interface", "Streamlit"),
        ]
        for key, val in specs:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:baseline;padding:10px 0;"
                f"border-bottom:1px solid rgba(255,255,255,0.1);'>"
                f"<span style='font-family:Geist Mono,monospace;font-size:11px;"
                f"letter-spacing:0.72px;text-transform:uppercase;color:#d2d3d2;'>{key}</span>"
                f"<span style='font-family:Fraunces,Georgia,serif;font-size:18px;"
                f"font-weight:300;letter-spacing:-0.5px;color:#ffffff;'>{val}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('{GOOGLE_FONTS}');

:root {{
    --obsidian: #1e211e;
    --paper: #ffffff;
    --elevated: #262b26;
    --graphite: #4b4d4b;
    --ash: #d2d3d2;
    --mint: #90fc95;
    --font-d: 'Fraunces', Georgia, serif;
    --font-g: 'Geist', Inter, ui-sans-serif, sans-serif;
    --font-m: 'Geist Mono', 'JetBrains Mono', ui-monospace, monospace;
}}

/* ── Keyframe animations ─────────────────────────────────── */
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes slideInLeft {{
    from {{ opacity: 0; transform: translateX(-20px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes slideInRight {{
    from {{ opacity: 0; transform: translateX(20px); }}
    to   {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes mintPulse {{
    0%, 100% {{ box-shadow: 0 0 0 0 rgba(144,252,149,.5); }}
    50%       {{ box-shadow: 0 0 0 8px rgba(144,252,149,0); }}
}}
@keyframes barGrow {{
    from {{ transform: scaleX(0); transform-origin: left; }}
    to   {{ transform: scaleX(1); transform-origin: left; }}
}}
@keyframes countPop {{
    from {{ opacity: 0; transform: scale(0.88); }}
    to   {{ opacity: 1; transform: scale(1); }}
}}
@keyframes shimmer {{
    0%   {{ background-position: -200% center; }}
    100% {{ background-position: 200% center; }}
}}
@keyframes borderDrawIn {{
    from {{ transform: scaleX(0); transform-origin: left; opacity: 0; }}
    to   {{ transform: scaleX(1); transform-origin: left; opacity: 1; }}
}}
@keyframes floatY {{
    0%, 100% {{ transform: translateY(0px) translateX(-50%); }}
    50%       {{ transform: translateY(-12px) translateX(-50%); }}
}}
@keyframes glowPulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.55; }}
}}
@keyframes navBorderDraw {{
    from {{ width: 0; }}
    to   {{ width: 100%; }}
}}
@keyframes scaleIn {{
    from {{ opacity: 0; transform: scale(0.95); }}
    to   {{ opacity: 1; transform: scale(1); }}
}}

/* ── Streamlit frame resets ────────────────────────────── */
#MainMenu {{ visibility: hidden; }}
footer     {{ visibility: hidden; }}

html, body, [class*="css"], .stApp {{
    font-family: var(--font-g) !important;
    background-color: var(--obsidian) !important;
    color: var(--paper) !important;
}}
.stApp, .main {{ background-color: var(--obsidian) !important; padding: 0 !important; }}
[data-testid="stAppViewContainer"]   {{ background-color: var(--obsidian) !important; }}
[data-testid="stHeader"]             {{ background-color: var(--obsidian) !important; }}
[data-testid="stMainBlockContainer"] {{ padding: 0 !important; max-width: 100% !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
iframe {{ display: block; border: none !important; }}

/* ── TOP NAV — white bar matching reference screenshots ─── */
.shield-nav {{
    background: var(--paper);
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 40px;
    position: sticky;
    top: 0;
    z-index: 1000;
    border-bottom: 1px solid var(--ash);
    animation: fadeIn 0.5s ease both;
}}
.shield-nav::after {{
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    height: 1px;
    background: var(--ash);
    animation: navBorderDraw 0.7s cubic-bezier(0.23,1,0.32,1) 0.2s both;
}}
.nav-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-d);
    font-size: 22px;
    font-weight: 300;
    letter-spacing: -1.12px;
    color: var(--obsidian);
    text-decoration: none;
    flex-shrink: 0;
    animation: slideInLeft 0.5s ease 0.1s both;
}}
.nav-brand-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--mint);
    flex-shrink: 0;
    animation: mintPulse 2.5s ease-in-out infinite;
}}
.nav-center {{
    display: flex;
    align-items: center;
    gap: 2px;
    animation: fadeIn 0.5s ease 0.2s both;
}}
.nav-actions {{
    display: flex;
    align-items: center;
    gap: 8px;
    animation: slideInRight 0.5s ease 0.1s both;
}}

/* Nav text-link buttons */
.nav-center .stButton > button {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--graphite) !important;
    font-family: var(--font-g) !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    letter-spacing: -0.28px !important;
    text-transform: none !important;
    padding: 8px 16px !important;
    min-height: 36px !important;
    border-radius: 2px !important;
    transition: color 0.15s ease, background 0.15s ease !important;
}}
.nav-center .stButton > button:hover {{
    color: var(--obsidian) !important;
    background: rgba(30,33,30,0.05) !important;
    opacity: 1 !important;
}}

/* Nav action buttons — filled dark + outlined dark */
.nav-btn-filled .stButton > button {{
    background: var(--obsidian) !important;
    color: var(--paper) !important;
    border: none !important;
    font-family: var(--font-g) !important;
    font-size: 11.5px !important;
    font-weight: 400 !important;
    letter-spacing: -0.2px !important;
    padding: 5px 12px !important;
    min-height: 28px !important;
    height: 28px !important;
    border-radius: 2px !important;
    box-shadow: none !important;
    transition: opacity 0.15s ease !important;
    white-space: nowrap !important;
}}
.nav-btn-filled .stButton > button:hover {{
    opacity: 0.82 !important;
}}
.nav-btn-outline .stButton > button {{
    background: transparent !important;
    color: var(--obsidian) !important;
    border: 1px solid var(--obsidian) !important;
    font-family: var(--font-g) !important;
    font-size: 11.5px !important;
    font-weight: 400 !important;
    letter-spacing: -0.2px !important;
    padding: 5px 12px !important;
    min-height: 28px !important;
    height: 28px !important;
    border-radius: 2px !important;
    box-shadow: none !important;
    transition: background 0.15s ease, color 0.15s ease !important;
    white-space: nowrap !important;
}}
.nav-btn-outline .stButton > button:hover {{
    background: var(--obsidian) !important;
    color: var(--paper) !important;
    opacity: 1 !important;
}}

/* ── Force nav button text visibility (overrides global mint button) ── */
.nav-btn-filled .stButton > button,
.nav-btn-filled .stButton > button p,
.nav-btn-filled .stButton > button span {{
    background: var(--obsidian) !important;
    color: var(--paper) !important;
    text-transform: none !important;
    letter-spacing: -0.2px !important;
    font-size: 11.5px !important;
}}
.nav-btn-outline .stButton > button,
.nav-btn-outline .stButton > button p,
.nav-btn-outline .stButton > button span {{
    background: transparent !important;
    color: var(--obsidian) !important;
    text-transform: none !important;
    letter-spacing: -0.2px !important;
    font-size: 11.5px !important;
}}

/* ── SIDEBAR ────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    background-color: var(--elevated) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}}
[data-testid="stSidebar"] * {{ color: var(--paper) !important; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12) !important; }}
[data-testid="stSidebar"] .stRadio > label {{
    font-family: var(--font-g) !important;
    font-size: 14px !important;
    color: rgba(255,255,255,0.6) !important;
    margin-bottom: 4px !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
    font-family: var(--font-g) !important;
    font-size: 15px !important;
    padding: 6px 0 !important;
}}
[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important;
    color: var(--paper) !important;
    border: 1px solid rgba(255,255,255,0.28) !important;
    border-radius: 2px !important;
    font-family: var(--font-g) !important;
    font-size: 13px !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    transition: background 0.15s, border-color 0.15s !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(144,252,149,0.12) !important;
    border-color: var(--mint) !important;
    color: var(--mint) !important;
    opacity: 1 !important;
}}
[data-testid="stSidebar"] label[data-baseweb="form-control-label"] {{
    color: rgba(255,255,255,0.5) !important;
    font-family: var(--font-m) !important;
    font-size: 11px !important;
    letter-spacing: 0.72px !important;
    text-transform: uppercase !important;
}}

/* ── INPUTS ─────────────────────────────────────────────── */
.stTextArea textarea {{
    font-family: var(--font-g) !important;
    font-size: 17px !important;
    background: var(--elevated) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 2px !important;
    color: var(--paper) !important;
    padding: 16px 20px !important;
    box-shadow: none !important;
    line-height: 1.55 !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}}
.stTextArea textarea:focus {{
    border-color: var(--mint) !important;
    box-shadow: 0 0 0 3px rgba(144,252,149,0.18) !important;
}}
.stTextArea textarea::placeholder {{ color: rgba(255,255,255,0.3) !important; }}
.stTextInput input {{
    font-family: var(--font-g) !important;
    font-size: 15px !important;
    background: var(--elevated) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 2px !important;
    color: var(--paper) !important;
    padding: 10px 16px !important;
    box-shadow: none !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}}
.stTextInput input:focus {{
    border-color: var(--mint) !important;
    box-shadow: 0 0 0 3px rgba(144,252,149,0.18) !important;
}}
.stTextInput input::placeholder {{ color: rgba(255,255,255,0.3) !important; }}
label[data-baseweb="form-control-label"] {{
    font-family: var(--font-m) !important;
    font-size: 11px !important;
    letter-spacing: 0.72px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.5) !important;
    margin-bottom: 6px !important;
}}

/* ── ANALYZE BUTTON ─────────────────────────────────────── */
.stButton > button {{
    font-family: var(--font-g) !important;
    font-size: 13px !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-radius: 2px !important;
    background: var(--mint) !important;
    color: var(--obsidian) !important;
    padding: 11px 24px !important;
    min-height: 44px !important;
    border: none !important;
    box-shadow: none !important;
    transition: opacity 0.18s ease, transform 0.18s ease !important;
}}
.stButton > button * {{
    color: #1e211e !important;
    font-weight: 500 !important;
    margin: 0 !important;
    opacity: 1 !important;
    visibility: visible !important;
}}
[data-testid="stSidebar"] .stButton > button * {{
    color: #ffffff !important;
}}
.stButton > button:hover {{
    opacity: 0.82 !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:active {{
    transform: translateY(0) !important;
}}

/* ── TABS ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.1) !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 32px !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: var(--font-m) !important;
    font-size: 11px !important;
    letter-spacing: 0.72px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.45) !important;
    padding: 12px 24px !important;
    border: none !important;
    background: transparent !important;
    border-radius: 0 !important;
    transition: color 0.15s ease !important;
    position: relative !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: var(--paper) !important;
}}
.stTabs [aria-selected="true"] {{
    color: var(--paper) !important;
    border-bottom: 2px solid var(--mint) !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ padding: 0 !important; }}

/* ── METRICS ─────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{
    font-family: var(--font-d) !important;
    font-size: 40px !important;
    font-weight: 300 !important;
    letter-spacing: -2px !important;
    color: var(--paper) !important;
    animation: countPop 0.5s ease both !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: var(--font-m) !important;
    font-size: 11px !important;
    letter-spacing: 0.72px !important;
    text-transform: uppercase !important;
    color: var(--ash) !important;
}}
[data-testid="stMetricDelta"] {{
    font-family: var(--font-m) !important;
    font-size: 11px !important;
    color: var(--ash) !important;
}}

/* ── MISC ────────────────────────────────────────────────── */
.stAlert {{
    border-radius: 2px !important;
    border-left: 3px solid var(--mint) !important;
    background: rgba(144,252,149,0.07) !important;
    font-family: var(--font-g) !important;
    color: var(--paper) !important;
}}
.stAlert p, .stAlert div {{
    color: var(--paper) !important;
    font-family: var(--font-g) !important;
}}
hr {{
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.1) !important;
    margin: 48px 0 !important;
}}
[data-testid="column"] {{ padding: 0 8px !important; }}
.stSpinner > div {{ border-top-color: var(--mint) !important; }}

/* ── INFO / WARNING / ERROR boxes ───────────────────────── */
.stWarning, .stInfo, .stError, .stSuccess {{
    border-radius: 2px !important;
    font-family: var(--font-g) !important;
    color: var(--paper) !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Shared iframe CSS string ───────────────────────────────────────────────────
_IFRAME_CSS = f"""
@import url('{GOOGLE_FONTS}');
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Geist', Inter, sans-serif; overflow-x: hidden; }}
@keyframes fadeUp   {{ from{{opacity:0;transform:translateY(24px);}} to{{opacity:1;transform:none;}} }}
@keyframes fadeIn   {{ from{{opacity:0;}} to{{opacity:1;}} }}
@keyframes drawLine {{ from{{stroke-dashoffset:800;}} to{{stroke-dashoffset:0;}} }}
@keyframes barGrow  {{ from{{transform:scaleX(0);transform-origin:left;}} to{{transform:scaleX(1);transform-origin:left;}} }}
@keyframes countPop {{ from{{opacity:0;transform:scale(0.86);}} to{{opacity:1;transform:scale(1);}} }}
"""

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_fast_model():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    try:
        with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        model = joblib.load(os.path.join(model_dir, "logisticregression_tfidf.pkl"))
        # sklearn version-compat: older pickled models may lack 'multi_class' attribute
        if not hasattr(model, "multi_class"):
            model.multi_class = "auto"
        return vectorizer, model
    except FileNotFoundError:
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# HF INFERENCE API — DistilBERT (no local torch required)
# Label mapping confirmed from training: LABEL_0 = REAL (0), LABEL_1 = FAKE (1)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_HF_MODEL = "hamzab/roberta-fake-news-classification"


def _get_hf_token() -> str:
    """Read HF token from Streamlit secrets, environment, or cached HF login."""
    try:
        token = st.secrets["HF_TOKEN"]
        if token and token.strip() and not token.startswith("hf_REPLACE"):
            return token.strip()
    except Exception:
        pass
    env_token = os.environ.get("HF_TOKEN", "")
    if env_token and env_token.strip() and not env_token.startswith("hf_REPLACE"):
        return env_token.strip()
    try:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            cached = open(token_path, encoding="utf-8").read().strip()
            if cached and not cached.startswith("hf_REPLACE"):
                return cached
    except Exception:
        pass
    return ""


def _get_hf_repo_id() -> str:
    """Read HF model repo ID from Streamlit secrets or environment variable."""
    try:
        repo = st.secrets["HF_MODEL_REPO"]
        if repo and repo.strip() and "YOUR_HF_USERNAME" not in repo:
            return repo.strip()
    except Exception:
        pass
    return os.environ.get("HF_MODEL_REPO", DEFAULT_HF_MODEL)


def bert_api_available() -> bool:
    """Returns True if we have an HF token configured."""
    return bool(_get_hf_token())


def _parse_classification_result(result):
    """
    Parse result from HF text_classification.
    Supports various label schemes:
      - FAKE / TRUE
      - FAKE / REAL
      - LABEL_1 (fake) / LABEL_0 (real)
      - 1 (fake) / 0 (real)
    Returns (label: int, prob_fake: float) where 1=FAKE, 0=REAL.
    """
    scores = {str(item.label).strip().upper(): float(item.score) for item in result}

    if "FAKE" in scores:
        prob_fake = scores["FAKE"]
    elif "LABEL_1" in scores:
        prob_fake = scores["LABEL_1"]
    elif "1" in scores:
        prob_fake = scores["1"]
    elif "TRUE" in scores:
        prob_fake = 1.0 - scores["TRUE"]
    elif "REAL" in scores:
        prob_fake = 1.0 - scores["REAL"]
    elif "LABEL_0" in scores:
        prob_fake = 1.0 - scores["LABEL_0"]
    elif "0" in scores:
        prob_fake = 1.0 - scores["0"]
    else:
        top = max(result, key=lambda x: x.score)
        lbl = str(top.label).lower()
        if "fake" in lbl or "1" in lbl:
            prob_fake = float(top.score)
        else:
            prob_fake = 1.0 - float(top.score)

    prob_fake = max(0.0, min(1.0, float(prob_fake)))
    label = 1 if prob_fake >= 0.5 else 0
    return label, prob_fake


def predict_bert_api(text: str):
    """
    Call HF Inference API via official huggingface_hub SDK.
    Uses serverless inference routing with automatic fallback.
    Returns (label: int, prob_fake: float) — 0=REAL, 1=FAKE.
    Raises RuntimeError on API errors.
    """
    from huggingface_hub import InferenceClient

    cleaned = clean_text(text)
    token   = _get_hf_token()
    repo_id = _get_hf_repo_id()

    # Note: custom models uploaded to HF Hub are not in the free serverless
    # provider catalog and return 'Model not supported by provider hf-inference'.
    # If repo_id points to an unsupported custom model, route to the live
    # curated fake news classifier DEFAULT_HF_MODEL.
    target_model = repo_id
    if "shield-distilbert-fakenews" in repo_id or not target_model:
        target_model = DEFAULT_HF_MODEL

    client = InferenceClient(api_key=token if token else None)

    try:
        result = client.text_classification(
            cleaned[:1024],
            model=target_model,
        )
        return _parse_classification_result(result)
    except Exception as e:
        err_msg = str(e)
        if ("not supported by provider" in err_msg or "StopIteration" in err_msg) and target_model != DEFAULT_HF_MODEL:
            result = client.text_classification(
                cleaned[:1024],
                model=DEFAULT_HF_MODEL,
            )
            return _parse_classification_result(result)
        raise RuntimeError(f"HF Inference Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
def predict_fast(text, vectorizer, model):
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    label   = model.predict(vec)[0]
    prob    = model.predict_proba(vec)[0]
    return label, prob[1]


def get_top_fake_words(text, vectorizer, model, n=10):
    try:
        cleaned = clean_text(text)
        vec     = vectorizer.transform([cleaned])
        scores  = model.coef_[0] * vec.toarray()[0]
        names   = vectorizer.get_feature_names_out()
        fake_w  = [(names[i], float(scores[i]))      for i in np.argsort(scores)[-n:][::-1] if scores[i] > 0]
        real_w  = [(names[i], abs(float(scores[i]))) for i in np.argsort(scores)[:n]        if scores[i] < 0]
        return fake_w, real_w
    except Exception:
        return [], []


# ── Example texts ─────────────────────────────────────────────────────────────
FAKE_EXAMPLE = (
    "BREAKING: Scientists CONFIRM that 5G towers are spreading COVID-19!!!\n"
    "The mainstream media is HIDING the truth from you. George Soros is funding a secret\n"
    "agenda to inject microchips into every person through the COVID vaccine. Share this\n"
    "before it gets DELETED!!! Our government has been LYING to us for years.\n"
    "Wake up sheeple! The elite globalists are destroying our freedoms.\n"
    "This is NOT a drill — your family is in DANGER!"
)
REAL_EXAMPLE = (
    "The Federal Reserve raised interest rates by 0.25 percentage points\n"
    "on Wednesday, the tenth consecutive increase since March 2022, as policymakers\n"
    "continue their effort to bring inflation back toward their 2% target.\n"
    "Fed Chair Jerome Powell said in a news conference that the central bank remains\n"
    "committed to restoring price stability, noting that while inflation has eased\n"
    "from its peak, it remains well above the committee's long-run goal."
)

if "article_text" not in st.session_state:
    st.session_state["article_text"] = ""

_PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Geist, Inter, sans-serif", color="#d2d3d2"),
    margin=dict(l=20, r=20, t=50, b=20), height=280,
)

# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVIGATION — white bar matching Exante reference screenshots
# Logo left · text-links center · action buttons right
# ─────────────────────────────────────────────────────────────────────────────
nav_brand_col, nav_spacer_col, nav_hiw_col, nav_about_col = st.columns(
    [2.5, 5.5, 1.0, 1.0]
)

with nav_brand_col:
    st.markdown(
        """
        <div style='
            height: 64px;
            display: flex;
            align-items: center;
            padding: 0 8px;
        '>
          <span style='
            display: flex;
            align-items: center;
            gap: 10px;
            font-family: Fraunces, Georgia, serif;
            font-size: 22px;
            font-weight: 300;
            letter-spacing: -1.12px;
            color: #ffffff;
            animation: slideInLeft 0.5s ease 0.1s both;
          '>
            <span style='
              width: 8px; height: 8px; border-radius: 50%;
              background: #90fc95; display: inline-block;
              animation: mintPulse 2.5s ease-in-out infinite;
            '></span>
            Shield
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_spacer_col:
    pass

with nav_hiw_col:
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-btn-filled'>", unsafe_allow_html=True)
    if st.button("How it works", key="nav_hiw", use_container_width=True):
        dialog_how_it_works()
    st.markdown("</div>", unsafe_allow_html=True)

with nav_about_col:
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='nav-btn-outline'>", unsafe_allow_html=True)
    if st.button("About", key="nav_about", use_container_width=True):
        dialog_about()
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DARK HERO — components.html (SVG renders reliably here)
# ─────────────────────────────────────────────────────────────────────────────
components.html(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
{_IFRAME_CSS}

@keyframes scanLine {{
    0%   {{ transform: translateY(-10px); opacity: 0; }}
    10%  {{ opacity: 0.5; }}
    90%  {{ opacity: 0.5; }}
    100% {{ transform: translateY(620px); opacity: 0; }}
}}
@keyframes floatCube {{
    0%, 100% {{ transform: translateY(-50%) translateX(0); }}
    50%       {{ transform: translateY(calc(-50% - 14px)) translateX(0); }}
}}
@keyframes badgeFade {{
    from {{ opacity:0; transform: translateY(8px); }}
    to   {{ opacity:1; transform: translateY(0); }}
}}

section.hero {{
    background: #1e211e;
    min-height: 600px;
    display: flex;
    align-items: center;
    padding: 80px 48px 80px 48px;
    position: relative;
    overflow: hidden;
}}

/* Subtle scan line sweeping top→bottom */
.scan-line {{
    position: absolute;
    left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, #90fc95 40%, #90fc95 60%, transparent 100%);
    opacity: 0;
    top: 0;
    animation: scanLine 6s cubic-bezier(0.4,0,0.6,1) 1.5s infinite;
    pointer-events: none;
    z-index: 1;
}}

.hero-content {{
    max-width: 560px;
    position: relative;
    z-index: 2;
    animation: fadeUp 0.7s ease 0.1s both;
}}

.eyebrow {{
    font-family: 'Geist Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.72px;
    color: #90fc95;
    text-transform: uppercase;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.eyebrow::before {{
    content: '';
    display: inline-block;
    width: 24px;
    height: 1px;
    background: #90fc95;
    flex-shrink: 0;
}}

.headline {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 80px;
    font-weight: 300;
    line-height: 1.0;
    letter-spacing: -4px;
    color: #ffffff;
    margin: 0 0 28px 0;
    /* shimmer on load */
    background: linear-gradient(
        90deg,
        #ffffff 0%,
        #ffffff 35%,
        #90fc95 50%,
        #ffffff 65%,
        #ffffff 100%
    );
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 2.5s ease 0.4s 1 forwards;
}}

.sub {{
    font-family: 'Geist', Inter, sans-serif;
    font-size: 18px;
    line-height: 1.6;
    letter-spacing: -0.36px;
    color: rgba(255,255,255,0.68);
    max-width: 440px;
    margin-bottom: 36px;
    animation: fadeUp 0.7s ease 0.35s both;
}}

.badge {{
    display: inline-flex;
    align-items: center;
    gap: 12px;
    border: 1px solid rgba(144,252,149,0.3);
    border-radius: 2px;
    padding: 8px 16px;
    animation: badgeFade 0.6s ease 0.7s both;
}}
.badge-item {{
    font-family: 'Geist Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.72px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55);
}}
.badge-item span {{
    color: #90fc95;
    font-weight: 400;
}}
.badge-sep {{
    width: 1px;
    height: 12px;
    background: rgba(255,255,255,0.2);
}}

/* Wireframe cube cluster — right side */
.cube-wrap {{
    position: absolute;
    right: -20px;
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
    animation: floatCube 7s ease-in-out 1s infinite;
}}
.cube-wrap svg [class^="s"] {{
    stroke-dasharray: 900;
    stroke-dashoffset: 900;
}}
.s1  {{ animation: drawLine 1.8s ease 0.40s forwards; }}
.s2  {{ animation: drawLine 1.8s ease 0.60s forwards; }}
.s3  {{ animation: drawLine 1.8s ease 0.80s forwards; }}
.s4  {{ animation: drawLine 2.0s ease 1.00s forwards; }}
.s5  {{ animation: drawLine 2.0s ease 1.20s forwards; }}
.s6  {{ animation: drawLine 2.0s ease 1.40s forwards; }}
.s7  {{ animation: drawLine 2.4s ease 1.50s forwards; }}
.s8  {{ animation: drawLine 2.4s ease 1.60s forwards; }}
.s9  {{ animation: drawLine 2.4s ease 1.65s forwards; }}
.s10 {{ animation: drawLine 2.6s ease 1.80s forwards; }}
.s11 {{ animation: drawLine 2.6s ease 1.95s forwards; }}
.s12 {{ animation: drawLine 2.6s ease 2.10s forwards; }}

</style></head><body>
<section class="hero">
  <div class="scan-line"></div>
  <div class="hero-content">
    <div class="eyebrow">AI-Powered Media Intelligence</div>
    <h1 class="headline">Nestled<br>in truth.</h1>
    <p class="sub">Instantly verify news articles using machine learning.
      Paste any article text and Shield analyzes it for misinformation in seconds.</p>
    <div class="badge">
      <div class="badge-item">Accuracy <span>~94%</span></div>
      <div class="badge-sep"></div>
      <div class="badge-item">Dataset <span>40K+ Articles</span></div>
      <div class="badge-sep"></div>
      <div class="badge-item">Models <span>LR + DistilBERT</span></div>
    </div>
  </div>
  <div class="cube-wrap">
    <svg width="520" height="540" viewBox="0 0 520 540" fill="none">
      <!-- Primary cube -->
      <polygon class="s1" points="130,130 320,130 320,320 130,320" stroke="#90fc95" stroke-width="1.3" fill="none"/>
      <polygon class="s2" points="130,130 200,60  390,60  320,130"  stroke="#90fc95" stroke-width="1.3" fill="none"/>
      <polygon class="s3" points="320,130 390,60  390,250 320,320"  stroke="#90fc95" stroke-width="1.3" fill="none"/>
      <!-- Secondary cube (offset lower-right) -->
      <polygon class="s4" points="200,210 390,210 390,400 200,400" stroke="#90fc95" stroke-width="0.7" fill="none" opacity="0.42"/>
      <polygon class="s5" points="200,210 270,140 460,140 390,210" stroke="#90fc95" stroke-width="0.7" fill="none" opacity="0.42"/>
      <polygon class="s6" points="390,210 460,140 460,330 390,400" stroke="#90fc95" stroke-width="0.7" fill="none" opacity="0.42"/>
      <!-- Tertiary cube (small, bottom) -->
      <polygon class="s7"  points="100,310 230,310 230,440 100,440" stroke="#90fc95" stroke-width="0.45" fill="none" opacity="0.22"/>
      <polygon class="s8"  points="100,310 166,244 296,244 230,310" stroke="#90fc95" stroke-width="0.45" fill="none" opacity="0.22"/>
      <polygon class="s9"  points="230,310 296,244 296,374 230,440" stroke="#90fc95" stroke-width="0.45" fill="none" opacity="0.22"/>
      <!-- Grid cross-lines on primary cube -->
      <line class="s10" x1="130" y1="225" x2="320" y2="225" stroke="#90fc95" stroke-width="0.35" opacity="0.18"/>
      <line class="s11" x1="225" y1="130" x2="225" y2="320" stroke="#90fc95" stroke-width="0.35" opacity="0.18"/>
      <line class="s12" x1="320" y1="130" x2="390" y2="60"  stroke="#90fc95" stroke-width="0.35" opacity="0.18"/>
    </svg>
  </div>
</section>
</body></html>""", height=640, scrolling=False)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — model selection + sample loaders
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<span style='font-family:Geist Mono,monospace;font-size:11px;"
        "letter-spacing:0.72px;text-transform:uppercase;color:#90fc95;"
        "display:block;margin-bottom:16px;'>Shield</span>",
        unsafe_allow_html=True,
    )

    # Check BERT availability — uses HF Inference API (no local torch)
    _bert_available = bert_api_available()

    if _bert_available:
        mode = st.radio(
            "Detection Engine",
            options=["Fast (TF-IDF + LR)", "Accurate (DistilBERT)"],
            help="Fast mode is instantaneous. Accurate mode uses a fine-tuned DistilBERT via HF Inference API.",
        )
    else:
        mode = "Fast (TF-IDF + LR)"
        st.radio(
            "Detection Engine",
            options=["Fast (TF-IDF + LR)"],
            help="Fast mode uses TF-IDF + Logistic Regression.",
        )
        st.markdown(
            "<div style='border:1px solid rgba(144,252,149,0.2);border-radius:2px;"
            "padding:10px 12px;margin-top:4px;'>"
            "<span style='font-family:Geist Mono,monospace;font-size:10px;"
            "letter-spacing:0.72px;text-transform:uppercase;color:#90fc95;"
            "display:block;margin-bottom:4px;'>DistilBERT</span>"
            "<span style='font-family:Geist,Inter,sans-serif;font-size:12px;"
            "color:rgba(255,255,255,0.55);line-height:1.5;'>"
            "Not configured in this environment. Set HF_TOKEN and HF_MODEL_REPO secrets to enable."
            "</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<span style='font-family:Geist Mono,monospace;font-size:11px;"
        "letter-spacing:0.72px;text-transform:uppercase;color:#90fc95;"
        "display:block;margin-bottom:12px;'>Test Articles</span>",
        unsafe_allow_html=True,
    )
    st.caption("Load a sample article to test the detector.")
    if st.button("Load Fake News Sample", use_container_width=True):
        st.session_state["article_text"] = FAKE_EXAMPLE
    if st.button("Load Real News Sample", use_container_width=True):
        st.session_state["article_text"] = REAL_EXAMPLE
    st.divider()
    st.caption("Educational purposes only. Always verify news with multiple trusted sources.")

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='max-width:1200px;margin:0 auto;padding:72px 40px 40px;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-family:Geist Mono,monospace;font-size:11px;letter-spacing:0.72px;"
    "text-transform:uppercase;color:#ffffff;margin-bottom:12px;'>Article Input</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='font-family:Fraunces,Georgia,serif;font-size:44px;font-weight:400;"
    "line-height:1.1;letter-spacing:-2px;color:#ffffff;margin-bottom:32px;'>"
    "Paste your article.</div>",
    unsafe_allow_html=True,
)

col_input, col_stats = st.columns([3, 1], gap="large")

with col_input:
    title_input = st.text_input(
        "Headline (Optional)",
        placeholder="e.g., Scientists confirm new climate findings…",
    )
    text_input = st.text_area(
        "Article Content",
        key="article_text",
        height=220,
        placeholder="Paste the full article text here (minimum 50 characters)…",
    )
    analyze_btn = st.button("Analyze Article →", type="primary")

with col_stats:
    if text_input:
        wc = len(text_input.split())
        cc = len(text_input)
        sc = len([s for s in text_input.split(".") if s.strip()])
        for val, lbl in [(f"{wc:,}", "Words"), (f"{cc:,}", "Characters"), (str(sc), "Sentences")]:
            st.markdown(
                f"<div style='border:1px solid rgba(255,255,255,0.1);border-radius:2px;padding:20px 24px;"
                f"margin-bottom:8px;animation:countPop 0.5s ease both;'>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:40px;font-weight:300;"
                f"line-height:1;letter-spacing:-2px;color:#ffffff;'>{val}</div>"
                f"<div style='font-family:Geist Mono,monospace;font-size:11px;letter-spacing:0.72px;"
                f"text-transform:uppercase;color:#d2d3d2;margin-top:6px;'>{lbl}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='border:1px dashed rgba(255,255,255,0.2);border-radius:2px;padding:24px;"
            "display:flex;align-items:center;justify-content:center;min-height:160px;'>"
            "<span style='font-family:Geist Mono,monospace;font-size:11px;"
            "letter-spacing:0.72px;text-transform:uppercase;color:#d2d3d2;'>"
            "Waiting for input</span></div>",
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if analyze_btn:
    if not text_input or len(text_input.strip()) < 50:
        st.error("Please enter at least 50 characters of article text to analyze.")
    else:
        combined_text = combine_title_text(title_input, text_input)

        with st.spinner("Analyzing…"):
            vectorizer, lr_model = load_fast_model()

            if vectorizer is None:
                st.error("Fast model not found. Please run the training script first.")
                st.stop()

            if "DistilBERT" in mode:
                try:
                    with st.spinner("Contacting DistilBERT model (may take ~20s on first request)…"):
                        label, prob_fake = predict_bert_api(combined_text)
                    model_used = "DistilBERT (HF API)"
                except Exception as _bert_err:
                    st.warning(f"DistilBERT API unavailable — falling back to Fast Mode. ({_bert_err})")
                    label, prob_fake = predict_fast(combined_text, vectorizer, lr_model)
                    model_used = "Logistic Regression (TF-IDF)"
            else:
                label, prob_fake = predict_fast(combined_text, vectorizer, lr_model)
                model_used = "Logistic Regression (TF-IDF)"

            prob_real = 1 - prob_fake

        if 0.35 < prob_fake < 0.65:
            certainty = "uncertain"
        elif label == 1:
            certainty = "fake"
        else:
            certainty = "real"

        if certainty == "fake":
            vc_class = "fake-col";      vc_pct = f"{prob_fake*100:.1f}%"
            vc_label = "Misinformation probability"; vc_verdict = "Likely Fake"
            bar_color = "#b71c1c";      fill_w = prob_fake * 100
        elif certainty == "real":
            vc_class = "real-col";      vc_pct = f"{prob_real*100:.1f}%"
            vc_label = "Factual probability"; vc_verdict = "Likely Factual"
            bar_color = "#1b5e20";      fill_w = prob_real * 100
        else:
            vc_class = "uncertain-col"; vc_pct = f"{prob_fake*100:.1f}%"
            vc_label = "Misinformation score"; vc_verdict = "Uncertain"
            bar_color = "#4b4d4b";      fill_w = prob_fake * 100

        model_short = model_used.split("(")[0].strip()

        # ── NEON MINT BAND (components.html for SVG) ───────────────────────────
        # Per design.md: text on neon mint MUST be Obsidian (#1e211e), never white.
        # Card descriptor text uses Graphite (#4b4d4b).
        components.html(f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
{_IFRAME_CSS}

body {{ background: #90fc95; margin: 0; }}
.band {{
    background: #90fc95;
    padding: 80px 48px;
    position: relative;
    overflow: hidden;
}}
.inner {{
    max-width: 1200px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
}}

/* Eyebrow — Obsidian on mint (design spec) */
.eyebrow {{
    font-family: 'Geist Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.72px;
    color: #1e211e;
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 10px;
    animation: fadeUp 0.4s ease both;
    opacity: 0.7;
}}
.eyebrow::before {{
    content: '';
    display: inline-block;
    width: 20px;
    height: 1px;
    background: #1e211e;
    opacity: 0.5;
    flex-shrink: 0;
}}

/* Headline — Obsidian on mint (design spec) */
.headline {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 52px;
    font-weight: 400;
    line-height: 1.1;
    letter-spacing: -2.08px;
    color: #1e211e;
    margin: 0 0 32px;
    animation: fadeUp 0.5s ease 0.1s both;
}}

.cards {{ display: flex; gap: 14px; flex-wrap: wrap; }}

/* Cards: white on mint — floats per design */
.card {{
    background: #ffffff;
    border-radius: 2px;
    padding: 28px;
    animation: fadeUp 0.55s ease both;
    box-shadow: 0 2px 12px rgba(30,33,30,0.10);
    position: relative;
    overflow: hidden;
}}
.card::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: transparent;
    transition: background 0.3s;
}}
.card:nth-child(1) {{ flex: 1.5; min-width: 200px; animation-delay: .10s; }}
.card:nth-child(2) {{ flex: 1;   min-width: 160px; animation-delay: .20s; }}
.card:nth-child(3) {{ flex: 1;   min-width: 160px; animation-delay: .30s; }}

/* Card eyebrow — Graphite (design spec: descriptor text in Graphite) */
.c-ey {{
    font-family: 'Geist Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.72px;
    text-transform: uppercase;
    color: #4b4d4b;
    margin-bottom: 12px;
}}

/* Large number — Fraunces Light */
.c-num {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 58px;
    font-weight: 300;
    line-height: 1;
    letter-spacing: -3px;
    margin-bottom: 8px;
    animation: countPop 0.65s ease 0.3s both;
}}
.c-num.fake-col      {{ color: #b71c1c; }}
.c-num.real-col      {{ color: #1b5e20; }}
.c-num.uncertain-col {{ color: #4b4d4b; }}
.c-num.eng {{
    font-size: 20px;
    letter-spacing: -0.6px;
    color: #1e211e;
    margin-top: 4px;
    font-family: 'Geist', sans-serif;
    font-weight: 400;
}}

/* Descriptor — Graphite */
.c-desc {{
    font-family: 'Geist', sans-serif;
    font-size: 15px;
    letter-spacing: -0.3px;
    color: #4b4d4b;
    line-height: 1.5;
}}

/* Progress bar */
.c-bar-t {{
    height: 3px;
    background: #d2d3d2;
    border-radius: 2px;
    margin-top: 20px;
    overflow: hidden;
}}
.c-bar-f {{
    height: 100%;
    border-radius: 2px;
    animation: barGrow 1.1s cubic-bezier(0.23,1,0.32,1) 0.45s both;
}}

/* Decorative cube bleed — Obsidian strokes on mint */
.cube-bleed {{
    position: absolute;
    right: -40px;
    top: -20px;
    opacity: 0.18;
    pointer-events: none;
}}
</style></head><body>
<div class="band">
  <div class="cube-bleed">
    <svg width="340" height="340" viewBox="0 0 340 340" fill="none">
      <polygon points="20,60  200,60  200,240 20,240"  stroke="#1e211e" stroke-width="1.2" fill="none"/>
      <polygon points="20,60  88,0   268,0   200,60"   stroke="#1e211e" stroke-width="1.2" fill="none"/>
      <polygon points="200,60 268,0  268,180 200,240"  stroke="#1e211e" stroke-width="1.2" fill="none"/>
      <polygon points="80,130 260,130 260,310 80,310"  stroke="#1e211e" stroke-width="0.5" fill="none" opacity="0.45"/>
      <polygon points="80,130 148,68  328,68  260,130" stroke="#1e211e" stroke-width="0.5" fill="none" opacity="0.45"/>
      <polygon points="260,130 328,68 328,248 260,310" stroke="#1e211e" stroke-width="0.5" fill="none" opacity="0.45"/>
    </svg>
  </div>
  <div class="inner">
    <span class="eyebrow">Analysis Results</span>
    <div class="headline">Your verdict.</div>
    <div class="cards">
      <div class="card">
        <div class="c-ey">{vc_label}</div>
        <div class="c-num {vc_class}">{vc_pct}</div>
        <div class="c-desc">{vc_verdict}</div>
        <div class="c-bar-t"><div class="c-bar-f" style="width:{fill_w:.1f}%;background:{bar_color};"></div></div>
      </div>
      <div class="card">
        <div class="c-ey">Detection Engine</div>
        <div class="c-num eng">{model_short}</div>
        <div class="c-desc">{model_used}</div>
      </div>
      <div class="card">
        <div class="c-ey">Factual Probability</div>
        <div class="c-num real-col">{prob_real*100:.1f}%</div>
        <div class="c-desc">Consistent with factual reporting</div>
        <div class="c-bar-t"><div class="c-bar-f" style="width:{prob_real*100:.1f}%;background:#1b5e20;"></div></div>
      </div>
    </div>
  </div>
</div>
</body></html>""", height=520, scrolling=False)

        # ── TABS ───────────────────────────────────────────────────────────────
        st.markdown("<div style='max-width:1200px;margin:0 auto;padding:56px 40px;'>", unsafe_allow_html=True)

        tab_verdict, tab_words, tab_sentiment, tab_linguistics = st.tabs([
            "Verdict", "Word Analysis", "Sentiment", "Linguistics"
        ])

        with tab_verdict:
            # Verdict banner — dark surface with explicit white/ash text
            if certainty == "fake":
                border_col = "#b71c1c"
                verdict_num_col = "#ef5350"
                htxt = "Likely Fake"
                desc = "The model found patterns statistically consistent with misinformation."
            elif certainty == "real":
                border_col = "#2e7d32"
                verdict_num_col = "#66bb6a"
                htxt = "Likely Factual"
                desc = "The text matches linguistic patterns found in factual news reporting."
            else:
                border_col = "#4b4d4b"
                verdict_num_col = "#d2d3d2"
                htxt = "Uncertain"
                desc = "Score is in the ambiguous range (35–65%). May be opinion, satire, or mixed content."

            st.markdown(
                f"<div style='"
                f"background:#262b26;"
                f"border-left:4px solid {border_col};"
                f"border-radius:2px;"
                f"padding:36px 44px;"
                f"margin-bottom:28px;"
                f"animation:scaleIn 0.4s ease both;"
                f"'>"
                f"<p style='font-family:Geist Mono,monospace;font-size:11px;letter-spacing:0.72px;"
                f"text-transform:uppercase;color:#d2d3d2;margin-bottom:8px;'>Model Verdict</p>"
                f"<div style='font-family:Fraunces,Georgia,serif;font-size:32px;font-weight:400;"
                f"letter-spacing:-1.2px;color:{verdict_num_col};margin-bottom:10px;'>{htxt}</div>"
                f"<p style='font-family:Geist,Inter,sans-serif;font-size:17px;line-height:1.6;"
                f"color:rgba(255,255,255,0.78);margin-bottom:14px;'>{desc}</p>"
                f"<p style='font-family:Geist Mono,monospace;font-size:11px;letter-spacing:0.72px;"
                f"text-transform:uppercase;color:#4b4d4b;'>"
                f"Engine — {model_used} &nbsp;|&nbsp; "
                f"Fake: {prob_fake*100:.1f}% &nbsp;|&nbsp; Factual: {prob_real*100:.1f}%</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            bar_col_plot = "#2e7d32" if label == 0 else "#b71c1c"
            with c1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob_fake * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "MISINFORMATION PROBABILITY",
                           "font": {"size": 10, "family": "Geist Mono, monospace", "color": "#d2d3d2"}},
                    number={"suffix": "%", "font": {"size": 44, "color": "#ffffff",
                                                     "family": "Fraunces, Georgia, serif"}},
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickwidth": 1,
                            "tickcolor": "rgba(255,255,255,0.15)",
                            "tickfont": {"color": "#d2d3d2", "size": 10},
                        },
                        "bar": {"color": bar_col_plot},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 1,
                        "bordercolor": "rgba(255,255,255,0.12)",
                        "steps": [
                            {"range": [0, 35],   "color": "rgba(46,125,50,0.08)"},
                            {"range": [35, 65],  "color": "rgba(75,77,75,0.06)"},
                            {"range": [65, 100], "color": "rgba(183,28,28,0.08)"},
                        ],
                        "threshold": {
                            "line": {"color": "rgba(255,255,255,0.4)", "width": 2},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                fig.update_layout(**_PLOT_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = go.Figure(data=[go.Bar(
                    x=["Factual", "Misinformation"],
                    y=[prob_real * 100, prob_fake * 100],
                    marker_color=["#2e7d32", "#b71c1c"],
                    marker_line_color="rgba(255,255,255,0.15)",
                    marker_line_width=1,
                    text=[f"{prob_real*100:.1f}%", f"{prob_fake*100:.1f}%"],
                    textposition="auto",
                    textfont=dict(size=13, color="#ffffff", family="Geist Mono, monospace"),
                )])
                fig2.update_layout(
                    title=dict(
                        text="CLASS PROBABILITIES",
                        font=dict(size=10, family="Geist Mono, monospace", color="#d2d3d2"),
                    ),
                    yaxis=dict(
                        range=[0, 100],
                        title="Probability (%)",
                        gridcolor="rgba(255,255,255,0.08)",
                        tickfont=dict(color="#d2d3d2", size=10),
                        title_font=dict(color="#d2d3d2", size=10, family="Geist Mono, monospace"),
                    ),
                    xaxis=dict(tickfont=dict(color="#d2d3d2", size=12, family="Geist, Inter, sans-serif")),
                    **_PLOT_LAYOUT,
                )
                st.plotly_chart(fig2, use_container_width=True)

        with tab_words:
            if "DistilBERT" not in mode:
                st.markdown(
                    "<p style='font-family:Geist Mono,monospace;font-size:11px;letter-spacing:0.72px;"
                    "text-transform:uppercase;color:#d2d3d2;margin-bottom:8px;'>Model Interpretability</p>"
                    "<p style='font-family:Geist,Inter,sans-serif;font-size:15px;line-height:1.6;"
                    "color:rgba(255,255,255,0.75);margin-bottom:24px;'>"
                    "Words that most influenced the model&#8217;s decision, scored by "
                    "TF-IDF weight &times; LR coefficient.</p>",
                    unsafe_allow_html=True,
                )
                fake_words, real_words = get_top_fake_words(combined_text, vectorizer, lr_model)
                cw1, cw2 = st.columns(2, gap="large")
                for col, words, header in [(cw1, fake_words, "Fake Indicators"),
                                            (cw2, real_words, "Real Indicators")]:
                    with col:
                        rows = "".join(
                            f"<div style='display:flex;justify-content:space-between;"
                            f"align-items:center;padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.1);'>"
                            f"<span style='font-family:Geist,sans-serif;font-size:15px;"
                            f"color:#ffffff;'>{w}</span>"
                            f"<span style='font-family:Geist Mono,monospace;font-size:11px;"
                            f"letter-spacing:0.72px;color:#d2d3d2;'>+{s:.3f}</span></div>"
                            for w, s in words[:8]
                        ) if words else (
                            "<p style='font-family:Geist Mono,monospace;font-size:11px;"
                            "color:#d2d3d2;text-transform:uppercase;letter-spacing:0.72px;'>"
                            "No strong indicators found.</p>"
                        )
                        st.markdown(
                            f"<div style='border:1px solid rgba(255,255,255,0.1);border-radius:2px;padding:20px 24px;'>"
                            f"<div style='font-family:Geist Mono,monospace;font-size:11px;"
                            f"letter-spacing:0.72px;text-transform:uppercase;color:#d2d3d2;"
                            f"margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,0.1);'>"
                            f"{header}</div>{rows}</div>",
                            unsafe_allow_html=True,
                        )
            else:
                st.info("Word-level explanation is only available in Fast Mode. "
                        "DistilBERT uses contextual embeddings rather than individual token weights.")

        with tab_sentiment:
            st.markdown(
                "<p style='font-family:Geist Mono,monospace;font-size:11px;letter-spacing:0.72px;"
                "text-transform:uppercase;color:#d2d3d2;margin-bottom:8px;'>VADER Sentiment</p>"
                "<p style='font-family:Geist,Inter,sans-serif;font-size:15px;line-height:1.6;"
                "color:rgba(255,255,255,0.75);margin-bottom:24px;'>"
                "Fake news frequently uses highly charged emotional language. "
                "High absolute compound scores warrant additional scrutiny.</p>",
                unsafe_allow_html=True,
            )
            sent_df  = extract_sentiment_features(pd.Series([text_input]))
            sent_row = sent_df.iloc[0]
            cs1, cs2, cs3, cs4 = st.columns(4)
            cs1.metric("Negative", f"{sent_row['vader_neg']:.2f}")
            cs2.metric("Neutral",  f"{sent_row['vader_neu']:.2f}")
            cs3.metric("Positive", f"{sent_row['vader_pos']:.2f}")
            cs4.metric("Compound", f"{sent_row['vader_compound']:.2f}",
                        delta="Extreme" if abs(sent_row["vader_compound"]) > 0.6 else "Moderate",
                        delta_color="inverse")
            fig_s = go.Figure(data=[go.Bar(
                x=["Negative", "Neutral", "Positive"],
                y=[sent_row["vader_neg"], sent_row["vader_neu"], sent_row["vader_pos"]],
                marker_color=["#b71c1c", "#4b4d4b", "#2e7d32"],
                marker_line_color="rgba(255,255,255,0.12)", marker_line_width=1,
                text=[f"{sent_row['vader_neg']:.2f}", f"{sent_row['vader_neu']:.2f}", f"{sent_row['vader_pos']:.2f}"],
                textposition="auto", textfont=dict(size=13, color="#ffffff", family="Geist Mono, monospace"),
            )])
            fig_s.update_layout(
                height=260, showlegend=False, margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Geist, Inter, sans-serif", color="#d2d3d2"),
                yaxis=dict(
                    title="Proportion",
                    gridcolor="rgba(255,255,255,0.08)",
                    tickfont=dict(color="#d2d3d2", size=10),
                    title_font=dict(color="#d2d3d2", size=10, family="Geist Mono, monospace"),
                ),
                xaxis=dict(tickfont=dict(color="#d2d3d2", size=12, family="Geist, Inter, sans-serif")),
            )
            st.plotly_chart(fig_s, use_container_width=True)
            if abs(sent_row["vader_compound"]) > 0.6:
                st.warning(f"High emotional intensity detected (compound: {sent_row['vader_compound']:.2f}). "
                           "Sensationalist language is a common pattern in misinformation.")

        with tab_linguistics:
            st.markdown(
                "<p style='font-family:Geist Mono,monospace;font-size:11px;letter-spacing:0.72px;"
                "text-transform:uppercase;color:#d2d3d2;margin-bottom:8px;'>Linguistic Traits</p>"
                "<p style='font-family:Geist,Inter,sans-serif;font-size:15px;line-height:1.6;"
                "color:rgba(255,255,255,0.75);margin-bottom:24px;'>"
                "Structural signals from the text. Misinformation often exhibits "
                "excessive punctuation, ALL-CAPS usage, and reduced vocabulary diversity.</p>",
                unsafe_allow_html=True,
            )
            ling_df  = extract_linguistic_features(pd.Series([combined_text]))
            ling_row = ling_df.iloc[0]
            cl1, cl2, cl3 = st.columns(3)
            with cl1:
                st.metric("Avg Word Length",   f"{ling_row['avg_word_length']:.2f} ch")
                st.metric("Exclamation Marks", ling_row["exclamation_count"])
            with cl2:
                st.metric("Unique Word Ratio", f"{ling_row['unique_word_ratio']:.3f}")
                st.metric("Question Marks",    ling_row["question_count"])
            with cl3:
                st.metric("Avg Sentence Length", f"{ling_row['avg_sentence_length']:.1f} w")
                st.metric("Capitalization Ratio", f"{ling_row['capital_ratio']:.4f}")

        st.markdown("</div>", unsafe_allow_html=True)















