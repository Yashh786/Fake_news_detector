import streamlit as st

st.set_page_config(
    page_title="Terms and Conditions | Shield",
    page_icon=":material/fact_check:",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALTITUDE_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --color-carbon-canvas:  #181818;
        --color-obsidian:       #111111;
        --color-graphite-card:  #1f1f1f;
        --color-iron-peak:      #323232;
        --color-bone:           #eeeeee;
        --color-fog:            #a4a19b;
        --color-voltage-blue:   #2b7fff;
        --font-serif:   'Libre Baskerville', 'Source Serif Pro', Georgia, serif;
        --font-inter:   'Inter', system-ui, -apple-system, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer    {visibility: hidden;}

    html, body, [class*="css"], .stApp {
        font-family: var(--font-inter) !important;
        background-color: var(--color-carbon-canvas) !important;
        color: var(--color-bone);
    }
    .stApp, .main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { 
        background-color: var(--color-carbon-canvas) !important; 
    }

    [data-testid="stSidebar"] {
        background-color: var(--color-obsidian) !important;
        border-right: 1px solid var(--color-graphite-card);
    }
    [data-testid="stSidebar"] * { color: var(--color-bone) !important; }
    [data-testid="stSidebar"] hr { border-color: var(--color-iron-peak) !important; }

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
    
    .legal-content {
        font-family: var(--font-inter);
        font-size: 15px;
        line-height: 1.7;
        color: var(--color-fog);
        max-width: 800px;
        padding: 0 32px;
    }
    .legal-content h2 {
        font-family: var(--font-serif) !important;
        font-size: 24px !important;
        font-weight: 400 !important;
        margin-top: 48px !important;
        margin-bottom: 16px !important;
        color: var(--color-bone) !important;
        border-bottom: 1px solid var(--color-iron-peak);
        padding-bottom: 8px;
        letter-spacing: -0.5px !important;
    }
    .legal-content p, .legal-content li {
        margin-bottom: 16px;
    }
    .legal-content a {
        color: var(--color-voltage-blue) !important;
        text-decoration: none;
    }
    
    hr { border-color: var(--color-iron-peak) !important; margin: 32px 0 !important; }
</style>
"""

st.markdown(ALTITUDE_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Terms & Conditions</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="legal-content">

<p>Last Updated: September 2026</p>

<h2>1. Purpose and Limitations</h2>
<p>Shield is an educational tool that uses machine learning to classify news text as factual or potentially misleading. It does not constitute professional fact-checking. Its outputs should not be used as the sole basis for decisions about the accuracy of any specific news article.</p>

<h2>2. Accuracy Disclaimer</h2>
<p>The model has known limitations. It was trained on a specific dataset and may produce incorrect classifications, particularly for satire, opinion pieces, regional news, or articles outside its training distribution. Confidence scores are probabilistic estimates, not guarantees of accuracy.</p>

<h2>3. Acceptable Use</h2>
<p>You may use this tool for personal research and educational purposes. You may not use it to systematically discredit legitimate journalism, generate misleading claims about the reliability of specific sources, or in any application where an incorrect classification could cause harm.</p>

<h2>4. No Warranty</h2>
<p>This application is provided as-is, without warranty of any kind, express or implied. The developer makes no representations about the accuracy, completeness, reliability, or suitability of the application for any particular purpose.</p>

<h2>5. Limitation of Liability</h2>
<p>To the maximum extent permitted by applicable law, the developer is not liable for any direct, indirect, incidental, or consequential damages arising from your use of, or reliance on, the application or its outputs.</p>

<h2>6. Intellectual Property</h2>
<p>The source code for this application is available under its project license. The underlying machine learning models are derived from publicly available datasets and pre-trained models. Attribution details are available in the project repository.</p>

<h2>7. Changes to These Terms</h2>
<p>These terms may be updated at any time. The updated date will be revised accordingly. Continued use of the application following any changes constitutes acceptance of the revised terms.</p>

<h2>8. Governing Law</h2>
<p>These terms are governed by applicable law in the jurisdiction where the developer is located. Any disputes arising from the use of this application shall be resolved under that jurisdiction.</p>

</div>
""", unsafe_allow_html=True)
