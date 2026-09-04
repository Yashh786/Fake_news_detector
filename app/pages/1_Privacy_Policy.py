import streamlit as st

st.set_page_config(
    page_title="Privacy Policy | Shield",
    page_icon=":material/policy:",
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
    <h1 class="hero-title">Privacy Policy</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="legal-content">

<p>Last Updated: September 2026</p>

<h2>1. Introduction</h2>
<p>At Shield, we respect your privacy and are committed to protecting your personal data. This Privacy Policy outlines how we collect, use, and safeguard the information you provide when using our Fake News Detection application.</p>

<h2>2. Data Collection</h2>
<p>We do not require account creation to use Shield. The text inputs you submit for analysis are processed transiently in memory. We do not store, log, or persist the articles or headlines you analyze on our servers.</p>

<h2>3. Third-Party Services</h2>
<p>Our application operates using state-of-the-art machine learning models (Logistic Regression, DistilBERT) hosted on our own infrastructure. We do not transmit your input texts to third-party APIs (such as OpenAI or Google) for classification.</p>

<h2>4. Cookies and Analytics</h2>
<p>We do not use tracking cookies or aggressive analytics. Standard session data necessary for maintaining the application state is stored locally on your device in accordance with standard web practices.</p>

<h2>5. Changes to This Policy</h2>
<p>We may update this Privacy Policy from time to time. Any changes will be posted on this page with an updated revision date. By continuing to use the service, you acknowledge the revised terms.</p>

<h2>6. Contact Us</h2>
<p>If you have any questions or concerns regarding your privacy while using Shield, please reach out to our support channel.</p>

</div>
""", unsafe_allow_html=True)
