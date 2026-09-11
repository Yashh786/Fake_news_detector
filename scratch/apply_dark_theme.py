import os
import re

app_path = r"c:\Users\Yash\OneDrive\Desktop\Yash\fake-news-detector\app\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Restore Sidebar Menu
code = code.replace(
    '/* Hide Streamlit header bar visually but keep layout intact */\n[data-testid="stHeader"] {{ visibility: hidden !important; height: 0 !important; min-height: 0 !important; }}\n\n/* ── SIDEBAR ALWAYS VISIBLE — hide the collapse button ─ */\n[data-testid="stSidebarCollapseButton"] {{ display: none !important; }}\n[data-testid="collapsedControl"]        {{ display: none !important; }}',
    '/* Restore standard Streamlit header so the sidebar hamburger remains accessible */\n/* [data-testid="stHeader"] {{ visibility: hidden !important; height: 0 !important; min-height: 0 !important; }} */\n/* [data-testid="stSidebarCollapseButton"] {{ display: none !important; }} */\n/* [data-testid="collapsedControl"]        {{ display: none !important; }} */'
)

# 2. Change background to dark
code = code.replace('background-color: var(--paper) !important;', 'background-color: var(--obsidian) !important;')
code = code.replace('background: var(--paper);', 'background: var(--obsidian);')

# Change global text color to light
code = code.replace('color: var(--obsidian) !important;', 'color: var(--paper) !important;')

# Top Nav borders
code = code.replace('border-bottom: 1px solid var(--ash);', 'border-bottom: 1px solid rgba(255,255,255,0.1);')
code = code.replace('border-bottom:1px solid #d2d3d2;', 'border-bottom:1px solid rgba(255,255,255,0.1);')

# Text Colors
code = code.replace('color:#1e211e;', 'color:#ffffff;')
code = code.replace('color:#4b4d4b;', 'color:#d2d3d2;')

# Borders
code = code.replace('border:1px solid #d2d3d2;', 'border:1px solid rgba(255,255,255,0.1);')
code = code.replace('border:1px dashed #d2d3d2;', 'border:1px dashed rgba(255,255,255,0.2);')

# Wait, the hero component CSS also had #1e211e which we replaced with #ffffff.
# We need to revert the hero background back to dark.
code = code.replace('section.hero {\n    background:#ffffff;', 'section.hero {\n    background:#1e211e;')

# The Mint Band explicitly had #1e211e which got turned into #ffffff. Let's make it look good on dark.
code = code.replace('body{background:#90fc95;}', 'body{background:#1e211e;}')
code = code.replace('.band{background:#90fc95;', '.band{background:#1e211e;border-top:1px solid rgba(144,252,149,0.2);')
code = code.replace('.card{background:#fff;', '.card{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);')
code = code.replace('.c-bar-t{height:3px;background:#d2d3d2;', '.c-bar-t{height:3px;background:rgba(255,255,255,0.1);')

# Plotly graphs
code = code.replace('font=dict(family="Geist, Inter, sans-serif", color="#ffffff")', 'font=dict(family="Geist, Inter, sans-serif", color="#ffffff")')
code = code.replace('tickcolor": "#d2d3d2"', 'tickcolor": "rgba(255,255,255,0.2)"')
code = code.replace('color": "#4b4d4b"', 'color": "#d2d3d2"')
code = code.replace('bordercolor": "#d2d3d2"', 'bordercolor": "rgba(255,255,255,0.2)"')
code = code.replace('gridcolor="#d2d3d2"', 'gridcolor="rgba(255,255,255,0.1)"')

# Tabs specific
code = code.replace('bg, border, hcol, htxt = "#fff5f5", "#b71c1c", "#ffffff", "Likely Fake"', 'bg, border, hcol, htxt = "#1e211e", "#ff5252", "#ff5252", "Likely Fake"')
code = code.replace('bg, border, hcol, htxt = "#f0fff1", "#1b5e20", "#ffffff", "Likely Factual"', 'bg, border, hcol, htxt = "#1e211e", "#90fc95", "#90fc95", "Likely Factual"')
code = code.replace('bg, border, hcol, htxt = "#f7f7f7", "#d2d3d2", "#ffffff", "Uncertain"', 'bg, border, hcol, htxt = "#1e211e", "rgba(255,255,255,0.2)", "#ffffff", "Uncertain"')

with open(app_path, 'w', encoding='utf-8') as f:
    f.write(code)

print("Replacement complete")
