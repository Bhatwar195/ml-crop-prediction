import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
# CROP KNOWLEDGE BASE  (no emojis)
# ─────────────────────────────────────────────────────────────────────────────
CROP_INFO = {
    "rice":        {"season": "Kharif  (Jun – Nov)",  "water": "High  1200 – 2000 mm",  "temp": "20 – 37 °C", "soil": "Clayey, Loamy"},
    "maize":       {"season": "Kharif / Rabi",        "water": "Medium  500 – 800 mm",  "temp": "18 – 32 °C", "soil": "Loamy, Sandy"},
    "chickpea":    {"season": "Rabi  (Oct – Mar)",    "water": "Low  300 – 400 mm",     "temp": "10 – 25 °C", "soil": "Sandy, Loamy"},
    "kidneybeans": {"season": "Kharif  (Jun – Oct)",  "water": "Medium  300 – 500 mm",  "temp": "16 – 24 °C", "soil": "Loamy, Clay"},
    "pigeonpeas":  {"season": "Kharif  (Jun – Nov)",  "water": "Low  600 – 1000 mm",   "temp": "18 – 29 °C", "soil": "Sandy, Loamy"},
    "mothbeans":   {"season": "Kharif  (Jun – Sep)",  "water": "Low  200 – 400 mm",    "temp": "24 – 38 °C", "soil": "Sandy, Loamy"},
    "mungbean":    {"season": "Kharif / Summer",      "water": "Low  300 – 400 mm",    "temp": "25 – 35 °C", "soil": "Sandy, Loamy"},
    "blackgram":   {"season": "Kharif  (Jun – Sep)",  "water": "Low  300 – 500 mm",    "temp": "25 – 35 °C", "soil": "Loamy, Clay"},
    "lentil":      {"season": "Rabi  (Oct – Mar)",    "water": "Low  250 – 400 mm",    "temp": "10 – 25 °C", "soil": "Loamy, Sandy"},
    "pomegranate": {"season": "Year-round",            "water": "Low–Med  500 – 800 mm","temp": "25 – 35 °C", "soil": "Loamy, Sandy"},
    "banana":      {"season": "Year-round",            "water": "High  1200 – 2200 mm", "temp": "20 – 35 °C", "soil": "Rich Loamy"},
    "mango":       {"season": "Summer  (Mar – Jun)",  "water": "Low–Med  750 – 1500 mm","temp": "24 – 30 °C", "soil": "Sandy, Loamy"},
    "grapes":      {"season": "Year-round",            "water": "Low  700 – 900 mm",    "temp": "15 – 35 °C", "soil": "Sandy, Loamy"},
    "watermelon":  {"season": "Summer  (Feb – Jun)",  "water": "Medium  400 – 600 mm", "temp": "22 – 30 °C", "soil": "Sandy, Loamy"},
    "muskmelon":   {"season": "Summer  (Feb – May)",  "water": "Medium  400 – 600 mm", "temp": "25 – 35 °C", "soil": "Sandy, Loamy"},
    "apple":       {"season": "Autumn  (Sep – Nov)",  "water": "Medium  1000 – 1250 mm","temp": "5 – 24 °C", "soil": "Loamy, Well-drained"},
    "orange":      {"season": "Winter  (Nov – Feb)",  "water": "Medium  750 – 1000 mm","temp": "13 – 38 °C", "soil": "Sandy, Loamy"},
    "papaya":      {"season": "Year-round",            "water": "Medium  1000 – 1500 mm","temp": "22 – 33 °C","soil": "Sandy, Loamy"},
    "coconut":     {"season": "Year-round",            "water": "High  1500 – 2500 mm", "temp": "20 – 32 °C", "soil": "Sandy, Loamy"},
    "cotton":      {"season": "Kharif  (May – Nov)",  "water": "Medium  700 – 1300 mm","temp": "21 – 35 °C", "soil": "Black, Sandy"},
    "jute":        {"season": "Kharif  (Mar – Jun)",  "water": "High  1000 – 2000 mm", "temp": "24 – 37 °C", "soil": "Loamy, Sandy"},
    "coffee":      {"season": "Year-round",            "water": "Medium  1500 – 2500 mm","temp": "15 – 28 °C","soil": "Rich, Well-drained"},
}

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,400;1,600&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --g1: #132d1e;
    --g2: #1e5c3a;
    --g3: #2d8653;
    --g4: #4db87a;
    --g5: #a8e6c1;
    --g6: #e8f7ef;
    --g7: #f4fbf7;
    --white: #ffffff;
    --text: #0c1f14;
    --muted: #4a7060;
    --border: rgba(30,92,58,0.13);
    --shadow: rgba(19,45,30,0.10);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--white) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    scroll-behavior: smooth;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="collapsedControl"] { display: none !important; }

.block-container {
    max-width: 100% !important;
    padding: 0 !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--g7); }
::-webkit-scrollbar-thumb { background: var(--g4); border-radius: 10px; }

/* ── NAVBAR ── */
.nav {
    position: sticky; top: 0; z-index: 999;
    background: var(--g1);
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 72px;
    box-shadow: 0 1px 0 rgba(255,255,255,0.05), 0 4px 24px rgba(0,0,0,0.3);
}
.nav-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 24px;
    font-weight: 600;
    color: var(--white);
    letter-spacing: 0.3px;
}
.nav-logo b { color: var(--g4); font-style: italic; font-weight: 600; }
.nav-right {
    display: flex;
    align-items: center;
    gap: 32px;
}
.nav-link {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.5);
}
.nav-tag {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--g1);
    background: var(--g4);
    padding: 4px 14px;
    border-radius: 100px;
}

/* ── HERO ── */
.hero {
    background: var(--g1);
    background-image:
        radial-gradient(ellipse 70% 80% at -5% 50%, rgba(77,184,122,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 105% 10%, rgba(45,134,83,0.35) 0%, transparent 55%),
        radial-gradient(ellipse 40% 50% at 50% 120%, rgba(19,45,30,0.8) 0%, transparent 60%);
    padding: 96px 72px 130px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -2px; left: 0; right: 0;
    height: 80px;
    background: var(--white);
    clip-path: ellipse(58% 100% at 50% 100%);
}

/* geometric decoration */
.geo {
    position: absolute;
    right: 72px;
    top: 50%;
    transform: translateY(-50%);
    width: 380px;
    height: 380px;
    z-index: 1;
}
.geo-circle {
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(77,184,122,0.2);
}
.geo-circle:nth-child(1){ width:380px;height:380px;top:0;left:0; animation: spin 40s linear infinite; }
.geo-circle:nth-child(2){ width:270px;height:270px;top:55px;left:55px; animation: spin 28s linear infinite reverse; }
.geo-circle:nth-child(3){ width:160px;height:160px;top:110px;left:110px; background:rgba(77,184,122,0.05); animation: spin 18s linear infinite; }
.geo-dot {
    position: absolute;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--g4);
    top: 50%; left: -4px;
    transform: translateY(-50%);
    box-shadow: 0 0 12px var(--g4);
}
@keyframes spin { from{transform:rotate(0deg);}to{transform:rotate(360deg);} }

/* vertical line decoration */
.v-line {
    position: absolute;
    left: 72px; top: 0; bottom: 0;
    width: 1px;
    background: linear-gradient(to bottom, transparent, rgba(77,184,122,0.3), transparent);
}

.hero-inner { position: relative; z-index: 2; max-width: 620px; }

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 30px;
}
.eyebrow-line {
    width: 32px; height: 1px;
    background: var(--g4);
}
.eyebrow-text {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--g4);
}

.hero-h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(46px, 5.8vw, 76px);
    font-weight: 300;
    color: var(--white);
    line-height: 1.05;
    letter-spacing: -1px;
    margin-bottom: 24px;
}
.hero-h1 i {
    font-style: italic;
    font-weight: 400;
    color: var(--g4);
}

.hero-p {
    font-size: 16px;
    font-weight: 300;
    color: rgba(255,255,255,0.55);
    line-height: 1.75;
    max-width: 440px;
    margin-bottom: 48px;
}

.hero-stats {
    display: flex;
    gap: 0;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    overflow: hidden;
    width: fit-content;
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(10px);
}
.hstat {
    padding: 18px 32px;
    border-right: 1px solid rgba(255,255,255,0.07);
    text-align: center;
}
.hstat:last-child { border-right: none; }
.hstat-n {
    font-family: 'Cormorant Garamond', serif;
    font-size: 28px;
    font-weight: 600;
    color: var(--white);
    line-height: 1;
    margin-bottom: 6px;
}
.hstat-l {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
}

/* ── PROCESS STRIP ── */
.process {
    background: var(--g7);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: stretch;
    justify-content: center;
}
.proc-step {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 26px 44px;
    flex: 1;
    max-width: 280px;
    border-right: 1px solid var(--border);
    position: relative;
}
.proc-step:last-child { border-right: none; }
.proc-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 36px;
    font-weight: 300;
    color: var(--g5);
    line-height: 1;
    flex-shrink: 0;
    min-width: 32px;
}
.proc-t {
    font-size: 13px;
    font-weight: 600;
    color: var(--g1);
    margin-bottom: 3px;
}
.proc-s {
    font-size: 12px;
    font-weight: 300;
    color: var(--muted);
    line-height: 1.4;
}

/* ── SECTION WRAPPER ── */
.section { padding: 72px 72px 40px; }
.section-sm { padding: 0 72px 64px; }

.s-eyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    color: var(--g3);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.s-eyebrow::before {
    content: '';
    display: inline-block;
    width: 24px; height: 1px;
    background: var(--g3);
}
.s-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 40px;
    font-weight: 400;
    color: var(--g1);
    line-height: 1.1;
    margin-bottom: 8px;
}
.s-rule {
    height: 1px;
    background: var(--border);
    margin: 28px 0 40px;
}

/* ── INPUT CARDS ── */
.icard {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 26px 24px 22px;
    box-shadow: 0 2px 16px var(--shadow);
    transition: border-color 0.25s, box-shadow 0.25s;
}
.icard:hover {
    border-color: rgba(45,134,83,0.28);
    box-shadow: 0 8px 36px rgba(19,45,30,0.1);
}
.icard-head {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
}
.icard-ico {
    width: 40px; height: 40px;
    border-radius: 10px;
    background: var(--g6);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.icard-ico svg { width: 20px; height: 20px; stroke: var(--g3); fill: none; stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }
.icard-t {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--g1);
}
.icard-s { font-size: 11px; color: var(--muted); margin-top: 3px; }

/* ── PH REFERENCE ── */
.ph-ref {
    margin-top: 14px;
    background: var(--g7);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
}
.ph-ref-title {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--g3);
    margin-bottom: 10px;
}
.ph-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    padding: 4px 0;
    border-bottom: 1px solid rgba(30,92,58,0.06);
    color: var(--muted);
}
.ph-row:last-child { border-bottom: none; }
.ph-val { font-weight: 600; color: var(--g2); font-size: 12px; }

/* ── STREAMLIT WIDGET OVERRIDES ── */
[data-testid="stNumberInput"] input {
    background: var(--g7) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    transition: all 0.2s ease !important;
    height: 44px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: var(--g3) !important;
    box-shadow: 0 0 0 3px rgba(45,134,83,0.1) !important;
    background: var(--white) !important;
    outline: none !important;
}
[data-testid="stNumberInput"] label {
    color: var(--muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    margin-bottom: 4px !important;
}
[data-testid="stNumberInput"] button {
    background: var(--g7) !important;
    border-color: var(--border) !important;
    color: var(--g3) !important;
    border-radius: 8px !important;
    height: 20px !important;
}
[data-testid="stNumberInput"] button:hover {
    background: var(--g6) !important;
    border-color: var(--g3) !important;
}

/* ── PREDICT BUTTON ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: var(--g1) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    padding: 17px 32px !important;
    box-shadow: 0 4px 20px rgba(19,45,30,0.25), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    transition: all 0.25s cubic-bezier(0.22,1,0.36,1) !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="stButton"] > button:hover {
    background: var(--g2) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(19,45,30,0.32) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── RESULT BANNER ── */
.result-banner {
    background: var(--g1);
    background-image:
        radial-gradient(ellipse 60% 80% at 0% 50%, rgba(77,184,122,0.15) 0%, transparent 55%),
        radial-gradient(ellipse 40% 60% at 100% 20%, rgba(45,134,83,0.4) 0%, transparent 50%);
    border-radius: 24px;
    padding: 52px 60px;
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 48px;
    align-items: center;
    position: relative;
    overflow: hidden;
    animation: rise 0.6s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes rise {
    from { opacity:0; transform:translateY(28px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-banner::before {
    content:'';
    position:absolute;
    top:0;left:0;right:0;
    height:2px;
    background:linear-gradient(90deg, transparent, var(--g4), transparent);
}
.result-banner::after {
    content:'';
    position:absolute;
    bottom:-100px;right:-80px;
    width:300px;height:300px;
    border-radius:50%;
    border:1px solid rgba(77,184,122,0.1);
}

.r-eye {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--g4);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.r-eye::before { content:''; display:inline-block; width:24px;height:1px;background:var(--g4); }

.r-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(48px, 6vw, 80px);
    font-weight: 400;
    font-style: italic;
    color: var(--white);
    line-height: 1;
    margin-bottom: 18px;
    text-transform: capitalize;
}
.r-desc {
    font-size: 14px;
    font-weight: 300;
    color: rgba(255,255,255,0.55);
    max-width: 420px;
    line-height: 1.7;
}

.r-params {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    min-width: 300px;
}
.r-param {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    transition: background 0.2s;
}
.r-param:hover { background: rgba(255,255,255,0.08); }
.r-param-v {
    font-family: 'Cormorant Garamond', serif;
    font-size: 24px;
    font-weight: 600;
    color: var(--white);
    line-height: 1;
    margin-bottom: 5px;
}
.r-param-k {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
}
.r-param.wide { grid-column: span 2; }

/* ── CROP INFO CARD ── */
.crop-info-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    animation: rise 0.6s 0.1s cubic-bezier(0.22,1,0.36,1) both;
}
.cinfo {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 22px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 12px var(--shadow);
    transition: transform 0.2s, box-shadow 0.2s;
}
.cinfo:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(19,45,30,0.12);
}
.cinfo::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--g3), var(--g4));
}
.cinfo-ico-wrap {
    width: 44px; height: 44px;
    border-radius: 12px;
    background: var(--g6);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
}
.cinfo-ico-wrap svg {
    width: 22px; height: 22px;
    stroke: var(--g3); fill: none;
    stroke-width: 1.5;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.cinfo-label {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.cinfo-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 18px;
    font-weight: 600;
    color: var(--g1);
    line-height: 1.3;
}

/* ── CONFIDENCE CHART wrapper ── */
.chart-wrapper {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 28px;
    box-shadow: 0 2px 16px var(--shadow);
    animation: rise 0.6s 0.2s cubic-bezier(0.22,1,0.36,1) both;
}
.chart-head {
    margin-bottom: 24px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--border);
}
.chart-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 600;
    color: var(--g1);
    margin-bottom: 4px;
}
.chart-sub {
    font-size: 12px;
    color: var(--muted);
    font-weight: 300;
}

/* ── FEATURES SECTION ── */
.feat-section {
    background: var(--g7);
    border-top: 1px solid var(--border);
    padding: 80px 72px;
}
.feat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 44px;
}
.feat {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 28px;
    transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
}
.feat:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 52px rgba(19,45,30,0.1);
    border-color: rgba(45,134,83,0.28);
}
.feat-ico {
    width: 52px; height: 52px;
    border-radius: 14px;
    background: var(--g6);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 22px;
    border: 1px solid var(--border);
}
.feat-ico svg {
    width: 24px; height: 24px;
    stroke: var(--g3); fill: none;
    stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round;
}
.feat-t {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 600;
    color: var(--g1);
    margin-bottom: 12px;
    line-height: 1.2;
}
.feat-p {
    font-size: 14px;
    font-weight: 300;
    color: var(--muted);
    line-height: 1.7;
}

/* ── FOOTER ── */
.footer {
    background: var(--g1);
    padding: 40px 72px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-top: 1px solid rgba(255,255,255,0.05);
}
.footer-logo {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 600;
    color: var(--white);
}
.footer-logo b { color: var(--g4); font-style: italic; }
.footer-copy {
    font-size: 12px;
    color: rgba(255,255,255,0.3);
    letter-spacing: 0.5px;
    margin-top: 6px;
}
.footer-tags { display: flex; gap: 8px; justify-content: flex-end; }
.ftag {
    font-size: 9px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px;
    padding: 4px 12px;
}

/* column gutter */
[data-testid="column"] { padding: 0 10px !important; }
[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load("models/crop_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav">
    <div class="nav-logo">Crop<b>Sense</b></div>
    <div class="nav-right">
        <span class="nav-link">Home</span>
        <span class="nav-link">Model</span>
        <span class="nav-link">About</span>
        <span class="nav-tag">AI Powered</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="v-line"></div>
    <div class="hero-inner">
        <div class="hero-eyebrow">
            <span class="eyebrow-line"></span>
            <span class="eyebrow-text">Random Forest Classifier</span>
        </div>
        <h1 class="hero-h1">Precision Farming<br>Powered by<br><i>Artificial Intelligence</i></h1>
        <p class="hero-p">
            Enter your soil composition and local climate measurements.
            The model analyzes 7 parameters to recommend the most
            suitable crop for your land conditions.
        </p>
        <div class="hero-stats">
            <div class="hstat">
                <div class="hstat-n">7</div>
                <div class="hstat-l">Parameters</div>
            </div>
            <div class="hstat">
                <div class="hstat-n">22</div>
                <div class="hstat-l">Crop Classes</div>
            </div>
            <div class="hstat">
                <div class="hstat-n">RF</div>
                <div class="hstat-l">Algorithm</div>
            </div>
            <div class="hstat">
                <div class="hstat-n">ML</div>
                <div class="hstat-l">Powered</div>
            </div>
        </div>
    </div>
    <div class="geo" aria-hidden="true">
        <div class="geo-circle"><div class="geo-dot"></div></div>
        <div class="geo-circle"></div>
        <div class="geo-circle"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS STRIP
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="process">
    <div class="proc-step">
        <div class="proc-num">01</div>
        <div>
            <div class="proc-t">Soil Nutrients</div>
            <div class="proc-s">Input N, P, K values</div>
        </div>
    </div>
    <div class="proc-step">
        <div class="proc-num">02</div>
        <div>
            <div class="proc-t">Climate Data</div>
            <div class="proc-s">Temperature, humidity, rainfall</div>
        </div>
    </div>
    <div class="proc-step">
        <div class="proc-num">03</div>
        <div>
            <div class="proc-t">Soil Chemistry</div>
            <div class="proc-s">pH acidity index</div>
        </div>
    </div>
    <div class="proc-step">
        <div class="proc-num">04</div>
        <div>
            <div class="proc-t">AI Recommendation</div>
            <div class="proc-s">Best crop with confidence</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FORM SECTION HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section">
    <div class="s-eyebrow">Input Parameters</div>
    <div class="s-title">Enter Soil &amp; Climate Data</div>
    <div class="s-rule"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# THREE COLUMN INPUTS
# ─────────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="icard">
        <div class="icard-head">
            <div class="icard-ico">
                <svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 0 1 10 10c0 5.5-4.5 10-10 10S2 17.5 2 12 6.5 2 12 2z"/><path d="M12 6v6l4 2"/></svg>
            </div>
            <div>
                <div class="icard-t">Macronutrients</div>
                <div class="icard-s">NPK soil composition (mg / kg)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    N = st.number_input("Nitrogen (N)",   value=90.0,  min_value=0.0, max_value=200.0, step=0.1)
    P = st.number_input("Phosphorus (P)", value=42.0,  min_value=0.0, max_value=200.0, step=0.1)
    K = st.number_input("Potassium (K)",  value=43.0,  min_value=0.0, max_value=200.0, step=0.1)

with c2:
    st.markdown("""
    <div class="icard">
        <div class="icard-head">
            <div class="icard-ico">
                <svg viewBox="0 0 24 24"><path d="M17 18a5 5 0 0 0-10 0"/><line x1="12" y1="2" x2="12" y2="9"/><line x1="4.22" y1="10.22" x2="5.64" y2="11.64"/><line x1="1" y1="18" x2="3" y2="18"/><line x1="21" y1="18" x2="23" y2="18"/><line x1="18.36" y1="11.64" x2="19.78" y2="10.22"/><line x1="23" y1="22" x2="1" y2="22"/><polyline points="8 6 12 2 16 6"/></svg>
            </div>
            <div>
                <div class="icard-t">Climate Conditions</div>
                <div class="icard-s">Local environmental measurements</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    temperature = st.number_input("Temperature (°C)", value=20.5,  min_value=-10.0, max_value=55.0,  step=0.1)
    humidity    = st.number_input("Humidity (%)",     value=82.0,  min_value=0.0,   max_value=100.0, step=0.1)
    rainfall    = st.number_input("Rainfall (mm)",    value=202.9, min_value=0.0,   max_value=500.0, step=0.1)

with c3:
    st.markdown("""
    <div class="icard">
        <div class="icard-head">
            <div class="icard-ico">
                <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>
            </div>
            <div>
                <div class="icard-t">Soil Chemistry</div>
                <div class="icard-s">Acidity / alkalinity index</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    ph = st.number_input("Soil pH  (0 – 14)", value=6.5, min_value=0.0, max_value=14.0, step=0.01)
    st.markdown("""
    <div class="ph-ref">
        <div class="ph-ref-title">pH Reference Guide</div>
        <div class="ph-row"><span class="ph-val">0 – 4</span><span>Strongly Acidic</span></div>
        <div class="ph-row"><span class="ph-val">4 – 6</span><span>Moderately Acidic</span></div>
        <div class="ph-row"><span class="ph-val">6 – 7</span><span>Slightly Acidic</span></div>
        <div class="ph-row"><span class="ph-val">7.0</span><span>Neutral</span></div>
        <div class="ph-row"><span class="ph-val">7 – 8</span><span>Slightly Alkaline</span></div>
        <div class="ph-row"><span class="ph-val">8 – 14</span><span>Strongly Alkaline</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='padding:32px 72px 16px;'>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 1.4, 1])
with btn_col:
    clicked = st.button("Analyze and Recommend Crop")
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if clicked:
    input_df = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=FEATURE_COLS
    )

    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    model_input = np.array(transformed)

    prediction   = model.predict(model_input)[0]
    probabilities = model.predict_proba(model_input)[0]
    classes       = model.classes_

    top_n       = 8
    sorted_idx  = np.argsort(probabilities)[::-1][:top_n]
    top_classes = [classes[i] for i in sorted_idx]
    top_probs   = [probabilities[i] * 100 for i in sorted_idx]

    # ── RESULT BANNER
    st.markdown(f"""
    <div style="padding:0 72px 32px;">
        <div class="result-banner">
            <div>
                <div class="r-eye">Recommendation Result</div>
                <div class="r-name">{prediction}</div>
                <div class="r-desc">
                    The Random Forest model analyzed your 7 soil and climate
                    parameters and identified <strong style="color:#fff">{prediction.title()}</strong>
                    as the most suitable crop for your growing conditions.
                </div>
            </div>
            <div class="r-params">
                <div class="r-param">
                    <div class="r-param-v">{N}</div>
                    <div class="r-param-k">Nitrogen</div>
                </div>
                <div class="r-param">
                    <div class="r-param-v">{P}</div>
                    <div class="r-param-k">Phosphorus</div>
                </div>
                <div class="r-param">
                    <div class="r-param-v">{K}</div>
                    <div class="r-param-k">Potassium</div>
                </div>
                <div class="r-param">
                    <div class="r-param-v">{ph}</div>
                    <div class="r-param-k">Soil pH</div>
                </div>
                <div class="r-param">
                    <div class="r-param-v">{temperature}°</div>
                    <div class="r-param-k">Temperature</div>
                </div>
                <div class="r-param">
                    <div class="r-param-v">{humidity}%</div>
                    <div class="r-param-k">Humidity</div>
                </div>
                <div class="r-param wide">
                    <div class="r-param-v">{rainfall} mm</div>
                    <div class="r-param-k">Rainfall</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CROP INFO CARD + CONFIDENCE CHART side by side
    info = CROP_INFO.get(prediction.lower(), None)

    st.markdown("<div style='padding:0 72px;'>", unsafe_allow_html=True)

    if info:
        st.markdown(f"""
        <div style="margin-bottom:16px;">
            <div class="s-eyebrow">Crop Profile</div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:28px;font-weight:400;color:var(--g1);margin-top:6px;text-transform:capitalize;">{prediction} — Growing Guide</div>
        </div>
        <div class="crop-info-grid" style="margin-bottom:32px;">
            <div class="cinfo">
                <div class="cinfo-ico-wrap">
                    <svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
                </div>
                <div class="cinfo-label">Growing Season</div>
                <div class="cinfo-value">{info['season']}</div>
            </div>
            <div class="cinfo">
                <div class="cinfo-ico-wrap">
                    <svg viewBox="0 0 24 24"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>
                </div>
                <div class="cinfo-label">Water Requirement</div>
                <div class="cinfo-value">{info['water']}</div>
            </div>
            <div class="cinfo">
                <div class="cinfo-ico-wrap">
                    <svg viewBox="0 0 24 24"><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z"/></svg>
                </div>
                <div class="cinfo-label">Ideal Temperature</div>
                <div class="cinfo-value">{info['temp']}</div>
            </div>
            <div class="cinfo">
                <div class="cinfo-ico-wrap">
                    <svg viewBox="0 0 24 24"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
                </div>
                <div class="cinfo-label">Suitable Soil Type</div>
                <div class="cinfo-value">{info['soil']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── CONFIDENCE CHART
    st.markdown("""
    <div class="chart-wrapper">
        <div class="chart-head">
            <div class="chart-title">Model Confidence Distribution</div>
            <div class="chart-sub">Top 8 crop probabilities predicted by the Random Forest classifier</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    colors = ["#1e5c3a" if c.lower() == prediction.lower() else "#a8e6c1" for c in top_classes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_probs,
        y=[c.title() for c in top_classes],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{p:.1f}%" for p in top_probs],
        textposition="outside",
        textfont=dict(family="DM Sans", size=12, color="#0c1f14"),
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>",
    ))

    fig.update_layout(
        height=340,
        margin=dict(l=0, r=60, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans", size=12, color="#4a7060"),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(30,92,58,0.07)",
            ticksuffix="%",
            range=[0, max(top_probs) * 1.22],
            zeroline=False,
            showline=False,
        ),
        yaxis=dict(
            showgrid=False,
            autorange="reversed",
            tickfont=dict(size=13, color="#0c1f14", family="DM Sans"),
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURES SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="feat-section">
    <div class="s-eyebrow">Why CropSense</div>
    <div class="s-title">Built for Modern Agriculture</div>
    <div class="feat-grid">
        <div class="feat">
            <div class="feat-ico">
                <svg viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            </div>
            <div class="feat-t">Soil-Aware Intelligence</div>
            <div class="feat-p">
                Processes Nitrogen, Phosphorus and Potassium ratios alongside
                soil pH to precisely match crop requirements to your land's
                natural chemistry and nutrient composition.
            </div>
        </div>
        <div class="feat">
            <div class="feat-ico">
                <svg viewBox="0 0 24 24"><path d="M17 18a5 5 0 0 0-10 0"/><line x1="12" y1="2" x2="12" y2="9"/><line x1="4.22" y1="10.22" x2="5.64" y2="11.64"/><line x1="1" y1="18" x2="3" y2="18"/><line x1="21" y1="18" x2="23" y2="18"/><line x1="18.36" y1="11.64" x2="19.78" y2="10.22"/></svg>
            </div>
            <div class="feat-t">Climate Integration</div>
            <div class="feat-p">
                Temperature, humidity and rainfall data are processed together
                to ensure the recommended crop thrives in your specific local
                climate conditions across the full growing season.
            </div>
        </div>
        <div class="feat">
            <div class="feat-ico">
                <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            </div>
            <div class="feat-t">Confidence Scoring</div>
            <div class="feat-p">
                Every recommendation is backed by a full probability
                distribution across all 22 crop classes, so you can see
                exactly how confident the model is in its output.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div>
        <div class="footer-logo">Crop<b>Sense</b></div>
        <div class="footer-copy">AI-Based Crop Recommendation System</div>
    </div>
    <div>
        <div class="footer-tags">
            <span class="ftag">Random Forest</span>
            <span class="ftag">Scikit-Learn</span>
            <span class="ftag">Streamlit</span>
            <span class="ftag">Plotly</span>
            <span class="ftag">Python</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)