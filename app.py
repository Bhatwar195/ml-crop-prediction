import streamlit as st
import pandas as pd
import joblib
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title=" AI Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# LOAD CSS
# ============================================
def load_css():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(BASE_DIR, "assets", "style.css")
    
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback basic CSS
        st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            [data-testid="stSidebar"] {display: none;}
            .stApp {
                background: linear-gradient(135deg, #f8fdf9 0%, #ffffff 50%, #f0f7f4 100%);
            }
        </style>
        """, unsafe_allow_html=True)

load_css()

# ============================================
# CROP EMOJI MAPPING
# ============================================
CROP_EMOJIS = {
    'rice': '🍚', 'wheat': '🌾', 'maize': '🌽', 'cotton': '🧵',
    'sugarcane': '🎋', 'coffee': '☕', 'mango': '🥭', 'banana': '🍌',
    'apple': '🍎', 'grapes': '🍇', 'orange': '🍊', 'papaya': '🍈',
    'coconut': '🥥', 'watermelon': '🍉', 'muskmelon': '🍈',
    'lentil': '🫘', 'chickpea': '🫘', 'kidneybeans': '🫘',
    'pigeonpeas': '🫛', 'mothbeans': '🫘', 'mungbean': '🫛',
    'blackgram': '🫘', 'pomegranate': '🍎', 'jute': '🌿', 'default': '🌱'
}

def get_crop_emoji(crop_name):
    return CROP_EMOJIS.get(crop_name.lower(), CROP_EMOJIS['default'])

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "models", "crop_model.pkl")
        preprocessor_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            return model, preprocessor, True
        return None, None, False
    except Exception as e:
        return None, None, False

model, preprocessor, model_loaded = load_model()

# ============================================
# NAVBAR
# ============================================
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <span class="navbar-logo">🌱</span>
        <span class="navbar-title">CropAI</span>
    </div>
    <div class="navbar-links">
        <span class="navbar-link">Home</span>
        <span class="navbar-link">Predict</span>
        <span class="navbar-link">About</span>
        <span class="navbar-link">Guide</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# HERO SECTION
# ============================================
st.markdown("""
<div class="hero-section">
    <span class="hero-badge">🚀 AI-Powered Agriculture</span>
    <span class="hero-emoji">🌾</span>
    <h1 class="hero-title">Crop Recommendation System</h1>
    <p class="hero-subtitle">
        Harness the power of Artificial Intelligence to make smarter farming decisions. 
        Get instant, accurate crop recommendations based on your soil composition and climate conditions.
    </p>
    <div class="hero-stats">
        <div class="hero-stat">
            <span class="hero-stat-number">22+</span>
            <span class="hero-stat-label">Crops</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-number">95%</span>
            <span class="hero-stat-label">Accuracy</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-number">7</span>
            <span class="hero-stat-label">Parameters</span>
        </div>
        <div class="hero-stat">
            <span class="hero-stat-number">24/7</span>
            <span class="hero-stat-label">Available</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================
# FEATURES SECTION
# ============================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">✨</span>
    <h2 class="section-title">Why Choose CropAI?</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🧠</span>
        <h3 class="feature-title">AI Powered</h3>
        <p class="feature-desc">Advanced ML algorithms trained on extensive agricultural datasets for accurate predictions</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">⚡</span>
        <h3 class="feature-title">Instant Results</h3>
        <p class="feature-desc">Get crop recommendations in seconds with our optimized real-time analysis engine</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🎯</span>
        <h3 class="feature-title">High Accuracy</h3>
        <p class="feature-desc">95%+ accuracy rate ensuring reliable recommendations you can trust</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="feature-card">
        <span class="feature-icon">🌍</span>
        <h3 class="feature-title">Eco-Friendly</h3>
        <p class="feature-desc">Promote sustainable farming practices with data-driven crop selection</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ============================================
# PREDICTION SECTION
# ============================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">🔬</span>
    <h2 class="section-title">Crop Prediction</h2>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.markdown("""
    <div class="error-box">
        <span class="error-icon">⚠️</span>
        <h3 class="error-title">Model Not Found</h3>
        <p class="error-text">Please ensure the model files are in the 'models' folder:</p>
        <code class="error-code">
            models/crop_model.pkl<br>
            models/preprocessor.pkl
        </code>
    </div>
    """, unsafe_allow_html=True)
else:
    # NPK Input Section
    st.markdown("""
    <div class="input-group">
        <div class="input-group-header">
            <span class="input-group-icon">🧬</span>
            <span class="input-group-title">NPK Values (Soil Nutrients)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        N = st.number_input(
            "🟢 Nitrogen (N) - kg/ha",
            min_value=0.0, max_value=200.0, value=90.0, step=1.0,
            help="Nitrogen content in soil (0-200 kg/ha)"
        )
    
    with col2:
        P = st.number_input(
            "🟣 Phosphorus (P) - kg/ha",
            min_value=0.0, max_value=200.0, value=42.0, step=1.0,
            help="Phosphorus content in soil (0-200 kg/ha)"
        )
    
    with col3:
        K = st.number_input(
            "🟠 Potassium (K) - kg/ha",
            min_value=0.0, max_value=300.0, value=43.0, step=1.0,
            help="Potassium content in soil (0-300 kg/ha)"
        )
    
    # Climate Input Section
    st.markdown("""
    <div class="input-group">
        <div class="input-group-header">
            <span class="input-group-icon">🌡️</span>
            <span class="input-group-title">Climate & Environmental Conditions</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temperature = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=0.0, max_value=50.0, value=20.5, step=0.1,
            help="Average temperature in Celsius"
        )
    
    with col2:
        humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=0.0, max_value=100.0, value=82.0, step=0.1,
            help="Relative humidity percentage"
        )
    
    with col3:
        ph = st.number_input(
            "⚗️ Soil pH",
            min_value=0.0, max_value=14.0, value=6.5, step=0.1,
            help="pH level of soil (0-14)"
        )
    
    with col4:
        rainfall = st.number_input(
            "🌧️ Rainfall (mm)",
            min_value=0.0, max_value=500.0, value=202.9, step=0.1,
            help="Average rainfall in mm"
        )
    
    # Predict Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_btn = st.button(" PREDICT BEST CROP", use_container_width=True)
    
    # Result
    if predict_btn:
        with st.spinner("🔄 Analyzing parameters..."):
            input_df = pd.DataFrame({
                "N": [N], "P": [P], "K": [K],
                "temperature": [temperature], "humidity": [humidity],
                "ph": [ph], "rainfall": [rainfall]
            })
            
            try:
                transformed_data = preprocessor.transform(input_df)
                prediction = model.predict(transformed_data)[0]
                emoji = get_crop_emoji(prediction)
                
                st.success("✅ Prediction Successful!")
                
                st.markdown(f"""
                <div class="result-container">
                    <span class="result-emoji">{emoji}</span>
                    <p class="result-label">Recommended Crop</p>
                    <h1 class="result-crop">{prediction}</h1>
                    <p class="result-description">
                        Based on your soil nutrients and climate conditions, 
                        <strong>{prediction.title()}</strong> is the optimal crop for your land.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Info Cards
                col1, col2 = st.columns(2)

                next_steps_html = """
                <div class="info-card">
                <h4 class="info-card-title green">✅ Next Steps</h4>

                <div class="info-item">
                <p class="info-item-title">1. Verify Soil</p>
                <p class="info-item-text">Get a professional soil test for confirmation</p>
                </div>

                <div class="info-item">
                <p class="info-item-title">2. Market Research</p>
                <p class="info-item-text">Check local demand and pricing trends</p>
                </div>

                <div class="info-item">
                <p class="info-item-title">3. Plan Irrigation</p>
                <p class="info-item-text">Design your water management system</p>
                </div>

                </div>
                """

                input_summary_html = f"""
                <div class="info-card">
                <h4 class="info-card-title orange">📊 Input Summary</h4>

                <div class="info-item orange">
                <p class="info-item-title orange">Nutrients (NPK)</p>
                <p class="info-item-text">N: {N} | P: {P} | K: {K}</p>
                </div>

                <div class="info-item orange">
                <p class="info-item-title orange">Climate</p>
                <p class="info-item-text">Temp: {temperature}°C | Humidity: {humidity}%</p>
                </div>

                <div class="info-item orange">
                <p class="info-item-title orange">Environment</p>
                <p class="info-item-text">pH: {ph} | Rainfall: {rainfall}mm</p>
                </div>

                </div>
                """

                with col1:
                    st.markdown(next_steps_html, unsafe_allow_html=True)

                with col2:
                    st.markdown(input_summary_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ============================================
# ABOUT SECTION
# ============================================

st.markdown("""
<div class="section-header">
<span class="section-icon">ℹ️</span>
<h2 class="section-title">About CropAI</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

about_mission = """
<div class="about-card">
<h3 class="about-title">🎯 Our Mission</h3>
<p class="about-text">
CropAI is an intelligent agricultural decision support system designed to 
revolutionize farming through data-driven insights and machine learning technology.
</p>
<p class="about-text">
We leverage cutting-edge AI algorithms to provide accurate, personalized 
crop recommendations that maximize yields while promoting sustainable and eco-friendly farming practices.
</p>
</div>
"""

about_working = """
<div class="about-card">
<h3 class="about-title">🔬 How It Works</h3>

<div class="info-item">
<p class="info-item-title">Step 1: Input Data</p>
<p class="info-item-text">Enter soil nutrients (N, P, K) and climate conditions</p>
</div>

<div class="info-item">
<p class="info-item-title">Step 2: AI Analysis</p>
<p class="info-item-text">Our ML model processes your data against thousands of records</p>
</div>

<div class="info-item">
<p class="info-item-title">Step 3: Get Results</p>
<p class="info-item-text">Receive instant, optimized crop recommendation</p>
</div>

</div>
"""

with col1:
    st.markdown(about_mission, unsafe_allow_html=True)

with col2:
    st.markdown(about_working, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ============================================
# CROP GUIDE SECTION
# ============================================
st.markdown("""
<div class="section-header">
    <span class="section-icon">📚</span>
    <h2 class="section-title">Crop Guide</h2>
</div>
""", unsafe_allow_html=True)

crops_data = [
    {'name': 'Rice', 'emoji': '🍚', 'n': '80-120', 'temp': '20-35°C', 'season': 'Kharif'},
    {'name': 'Wheat', 'emoji': '🌾', 'n': '100-120', 'temp': '15-25°C', 'season': 'Rabi'},
    {'name': 'Maize', 'emoji': '🌽', 'n': '80-120', 'temp': '18-32°C', 'season': 'Kharif'},
    {'name': 'Cotton', 'emoji': '🧵', 'n': '60-80', 'temp': '21-30°C', 'season': 'Kharif'},
    {'name': 'Sugarcane', 'emoji': '🎋', 'n': '150-200', 'temp': '20-35°C', 'season': 'Annual'},
    {'name': 'Coffee', 'emoji': '☕', 'n': '80-100', 'temp': '15-25°C', 'season': 'Perennial'},
]

cols = st.columns(6)
for idx, crop in enumerate(crops_data):
    with cols[idx]:
        st.markdown(f"""
        <div class="crop-card">
            <span class="crop-emoji">{crop['emoji']}</span>
            <h4 class="crop-name">{crop['name']}</h4>
            <p class="crop-info">
                <strong>N:</strong> {crop['n']}<br>
                <strong>Temp:</strong> {crop['temp']}<br>
                <strong>Season:</strong> {crop['season']}
            </p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <span class="footer-logo">🌱</span>
    <h3 class="footer-brand">CropAI</h3>
    <p class="footer-text">
        Intelligent Crop Recommendation System<br>
        Built with ❤️ using Streamlit & Machine Learning<br>
        Empowering farmers with AI-driven insights for sustainable agriculture
    </p>
    <div class="footer-links">
        <span class="footer-link">Privacy Policy</span>
        <span class="footer-link">Terms of Service</span>
        <span class="footer-link">Contact Us</span>
        <span class="footer-link">Documentation</span>
    </div>
    <p class="footer-copyright">© 2024 CropAI. All Rights Reserved.</p>
</div>
""", unsafe_allow_html=True)