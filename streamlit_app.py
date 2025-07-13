import os
import tempfile
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time
from datetime import datetime
from io import BytesIO
from fpdf import FPDF
import qrcode
from sklearn.ensemble import IsolationForest
import streamlit.components.v1 as components

# Configure Streamlit environment
os.environ['STREAMLIT_SERVER_ROOT'] = tempfile.gettempdir()
os.environ['STREAMLIT_CONFIG_DIR'] = str(Path(tempfile.gettempdir()) / '.streamlit')
os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'

# Page configuration
st.set_page_config(
    page_title="InoSense AI | Cyber Plant Command",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for neon cyberpunk theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Orbitron', 'Roboto', sans-serif;
    background: linear-gradient(135deg, #0a0f1a, #1a2a44, #0a1a2a);
    color: #00ffcc;
    overflow-x: hidden;
}

.stMetric {
    border-radius: 20px;
    padding: 25px;
    background: rgba(0, 255, 204, 0.1);
    box-shadow: 0 8px 20px rgba(0, 255, 204, 0.3), 0 0 10px #00ffcc;
    transition: all 0.6s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.stMetric:hover {
    transform: translateY(-12px) scale(1.1);
    box-shadow: 0 15px 35px rgba(0, 255, 204, 0.5), 0 0 20px #00ffcc;
}

@keyframes neonPulse {
    0% { box-shadow: 0 0 5px #00ffcc, 0 0 15px #00ffcc; }
    50% { box-shadow: 0 0 20px #00ffcc, 0 0 30px #ff00ff; }
    100% { box-shadow: 0 0 5px #00ffcc, 0 0 15px #00ffcc; }
}
.emergency-alert {
    animation: neonPulse 2.5s infinite;
    background: linear-gradient(45deg, #ff1a1a, #ff4d4d, #ff00ff);
    color: #fff;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-weight: 700;
    text-shadow: 0 0 10px #fff, 0 0 20px #ff00ff;
}

.plot-container {
    border-radius: 25px;
    box-shadow: 0 12px 40px rgba(0, 255, 204, 0.4), 0 0 15px #00ffcc;
    overflow: hidden;
    transition: all 0.7s ease-in-out;
}
.plot-container:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 20px 50px rgba(0, 255, 204, 0.6), 0 0 25px #00ffcc;
}

.stButton>button {
    background: linear-gradient(45deg, #00ccff, #ff00ff, #00ffcc);
    color: #fff;
    border: none;
    padding: 15px 30px;
    border-radius: 12px;
    font-size: 1.2em;
    text-transform: uppercase;
    letter-spacing: 1px;
    transition: all 0.6s ease;
}
.stButton>button:hover {
    transform: translateY(-5px) scale(1.15);
    box-shadow: 0 10px 30px rgba(0, 255, 204, 0.7), 0 0 30px #ff00ff;
    background: linear-gradient(45deg, #ff00ff, #00ffcc, #00ccff);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.stTabs [data-baseweb="tab-list"], .stHeader {
    animation: fadeInUp 1.5s ease-out;
}

.progress-bar {
    height: 20px;
    background: #00ffcc;
    border-radius: 10px;
    box-shadow: 0 0 10px #00ffcc;
    transition: width 1s ease;
}
</style>
""", unsafe_allow_html=True)

# Sidebar with interactive controls
with st.sidebar:
    st.title("🌠 Cyber Command Hub")
    dark_mode = st.toggle("☀️ Neon Mode", False)
    if dark_mode:
        st.markdown("<style>body {background-color: #e0f7fa; color: #1a2a44;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body {background-color: #0a0f1a;}</style>", unsafe_allow_html=True)

    emergency = st.button("🚨 Critical Breach", help="Simulate system failure", use_container_width=True)
    if emergency:
        st.session_state.emergency = True
        st.toast("🔴 Breach detected!", icon="🚨")
        components.html("<audio autoplay><source src='https://www.soundjay.com/buttons/beep-01a.mp3' type='audio/mpeg'></audio>")

    ai_mode = st.selectbox("🧠 AI Core", ["Isolation Forest v2.0", "DeepNet v3.5", "Quantum Hybrid v1.0"], index=0)
    zoom_level = st.slider("🔍 Holo-Zoom", min_value=1.0, max_value=2.5, value=1.7, step=0.1)
    voice_input = st.text_input("🎙️ Voice Command", placeholder="Say: 'Show 3D Holo'")
    if voice_input.lower() in ["show 3d holo", "display 3d"]:
        with st.spinner("Processing voice..."):
            time.sleep(1.2)
            st.success("Holo activated: 3D model displayed!")

    st.divider()
    st.markdown(f"""
    **Creator:** Varun  
    **Version:** 2.1.0  
    **Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M IST')}  
    **Portfolio:** [GitHub](https://github.com/SaiVarunPappla) | [LinkedIn](https://www.linkedin.com/in/pappla-sai-varun-874902200/)  
    **Achievements:** AI Innovation Finalist 2025, AWS Certified
    """)

# Main header with animation
st.markdown("<h1 style='text-align: center; animation: fadeInUp 1.5s ease-out; color: #00ffcc; text-shadow: 0 0 20px #00ffcc, 0 0 30px #ff00ff;'>🌌 InoSense AI: Cyber Plant Command</h1>", unsafe_allow_html=True)
st.caption("Unleashing AI-driven industrial evolution with holographic precision")

# Advanced plant simulator with streaming
class PlantSimulator:
    def __init__(self):
        self.state = {"temperature": 347.0, "flow_rate": 42.0, "energy_usage": 4.2, "risk_score": 45.0}
        self.history = pd.DataFrame(columns=["time", "temperature", "flow_rate", "energy_usage", "risk_score"])
        self.progress = 0

    def update(self):
        t = time.time()
        self.state["temperature"] = 347 + 6 * np.sin(t * 0.1) + np.random.normal(0, 2.0)
        self.state["flow_rate"] = 42 + 3 * np.cos(t * 0.15) + np.random.normal(0, 1.0)
        self.state["energy_usage"] = 4.2 - 0.15 * np.sin(t * 0.2) + np.random.normal(0, 0.1)
        self.state["risk_score"] = min(100, max(0, 45 + 20 * np.random.normal(0, 1)))
        self.progress = min(100, self.progress + np.random.uniform(0.1, 0.5))
        self.history.loc[len(self.history)] = [datetime.now(), *self.state.values()]
        return self.state

plant = PlantSimulator()

# Advanced AI predictor with simulated deep learning
class AIPredictor:
    def __init__(self):
        self.model = IsolationForest(n_estimators=200, contamination=0.1)
        self._fit_initial_model()

    def _fit_initial_model(self):
        np.random.seed(42)
        temp_data = np.random.normal(347, 20, 200)
        flow_data = np.random.normal(42, 4, 200)
        energy_data = np.random.normal(4.2, 0.3, 200)
        training_data = np.column_stack((temp_data, flow_data, energy_data))
        self.model.fit(training_data)

    def predict(self, state):
        features = np.array([state["temperature"], state["flow_rate"], state["energy_usage"]]).reshape(1, -1)
        anomaly_score = self.model.score_samples(features)[0]
        deep_score = self._simulate_deep_learning(features)
        return {
            "risk_score": (abs(anomaly_score) * 60 + deep_score * 40),
            "is_anomaly": anomaly_score < -0.7 or deep_score > 1.5,
            "recommendation": self._get_recommendation(abs(anomaly_score), deep_score)
        }

    def _simulate_deep_learning(self, features):
        return np.tanh(features[0][0] / 100 + features[0][1] / 20 - features[0][2]) * 2.5

    def _get_recommendation(self, anomaly_score, deep_score):
        total_risk = anomaly_score + deep_score
        if total_risk > 3.2:
            return "CRITICAL: Emergency lockdown initiated"
        elif total_risk > 2.2:
            return "WARNING: Adjust systems, notify team"
        else:
            return "STABLE: AI optimization active"

ai_predictor = AIPredictor()

# Real-time metrics with streaming
metrics_container = st.empty()
def update_metrics():
    with metrics_container.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🌡️ Core Temp", f"{plant.state['temperature']:.1f}°K", f"{np.random.uniform(-2, 2):.1f}°C")
        with col2:
            st.metric("🔄 Flow Rate", f"{plant.state['flow_rate']:.1f} L/s", f"{np.random.uniform(-1, 1):.1f}%")
        with col3:
            st.metric("⚡ Power Use", f"{plant.state['energy_usage']:.1f} MW", f"{np.random.uniform(-0.3, 0.3):.1f} MW")
        with col4:
            st.metric("⚠️ Risk Index", f"{'High' if plant.state['risk_score'] > 70 else 'Medium' if plant.state['risk_score'] > 30 else 'Low'}",
                      f"{plant.state['risk_score']:.0f}%", delta_color="inverse")
        st.progress(plant.progress)

# Advanced 3D holographic visualization
def create_3d_holo(state, zoom):
    x, y = np.meshgrid(np.linspace(0, 10, 40), np.linspace(0, 10, 40))
    z = np.sin(x) * np.cos(y) * (state["temperature"] / 500) + np.random.normal(0, 0.2, (40, 40))
    fig = go.Figure(data=[
        go.Surface(z=z, colorscale='Plasma', opacity=0.85, showscale=False),
        go.Scatter3d(
            x=[5 + np.random.normal(0, 0.5)], y=[5 + np.random.normal(0, 0.5)], z=[z[20, 20] + np.random.normal(0, 0.5)],
            mode='markers', marker=dict(size=15, color='cyan', symbol='diamond', line=dict(width=2, color='white')),
            hovertext=[f"Hotspot: {state['temperature']:.1f}°K"], hoverinfo="text"
        )
    ])
    fig.update_layout(
        scene=dict(xaxis_title='X-Axis', yaxis_title='Y-Axis', zaxis_title='Z-Axis',
                   camera=dict(eye=dict(x=zoom, y=zoom, z=zoom))),
        margin=dict(l=0, r=0, b=0, t=40),
        title="Holographic Plant Matrix"
    )
    return fig

def create_risk_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cyber Risk Spectrum"},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "#ff00ff"},
            'steps': [
                {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.7)"},
                {'range': [30, 70], 'color': "rgba(255, 255, 0, 0.7)"},
                {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.7)"}],
            'threshold': {'line': {'color': "#ff00ff", 'width': 5}, 'thickness': 1.0, 'value': value}
        }
    ))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), height=300)
    return fig

# Tab layout with gamified elements
tab1, tab2, tab3 = st.tabs(["🌍 Holo Live", "🧠 AI Nexus", "📊 Data Core"])

with tab1:
    st.subheader("Holographic Live Feed")
    st.plotly_chart(create_3d_holo(plant.state, zoom_level), use_container_width=True, config={'displayModeBar': True})
    if st.session_state.get('emergency'):
        st.markdown('<div class="emergency-alert">🚨 CYBER ALERT: SYSTEM BREACH DETECTED</div>', unsafe_allow_html=True)
        components.html("<audio autoplay><source src='https://www.soundjay.com/buttons/beep-02.mp3' type='audio/mpeg'></audio>")

with tab2:
    st.subheader("AI-Powered Nexus Insights")
    prediction = ai_predictor.predict(plant.state)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(create_risk_gauge(prediction["risk_score"]), use_container_width=True)
    with col2:
        st.info(f"""
        **AI Directive:** {prediction["recommendation"]}  
        - Anomaly Status: {prediction['is_anomaly']}  
        - Confidence: {min(98, max(60, 100 - prediction['risk_score'])):.0f}%  
        - Quantum Score: {ai_predictor._simulate_deep_learning(np.array([[plant.state['temperature'], plant.state['flow_rate'], plant.state['energy_usage']]])):.2f}
        """)
        if st.button("🚀 Quantum Optimize", use_container_width=True):
            with st.spinner("Initiating quantum sync..."):
                time.sleep(1.5)
                st.success(f"Optimization complete! Efficiency boost: {np.random.uniform(15, 25):.1f}%")
                plant.progress = 0  # Reset progress for gamification

with tab3:
    st.subheader("Data Core Analytics")
    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="InoSense AI Cyber Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}", ln=1)
        pdf.cell(200, 10, txt=f"Risk Index: {plant.state['risk_score']:.1f}%", ln=1)
        pdf.cell(200, 10, txt=f"AI Confidence: {min(98, max(60, 100 - prediction['risk_score'])):.0f}%", ln=2)
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(f"InoSense Report - {datetime.now()}")
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white')
        img.save("report_qr.png")
        pdf.image("report_qr.png", x=50, w=100)
        return pdf.output(dest='S').encode('latin1')

    st.download_button(
        label="📥 Export Cyber Report",
        data=generate_report(),
        file_name=f"inosense_cyber_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )
    st.image("https://via.placeholder.com/800x400?text=Cyber+Trend+Forecast", caption="AI-Powered Future Insights")
    st.markdown(f'<div class="progress-bar" style="width: {plant.progress}%;"></div>', unsafe_allow_html=True)
    if plant.progress >= 100:
        st.success("🎉 Achievement Unlocked: Optimization Master!")

# Real-time streaming with smooth updates
def stream_data():
    while True:
        plant.update()
        update_metrics()
        time.sleep(2)  # Stream every 2 seconds

# Start streaming
if __name__ == "__main__":
    stream_data()

# Interactive footer with showcase
st.divider()
st.markdown("<h4 style='text-align: center; color: #ff00ff; text-shadow: 0 0 20px #ff00ff, 0 0 30px #00ffcc;'>© 2025 InoSense AI | Mastering the Future of Industry</h4>", unsafe_allow_html=True)
st.caption("""
Crafted by Varun | Tech: Streamlit, scikit-learn, Plotly, Python  
[GitHub](https://github.com/SaiVarunPappla) | [Live Demo](https://inosense-xt2pxk8gveb5zvnm388qyu.streamlit.app/) | [LinkedIn](https://www.linkedin.com/in/pappla-sai-varun-874902200/)  
**Highlights:** AI Innovation Finalist 2025, Full-Stack AI Expert, Cloud-Deployed Solution
""")