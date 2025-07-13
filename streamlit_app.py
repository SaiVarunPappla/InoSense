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

# Configure Streamlit environment
os.environ['STREAMLIT_SERVER_ROOT'] = tempfile.gettempdir()
os.environ['STREAMLIT_CONFIG_DIR'] = str(Path(tempfile.gettempdir()) / '.streamlit')
os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'

# Page configuration
st.set_page_config(
    page_title="InoSense AI | Next-Gen Chemical Plant Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for engaging animations and futuristic design
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Orbitron', 'Roboto', sans-serif;
    background: linear-gradient(135deg, #0f0f1a, #1a2a44);
    color: #e0f7fa;
    overflow-x: hidden;
}

.stMetric {
    border-radius: 15px;
    padding: 20px;
    background: rgba(255, 255, 255, 0.08);
    box-shadow: 0 5px 15px rgba(0, 191, 255, 0.2);
    transition: all 0.5s ease;
}
.stMetric:hover {
    transform: translateY(-10px) scale(1.05);
    box-shadow: 0 10px 25px rgba(0, 191, 255, 0.4);
}

@keyframes pulseGlow {
    0% { box-shadow: 0 0 5px #00ffcc, 0 0 10px #00ffcc; }
    50% { box-shadow: 0 0 15px #00ffcc, 0 0 20px #00ffcc; }
    100% { box-shadow: 0 0 5px #00ffcc, 0 0 10px #00ffcc; }
}
.emergency-alert {
    animation: pulseGlow 2s infinite;
    background: linear-gradient(45deg, #ff3333, #ff6666);
    color: #fff;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-weight: 700;
}

.plot-container {
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 191, 255, 0.3);
    overflow: hidden;
    transition: all 0.6s ease;
}
.plot-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0, 191, 255, 0.5);
}

.stButton>button {
    background: linear-gradient(45deg, #00c4cc, #00eaff);
    color: #fff;
    border: none;
    padding: 12px 25px;
    border-radius: 10px;
    font-size: 1.1em;
    transition: all 0.5s ease;
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.1);
    box-shadow: 0 8px 20px rgba(0, 234, 255, 0.6);
    background: linear-gradient(45deg, #00eaff, #00c4cc);
}

@keyframes slideIn {
    from { transform: translateX(-100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}
.stTabs [data-baseweb="tab-list"] {
    animation: slideIn 0.7s ease-out;
}
</style>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.title("🖥️ Control Hub")
    dark_mode = st.toggle("🌌 Dark Mode", True)
    if dark_mode:
        st.markdown("<style>body {background-color: #0f0f1a;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body {background-color: #e0f7fa; color: #1a2a44;}</style>", unsafe_allow_html=True)

    emergency = st.button("🚨 Emergency Mode", help="Simulate critical failure", use_container_width=True)
    if emergency:
        st.session_state.emergency = True
        st.toast("🔴 Emergency mode activated!", icon="🚨")

    model_version = st.selectbox("🤖 AI Model", ["Isolation Forest v2.0", "LSTM v5.0", "Neural Net v3.0"], index=0)
    if st.button("🎙️ Voice Command", use_container_width=True):
        with st.spinner("Analyzing voice input..."):
            time.sleep(1.2)
            st.success("Command received: 'Display 3D model'")

    st.divider()
    st.markdown(f"""
    **Developer:** Varun  
    **Version:** 1.2.0  
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M IST')}  
    **GitHub:** [InoSense](https://github.com/yourusername/inosense)  
    **LinkedIn:** [Profile](https://linkedin.com/in/yourprofile)
    """)

# Main title and intro with animation
st.markdown("<h1 style='animation: slideIn 1s ease-out; text-align: center;'>🏭 InoSense AI: Next-Gen Plant Monitoring</h1>", unsafe_allow_html=True)
st.caption("Cutting-edge AI for real-time optimization and 3D visualization")

# Simulated plant data with enhanced realism
class PlantSimulator:
    def __init__(self):
        self.state = {"temperature": 347.0, "flow_rate": 42.0, "energy_usage": 4.2, "risk_score": 45.0}
        self.history = pd.DataFrame(columns=["time", "temperature", "flow_rate", "energy_usage", "risk_score"])

    def update(self):
        t = time.time()
        self.state["temperature"] = 347 + 5 * np.sin(t * 0.1) + np.random.normal(0, 1.5)
        self.state["flow_rate"] = 42 + 2 * np.cos(t * 0.15) + np.random.normal(0, 0.7)
        self.state["energy_usage"] = 4.2 - 0.1 * np.sin(t * 0.2) + np.random.normal(0, 0.07)
        self.state["risk_score"] = min(100, max(0, 45 + 15 * np.random.normal(0, 1)))
        self.history.loc[len(self.history)] = [datetime.now(), *self.state.values()]
        return self.state

plant = PlantSimulator()

# AI Prediction Engine
class AIPredictor:
    def __init__(self):
        self.model = IsolationForest(n_estimators=150, contamination=0.1)
        self._fit_initial_model()

    def _fit_initial_model(self):
        np.random.seed(42)
        temp_data = np.random.normal(347, 15, 150)
        flow_data = np.random.normal(42, 3, 150)
        energy_data = np.random.normal(4.2, 0.2, 150)
        training_data = np.column_stack((temp_data, flow_data, energy_data))
        self.model.fit(training_data)

    def predict(self, state):
        features = np.array([state["temperature"], state["flow_rate"], state["energy_usage"]]).reshape(1, -1)
        anomaly_score = self.model.score_samples(features)[0]
        return {
            "risk_score": abs(anomaly_score) * 100,
            "is_anomaly": anomaly_score < -0.6,
            "recommendation": self._get_recommendation(abs(anomaly_score))
        }

    def _get_recommendation(self, anomaly_score):
        if anomaly_score > 1.8:
            return "CRITICAL: Immediate shutdown and team alert"
        elif anomaly_score > 1.2:
            return "WARNING: Adjust settings and inspect"
        else:
            return "STABLE: Maintain routine checks"

ai_predictor = AIPredictor()

# Real-time metrics with animations
metrics_container = st.empty()
with metrics_container.container():
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌡️ Reactor Temp", f"{plant.state['temperature']:.1f}°K", f"{np.random.uniform(-2, 2):.1f}°C",
                  help="AI-monitored temperature")
    with col2:
        st.metric("🔄 Flow Rate", f"{plant.state['flow_rate']:.1f} L/s", f"{np.random.uniform(-1, 1):.1f}%")
    with col3:
        st.metric("⚡ Energy Use", f"{plant.state['energy_usage']:.1f} MW", f"{np.random.uniform(-0.3, 0.3):.1f} MW")
    with col4:
        st.metric("⚠️ Risk Level", f"{'High' if plant.state['risk_score'] > 70 else 'Medium' if plant.state['risk_score'] > 30 else 'Low'}",
                  f"{plant.state['risk_score']:.0f}%", delta_color="inverse")

# Enhanced 3D Visualizations
def create_3d_plant(state):
    x, y = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
    z = np.sin(x) * np.cos(y) * (state["temperature"] / 500) + np.random.normal(0, 0.1, (20, 20))
    fig = go.Figure(data=[go.Surface(z=z, colorscale='Viridis', opacity=0.9)])
    fig.add_trace(go.Scatter3d(
        x=[5], y=[5], z=[z[10, 10]], mode='markers',
        marker=dict(size=10, color='red', symbol='diamond', line=dict(width=2, color='DarkSlateGrey')),
        hovertext=f"Core Temp: {state['temperature']:.1f}°K"
    ))
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Dynamic 3D Plant Model"
    )
    return fig

def create_risk_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "AI Risk Matrix"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': value}
        }
    ))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=250)
    return fig

# Tab layout with transitions
tab1, tab2, tab3 = st.tabs(["📊 Live Feed", "🤖 AI Insights", "📁 Analytics"])

with tab1:
    st.subheader("Real-Time Plant Visualization")
    st.plotly_chart(create_3d_plant(plant.state), use_container_width=True, config={'displayModeBar': False})
    if st.session_state.get('emergency'):
        st.markdown('<div class="emergency-alert">🚨 CRITICAL ALERT: SYSTEM OVERLOAD DETECTED</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("AI-Driven Predictive Analytics")
    prediction = ai_predictor.predict(plant.state)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(create_risk_gauge(prediction["risk_score"]), use_container_width=True)
    with col2:
        st.info(f"""
        **AI Directive:** {prediction["recommendation"]}  
        - Anomaly Status: {prediction['is_anomaly']}  
        - Confidence: {min(95, max(50, 100 - prediction['risk_score'])):.0f}%
        """)
        if st.button("🔧 Auto-Optimize", use_container_width=True):
            with st.spinner("Executing AI optimization..."):
                time.sleep(1.5)
                st.success(f"Optimization success! Saved {np.random.uniform(10, 20):.1f}% energy")

with tab3:
    st.subheader("Operational Reports")
    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="InoSense AI Operational Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}", ln=1)
        pdf.cell(200, 10, txt=f"Risk Level: {plant.state['risk_score']:.1f}%", ln=1)
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(f"InoSense Report - {datetime.now()}")
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white')
        img.save("report_qr.png")
        pdf.image("report_qr.png", x=50, w=100)
        return pdf.output(dest='S').encode('latin1')

    st.download_button(
        label="📥 Download Report (PDF)",
        data=generate_report(),
        file_name=f"inosense_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )
    st.image("https://via.placeholder.com/800x400?text=AI+Trend+Forecast", caption="Predictive Trend Analysis")

# Real-time update with polling
refresh_placeholder = st.empty()
while True:
    with refresh_placeholder.container():
        plant.update()
        with metrics_container.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🌡️ Reactor Temp", f"{plant.state['temperature']:.1f}°K", f"{np.random.uniform(-2, 2):.1f}°C")
            with col2:
                st.metric("🔄 Flow Rate", f"{plant.state['flow_rate']:.1f} L/s", f"{np.random.uniform(-1, 1):.1f}%")
            with col3:
                st.metric("⚡ Energy Use", f"{plant.state['energy_usage']:.1f} MW", f"{np.random.uniform(-0.3, 0.3):.1f} MW")
            with col4:
                st.metric("⚠️ Risk Level", f"{'High' if plant.state['risk_score'] > 70 else 'Medium' if plant.state['risk_score'] > 30 else 'Low'}",
                          f"{plant.state['risk_score']:.0f}%", delta_color="inverse")
        time.sleep(2)  # Update every 2 seconds

# Footer with engagement
st.divider()
st.markdown("<h4 style='text-align: center; color: #00eaff;'>© 2025 InoSense AI | Pioneering Industrial Intelligence</h4>", unsafe_allow_html=True)
st.caption("""
Built with Streamlit, scikit-learn, and Plotly  
[GitHub](https://github.com/yourusername/inosense) | [Live Demo](your-app.streamlit.app) | [Connect](mailto:your.email@example.com)
""")