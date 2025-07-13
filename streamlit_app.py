import os
import tempfile
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import time
from datetime import datetime
import base64
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
    page_title="InoSense AI | Advanced Chemical Plant Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling and animations
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
    transition: all 0.3s ease;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    color: #e0e0e0;
}

.stMetric {
    border-radius: 10px;
    padding: 15px;
    background: rgba(255, 255, 255, 0.05);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.stMetric:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.4);
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
.emergency-alert {
    animation: pulse 1.5s infinite;
    background: #ff4d4d;
    color: white;
    padding: 10px;
    border-radius: 5px;
}

.plot-container {
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    overflow: hidden;
    transition: all 0.3s ease;
}
.plot-container:hover {
    box-shadow: 0 15px 35px rgba(0,0,0,0.5);
}

.stButton>button {
    background: linear-gradient(45deg, #4facfe, #00f2fe);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 15px rgba(0, 242, 254, 0.5);
}
</style>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.title("⚙️ Control Panel")
    dark_mode = st.toggle("🌙 Dark Mode", True)
    if dark_mode:
        st.markdown("<style>body {background-color: #1a1a2e;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body {background-color: #f0f2f6; color: #333;}</style>", unsafe_allow_html=True)

    emergency = st.button("🚨 Trigger Emergency Protocol", help="Simulate a critical failure", use_container_width=True)
    if emergency:
        st.session_state.emergency = True
        st.toast("Emergency protocol activated!", icon="⚠️")

    model_version = st.selectbox("🤖 AI Model", ["LSTM v4.2", "Transformer v2.1", "Isolation Forest v1.0"], index=0)
    if st.button("🎤 Voice Command", use_container_width=True):
        with st.spinner("Listening..."):
            time.sleep(1.5)
            st.success("Voice command received: 'Show risk assessment'")

    st.divider()
    st.markdown(f"""
    **Developer:** Varun  
    **Version:** 1.1.0  
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M IST')}  
    **GitHub:** [InoSense Repo](https://github.com/yourusername/inosense)
    """)

# Main title and intro
st.title("🏭 InoSense AI: Intelligent Chemical Plant Monitoring")
st.caption("Real-time AI-driven optimization with predictive analytics and 3D visualization")

# Simulated plant data
class PlantSimulator:
    def __init__(self):
        self.state = {
            "temperature": 347.0,
            "flow_rate": 42.0,
            "energy_usage": 4.2,
            "risk_score": 45.0
        }
        self.history = pd.DataFrame(columns=["time", "temperature", "flow_rate", "energy_usage", "risk_score"])

    def update(self):
        t = time.time()
        self.state["temperature"] = 347 + 5 * np.sin(t * 0.1) + np.random.normal(0, 1)
        self.state["flow_rate"] = 42 + 2 * np.cos(t * 0.15) + np.random.normal(0, 0.5)
        self.state["energy_usage"] = 4.2 - 0.1 * np.sin(t * 0.2) + np.random.normal(0, 0.05)
        self.state["risk_score"] = min(100, max(0, 45 + 10 * np.random.normal(0, 1)))
        self.history.loc[len(self.history)] = [datetime.now(), *self.state.values()]
        return self.state

plant = PlantSimulator()

# AI Prediction Engine
class AIPredictor:
    def __init__(self):
        self.model = IsolationForest(n_estimators=100, contamination=0.1)
        # Fit the model with initial synthetic data
        self._fit_initial_model()

    def _fit_initial_model(self):
        # Generate synthetic training data (e.g., 100 samples)
        np.random.seed(42)
        temp_data = np.random.normal(347, 10, 100)
        flow_data = np.random.normal(42, 2, 100)
        energy_data = np.random.normal(4.2, 0.1, 100)
        training_data = np.column_stack((temp_data, flow_data, energy_data))
        self.model.fit(training_data)

    def predict(self, state):
        features = np.array([state["temperature"], state["flow_rate"], state["energy_usage"]]).reshape(1, -1)
        anomaly_score = self.model.score_samples(features)[0]
        return {
            "risk_score": abs(anomaly_score) * 100,
            "is_anomaly": anomaly_score < -0.5,
            "recommendation": self._get_recommendation(abs(anomaly_score))
        }

    def _get_recommendation(self, anomaly_score):
        if anomaly_score > 1.5:
            return "CRITICAL: Immediate shutdown and maintenance required"
        elif anomaly_score > 1.0:
            return "WARNING: Adjust parameters and schedule inspection"
        else:
            return "NORMAL: Continue operations with routine checks"

ai_predictor = AIPredictor()

# Real-time metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🌡️ Reactor Temp", f"{plant.state['temperature']:.1f}°K", f"{np.random.uniform(-2, 2):.1f}°C",
              help="Real-time reactor temperature with AI monitoring")
with col2:
    st.metric("🔄 Flow Rate", f"{plant.state['flow_rate']:.1f} L/s", f"{np.random.uniform(-1, 1):.1f}%")
with col3:
    st.metric("⚡ Energy Usage", f"{plant.state['energy_usage']:.1f} MW", f"{np.random.uniform(-0.3, 0.3):.1f} MW")
with col4:
    st.metric("⚠️ Risk Level", f"{'High' if plant.state['risk_score'] > 70 else 'Medium' if plant.state['risk_score'] > 30 else 'Low'}",
              f"{plant.state['risk_score']:.0f}%", delta_color="inverse")

# Visualizations
def create_risk_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "AI Risk Assessment"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': value}
        }
    ))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=200)
    return fig

def create_3d_plant(state):
    x, y, z = np.random.rand(3, 50) * 10
    temp_color = state["temperature"] / 500  # Normalize to 0-1
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=8, color=temp_color, colorscale='thermal', opacity=0.8, cmin=0, cmax=1)
    )])
    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', camera=dict(eye=dict(x=1.5, y=1.5, z=0.7))),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Real-Time 3D Plant Model"
    )
    return fig

# Tab layout
tab1, tab2, tab3 = st.tabs(["📊 Live Monitoring", "🤖 AI Insights", "📁 Reports"])

with tab1:
    st.subheader("Real-Time Plant Visualization")
    st.plotly_chart(create_3d_plant(plant.state), use_container_width=True, config={'displayModeBar': False})
    if st.session_state.get('emergency'):
        st.markdown('<div class="emergency-alert">🚨 CRITICAL ALERT: REACTOR OVERHEATING - SHUTDOWN IMMEDIATE</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("AI Predictive Analytics")
    prediction = ai_predictor.predict(plant.state)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(create_risk_gauge(prediction["risk_score"]), use_container_width=True)
    with col2:
        st.info(f"""
        **AI Recommendation:** {prediction["recommendation"]}  
        - Anomaly Detected: {prediction['is_anomaly']}  
        - Confidence: {min(95, max(50, 100 - prediction['risk_score'])):.0f}%
        """)
        if st.button("🔄 Optimize System", use_container_width=True):
            with st.spinner("Optimizing parameters with AI..."):
                time.sleep(1.5)
                st.success(f"Optimization complete! Saved {np.random.uniform(5, 15):.1f}% energy")

with tab3:
    st.subheader("System Reports")
    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="InoSense AI System Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}", ln=1)
        pdf.cell(200, 10, txt=f"Risk Score: {plant.state['risk_score']:.1f}%", ln=1)
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(f"InoSense Report - {datetime.now()}")
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white')
        img.save("report_qr.png")
        pdf.image("report_qr.png", x=50, w=100)
        return pdf.output(dest='S').encode('latin1')

    st.download_button(
        label="📥 Download Detailed Report (PDF)",
        data=generate_report(),
        file_name=f"inosense_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )
    st.image("https://via.placeholder.com/800x400?text=Advanced+Trend+Analysis", caption="AI-Powered Trend Insights")

# Real-time update
time.sleep(1)  # Simulate data refresh
st.experimental_rerun()

# Footer
st.divider()
st.caption("""
© 2025 InoSense AI | Built with Streamlit, scikit-learn, and Plotly  
[GitHub](https://github.com/yourusername/inosense) | [Live Demo](your-app.streamlit.app) | [Contact](mailto:your.email@example.com)
""")