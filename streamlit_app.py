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
    page_title="InoSense AI | Industrial Process Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Arial', 'Helvetica', sans-serif;
    background-color: #1e2a44;
    color: #ffffff;
    overflow-x: hidden;
}

.stMetric {
    border-radius: 10px;
    padding: 15px;
    background-color: #2e3a55;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease;
}
.stMetric:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
}

.emergency-alert {
    background-color: #ff4444;
    color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    font-weight: bold;
}

.plot-container {
    border-radius: 10px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    overflow: hidden;
    transition: transform 0.3s ease;
}
.plot-container:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
}

.stButton>button {
    background-color: #4a90e2;
    color: #ffffff;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 1.1em;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #357abd;
}
</style>
""", unsafe_allow_html=True)

# Sidebar with controls
with st.sidebar:
    st.header("Control Panel")
    dark_mode = st.toggle("Light Mode", False)
    if dark_mode:
        st.markdown("<style>body {background-color: #f0f4f8; color: #333333;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>body {background-color: #1e2a44; color: #ffffff;}</style>", unsafe_allow_html=True)

    emergency = st.button("Trigger Emergency", help="Simulate a critical failure", use_container_width=True)
    if emergency:
        st.session_state.emergency = True
        st.toast("Emergency triggered!", icon="⚠️")

    ai_mode = st.selectbox("AI Model", ["Isolation Forest v2.0", "Predictive Model v3.0"], index=0)
    zoom_level = st.slider("3D Zoom Level", min_value=1.0, max_value=2.0, value=1.5, step=0.1)

    st.divider()
    st.markdown(f"""
    **Developed by:** Varun  
    **Version:** 2.2.0  
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M IST')}  
    **Links:** [GitHub](https://github.com/SaiVarunPappla) | [LinkedIn](https://www.linkedin.com/in/pappla-sai-varun-874902200/)  
    **Credentials:** AWS Certified, AI/ML Training
    """)

# Main header
st.header("InoSense AI | Industrial Process Monitor")
st.caption("Real-time monitoring and optimization for chemical plants")

# Plant simulator with data history
class PlantSimulator:
    def __init__(self):
        self.state = {"temperature": 347.0, "flow_rate": 42.0, "energy_usage": 4.2, "risk_score": 45.0}
        self.history = pd.DataFrame(columns=["time", "temperature", "flow_rate", "energy_usage", "risk_score"])

    def update(self):
        t = time.time()
        self.state["temperature"] = 347 + 5 * np.sin(t * 0.1) + np.random.normal(0, 1.5)
        self.state["flow_rate"] = 42 + 2 * np.cos(t * 0.15) + np.random.normal(0, 0.7)
        self.state["energy_usage"] = 4.2 - 0.1 * np.sin(t * 0.2) + np.random.normal(0, 0.05)
        self.state["risk_score"] = min(100, max(0, 45 + 15 * np.random.normal(0, 1)))
        self.history.loc[len(self.history)] = [datetime.now(), *self.state.values()]
        return self.state

plant = PlantSimulator()

# AI predictor
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
            return "Critical: Immediate shutdown required"
        elif anomaly_score > 1.2:
            return "Warning: Schedule maintenance"
        else:
            return "Normal: Continue operations"

ai_predictor = AIPredictor()

# Real-time metrics
metrics_container = st.empty()
def update_metrics():
    with metrics_container.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Reactor Temperature", f"{plant.state['temperature']:.1f}°K", f"{np.random.uniform(-2, 2):.1f}°C")
        with col2:
            st.metric("Flow Rate", f"{plant.state['flow_rate']:.1f} L/s", f"{np.random.uniform(-1, 1):.1f}%")
        with col3:
            st.metric("Energy Usage", f"{plant.state['energy_usage']:.1f} MW", f"{np.random.uniform(-0.3, 0.3):.1f} MW")
        with col4:
            st.metric("Risk Level", f"{plant.state['risk_score']:.0f}%", delta_color="inverse")

# Enhanced 3D visualization
def create_3d_plant(state, zoom):
    x, y = np.meshgrid(np.linspace(0, 10, 30), np.linspace(0, 10, 30))
    z = np.sin(x) * np.cos(y) * (state["temperature"] / 500) + np.random.normal(0, 0.1, (30, 30))
    fig = go.Figure(data=[go.Surface(z=z, colorscale='Viridis', opacity=0.8)])
    fig.add_trace(go.Scatter3d(
        x=[5], y=[5], z=[z[15, 15]], mode='markers',
        marker=dict(size=10, color='red'),
        hovertemplate="Core Temp: %{z:.1f}°K<extra></extra>"
    ))
    fig.update_layout(
        scene=dict(xaxis_title='X-Axis', yaxis_title='Y-Axis', zaxis_title='Z-Axis',
                   camera=dict(eye=dict(x=zoom, y=zoom, z=zoom))),
        margin=dict(l=0, r=0, b=0, t=30),
        title="3D Plant Model"
    )
    return fig

def create_risk_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Assessment"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.75, 'value': value}
        }
    ))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), height=250)
    return fig

# Tab layout
tab1, tab2, tab3 = st.tabs(["Live Monitoring", "AI Insights", "Reports"])

with tab1:
    st.subheader("Live Plant Monitoring")
    st.plotly_chart(create_3d_plant(plant.state, zoom_level), use_container_width=True)
    if st.session_state.get('emergency'):
        st.markdown('<div class="emergency-alert">Warning: Critical Failure Detected</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("AI-Powered Insights")
    prediction = ai_predictor.predict(plant.state)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(create_risk_gauge(prediction["risk_score"]), use_container_width=True)
    with col2:
        st.info(f"""
        **Recommendation:** {prediction["recommendation"]}  
        - Anomaly Detected: {prediction['is_anomaly']}  
        - Confidence Level: {min(95, max(50, 100 - prediction['risk_score'])):.0f}%
        """)
        if st.button("Optimize Parameters", use_container_width=True):
            with st.spinner("Optimizing system..."):
                time.sleep(1.5)
                st.success(f"Optimization complete! Saved {np.random.uniform(10, 20):.1f}% energy")

with tab3:
    st.subheader("Operational Reports")
    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="InoSense AI Operational Report", ln=1, align='C')
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
        label="Download Report (PDF)",
        data=generate_report(),
        file_name=f"inosense_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )

    # Add trend visualization
    st.subheader("Historical Trends")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=plant.history["temperature"], mode='lines', name='Temperature'))
    fig.add_trace(go.Scatter(y=plant.history["risk_score"], mode='lines', name='Risk Score'))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        title="Trend Analysis",
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# Real-time streaming with fixed progress
def stream_data():
    while True:
        plant.update()
        update_metrics()
        st.progress(plant.progress / 100)  # Normalize progress to 0.0-1.0
        time.sleep(2)  # Update every 2 seconds

# Start streaming
if __name__ == "__main__":
    stream_data()

# Footer
st.divider()
st.markdown("<h4 style='text-align: center; color: #ffffff;'>© 2025 InoSense AI | Advanced Industrial Solutions</h4>", unsafe_allow_html=True)
st.caption("""
Developed by Varun | Technologies: Python, Streamlit, Plotly, scikit-learn  
[GitHub](https://github.com/SaiVarunPappla) | [LinkedIn](https://www.linkedin.com/in/pappla-sai-varun-874902200/)  
""")