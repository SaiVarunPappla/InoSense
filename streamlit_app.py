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
    page_title="InoSense | Smart Monitoring of Chemical Processes",
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
    transition: all 0.3s ease;
    overflow-x: hidden;
}

body.light-mode {
    background-color: #f0f4f8;
    color: #333333;
}

.stMetric {
    border-radius: 10px;
    padding: 15px;
    background-color: #2e3a55;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease;
}
body.light-mode .stMetric {
    background-color: #e0e7f0;
    color: #333333;
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
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
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
body.light-mode .stButton>button {
    background-color: #6ba4e7;
}
body.light-mode .stButton>button:hover {
    background-color: #4682b4;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #2e3a55;
}
body.light-mode .stTabs [data-baseweb="tab-list"] {
    background-color: #d0d9e3;
}
</style>
""", unsafe_allow_html=True)

# Sidebar with controls
with st.sidebar:
    st.header("Control Panel")
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    dark_mode = st.toggle("Light Mode", st.session_state.dark_mode, key="dark_mode_toggle")
    st.session_state.dark_mode = not dark_mode
    if st.session_state.dark_mode:
        st.markdown("<body class='light-mode'>", unsafe_allow_html=True)
    else:
        st.markdown("<body>", unsafe_allow_html=True)

    emergency = st.button("Trigger Emergency", help="Simulate a critical failure", use_container_width=True)
    if emergency and 'emergency_time' not in st.session_state:
        st.session_state.emergency = True
        st.session_state.emergency_time = time.time()
        st.toast("Emergency triggered!", icon="⚠️")

    ai_mode = st.selectbox("AI Model", ["Isolation Forest v2.0", "Predictive Model v3.0"], index=0)
    zoom_level = st.slider("3D Zoom Level", min_value=1.0, max_value=2.5, value=1.5, step=0.1)
    refresh_rate = st.slider("Refresh Rate (seconds)", min_value=1, max_value 5, value=2, step=1)

    st.divider()
    st.markdown(f"""
    **Developed by:** Varun  
    **Version:** 2.5.0  
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M IST')}  
    **Links:** [GitHub](https://github.com/SaiVarunPappla) | [LinkedIn](https://www.linkedin.com/in/pappla-sai-varun-874902200/)  
    **Experience:** 2+ Years in AI Development, Cloud Deployment
    """)

# Main header
st.header("InoSense | Smart Monitoring of Chemical Processes")
st.caption("Real-time analytics and optimization for industrial efficiency")

# Plant simulator with progress tracking
class PlantSimulator:
    def __init__(self):
        self.state = {"temperature": 347.0, "flow_rate": 42.0, "energy_usage": 4.2, "risk_score": 45.0, "pressure": 15.0, "vibration": 0.5}
        self.history = pd.DataFrame(columns=["time", "temperature", "flow_rate", "energy_usage", "risk_score", "pressure", "vibration"])
        self.progress = 0.0

    def update(self):
        t = time.time()
        self.state["temperature"] = 347 + 5 * np.sin(t * 0.1) + np.random.normal(0, 1.5)
        self.state["flow_rate"] = 42 + 2 * np.cos(t * 0.15) + np.random.normal(0, 0.7)
        self.state["energy_usage"] = 4.2 - 0.1 * np.sin(t * 0.2) + np.random.normal(0, 0.05)
        self.state["risk_score"] = min(100, max(0, 45 + 15 * np.random.normal(0, 1)))
        self.state["pressure"] = 15 + 1 * np.sin(t * 0.12) + np.random.normal(0, 0.3)
        self.state["vibration"] = 0.5 + 0.1 * np.cos(t * 0.18) + np.random.normal(0, 0.05)
        self.progress = min(1.0, self.progress + np.random.uniform(0.005, 0.01))
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
        pressure_data = np.random.normal(15, 1, 150)
        training_data = np.column_stack((temp_data, flow_data, energy_data, pressure_data))
        self.model.fit(training_data)

    def predict(self, state):
        features = np.array([state["temperature"], state["flow_rate"], state["energy_usage"], state["pressure"]]).reshape(1, -1)
        anomaly_score = self.model.score_samples(features)[0]
        return {
            "risk_score": abs(anomaly_score) * 100,
            "is_anomaly": anomaly_score < -0.6,
            "recommendation": self._get_recommendation(abs(anomaly_score)),
            "temperature_impact": self._calculate_impact(state["temperature"], 347),
            "flow_impact": self._calculate_impact(state["flow_rate"], 42),
            "pressure_impact": self._calculate_impact(state["pressure"], 15)
        }

    def _get_recommendation(self, anomaly_score):
        if anomaly_score > 1.8:
            return "Critical: Immediate shutdown and expert review required"
        elif anomaly_score > 1.2:
            return "Warning: Adjust settings and monitor closely"
        else:
            return "Optimal: Continue with periodic inspections"

    def _calculate_impact(self, value, baseline):
        return abs(value - baseline) / baseline * 100

ai_predictor = AIPredictor()

# Real-time metrics with dynamic updates
metrics_container = st.empty()
progress_container = st.empty()

def update_display():
    plant.update()
    with metrics_container.container():
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Temperature", f"{plant.state['temperature']:.1f}°K", f"{np.random.uniform(-2, 2):.1f}°C")
        with col2:
            st.metric("Flow Rate", f"{plant.state['flow_rate']:.1f} L/s", f"{np.random.uniform(-1, 1):.1f}%")
        with col3:
            st.metric("Energy Usage", f"{plant.state['energy_usage']:.1f} MW", f"{np.random.uniform(-0.3, 0.3):.1f} MW")
        with col4:
            st.metric("Risk Score", f"{plant.state['risk_score']:.0f}%", delta_color="inverse")
        with col5:
            st.metric("Pressure", f"{plant.state['pressure']:.1f} bar", f"{np.random.uniform(-0.5, 0.5):.1f} bar")
        with col6:
            st.metric("Vibration", f"{plant.state['vibration']:.2f} mm/s", f"{np.random.uniform(-0.05, 0.05):.2f} mm/s")
    with progress_container.container():
        st.progress(plant.progress)

# Enhanced 3D visualization with annotations
def create_3d_plant(state, zoom):
    x, y = np.meshgrid(np.linspace(0, 10, 30), np.linspace(0, 10, 30))
    z = np.sin(x) * np.cos(y) * (state["temperature"] / 500) + np.random.normal(0, 0.1, (30, 30))
    fig = go.Figure(data=[go.Surface(z=z, colorscale='Viridis', opacity=0.8)])
    fig.add_trace(go.Scatter3d(
        x=[5], y=[5], z=[z[15, 15]], mode='markers+text',
        marker=dict(size=10, color='red'),
        text=[f"Temp: {state['temperature']:.1f}°K"],
        textposition="top center",
        hovertemplate="Core Temp: %{z:.1f}°K<extra></extra>"
    ))
    fig.update_layout(
        scene=dict(xaxis_title='X-Axis', yaxis_title='Y-Axis', zaxis_title='Z-Axis',
                   camera=dict(eye=dict(x=zoom, y=zoom, z=zoom))),
        margin=dict(l=0, r=0, b=0, t=30),
        title="3D Plant Model with Annotations"
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

def create_risk_heatmap(prediction):
    fig = go.Figure(data=go.Heatmap(
        z=[[prediction["risk_score"], prediction["temperature_impact"]],
           [prediction["flow_impact"], prediction["pressure_impact"]]],
        x=['Risk Score', 'Temp Impact'],
        y=['Flow Impact', 'Pressure Impact'],
        colorscale='RdYlGn',
        zmin=0, zmax=100,
        hovertemplate="%{x}: %{z:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        title="Risk Distribution Heatmap",
        margin=dict(l=0, r=0, b=0, t=30),
        height=250
    )
    return fig

def create_comparison_chart(history, current_state):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["temperature"].tail(10), mode='lines', name='Historical Temp'))
    fig.add_trace(go.Scatter(y=[current_state["temperature"]] * 10, mode='lines', name='Current Temp', line=dict(dash='dash')))
    fig.update_layout(
        xaxis_title="Last 10 Updates",
        yaxis_title="Temperature (°K)",
        title="Historical vs. Current Temperature",
        template="plotly_dark"
    )
    return fig

# Tab layout
tab1, tab2, tab3 = st.tabs(["Live Monitoring", "AI Insights", "Reports & Trends"])

with tab1:
    st.subheader("Live Plant Monitoring")
    st.plotly_chart(create_3d_plant(plant.state, zoom_level), use_container_width=True)
    if st.session_state.get('emergency', False):
        st.markdown('<div class="emergency-alert">Warning: Critical Failure Detected</div>', unsafe_allow_html=True)
        if 'emergency_time' in st.session_state and time.time() - st.session_state.emergency_time > 10:
            st.session_state.emergency = False
            del st.session_state.emergency_time
            st.success("Emergency resolved automatically.")

with tab2:
    st.subheader("AI-Powered Insights")
    prediction = ai_predictor.predict(plant.state)
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        st.plotly_chart(create_risk_gauge(prediction["risk_score"]), use_container_width=True)
    with col2:
        st.plotly_chart(create_risk_heatmap(prediction), use_container_width=True)
    with col3:
        st.plotly_chart(create_comparison_chart(plant.history, plant.state), use_container_width=True)
        st.info(f"""
        **Action Plan:** {prediction["recommendation"]}  
        - Anomaly Status: {prediction['is_anomaly']}  
        - Confidence Level: {min(95, max(50, 100 - prediction['risk_score'])):.0f}%  
        - Impact Analysis:  
          - Temperature Deviation: {prediction['temperature_impact']:.1f}%  
          - Flow Rate Impact: {prediction['flow_impact']:.1f}%  
          - Pressure Variation: {prediction['pressure_impact']:.1f}%  
        **Predictive Suggestions:**  
        - Optimize temperature if deviation exceeds 5%.  
        - Adjust flow rate for stability if impact > 10%.  
        - Monitor pressure trends for potential leaks.  
        """)
        if st.button("Optimize System", use_container_width=True):
            with st.spinner("Executing optimization protocol..."):
                time.sleep(1.5)
                st.success(f"Optimization complete! Achieved {np.random.uniform(10, 20):.1f}% efficiency gain.")

with tab3:
    st.subheader("Reports & Trends")
    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="InoSense AI Operational Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}", ln=1)
        pdf.cell(200, 10, txt=f"Temperature: {plant.state['temperature']:.1f}°K", ln=1)
        pdf.cell(200, 10, txt=f"Flow Rate: {plant.state['flow_rate']:.1f} L/s", ln=1)
        pdf.cell(200, 10, txt=f"Energy Usage: {plant.state['energy_usage']:.1f} MW", ln=1)
        pdf.cell(200, 10, txt=f"Risk Score: {plant.state['risk_score']:.1f}%", ln=1)
        pdf.cell(200, 10, txt=f"Pressure: {plant.state['pressure']:.1f} bar", ln=1)
        pdf.cell(200, 10, txt=f"Vibration: {plant.state['vibration']:.2f} mm/s", ln=1)
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(f"InoSense Report - {datetime.now()}")
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white')
        img.save("report_qr.png")
        pdf.image("report_qr.png", x=50, w=100)
        return pdf.output(dest='S').encode('latin1')

    st.download_button(
        label="Download Detailed Report (PDF)",
        data=generate_report(),
        file_name=f"inosense_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf"
    )

    def export_data():
        csv = plant.history.to_csv(index=False)
        return csv.encode()

    st.download_button(
        label="Export Historical Data (CSV)",
        data=export_data(),
        file_name=f"inosense_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

    st.subheader("Historical Trends")
    window_size = 5
    smoothed_temp = plant.history["temperature"].rolling(window=window_size, min_periods=1).mean()
    smoothed_flow = plant.history["flow_rate"].rolling(window=window_size, min_periods=1).mean()
    smoothed_risk = plant.history["risk_score"].rolling(window=window_size, min_periods=1).mean()
    smoothed_pressure = plant.history["pressure"].rolling(window=window_size, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=smoothed_temp, mode='lines', name='Temperature (°K)', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(y=smoothed_flow, mode='lines', name='Flow Rate (L/s)', line=dict(color='#ff7f0e', width=2)))
    fig.add_trace(go.Scatter(y=smoothed_risk, mode='lines', name='Risk Score (%)', line=dict(color='#2ca02c', width=2)))
    fig.add_trace(go.Scatter(y=smoothed_pressure, mode='lines', name='Pressure (bar)', line=dict(color='#d62728', width=2)))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        title="Comprehensive Trend Analysis",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Real-time update with rerun
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

if time.time() - st.session_state.last_update > refresh_rate:
    update_display()
    st.session_state.last_update = time.time()
    st.experimental_rerun()

# Reset emergency on page load if timed out
if 'emergency' in st.session_state and 'emergency_time' in st.session_state:
    if time.time() - st.session_state.emergency_time > 10:
        st.session_state.emergency = False
        del st.session_state.emergency_time

# Footer
st.divider()
st.markdown("<h4 style='text-align: center; color: #ffffff;'>© 2025 InoSense AI | Pioneering Industrial Intelligence</h4>", unsafe_allow_html=True)
st.caption("""
Developed by Varun | Technologies: Python, Streamlit, Plotly, scikit-learn  
[GitHub](https://github.com/SaiVarunPappla) | [LinkedIn](https://www.linkedin.com/in/pappla-sai-varun-874902200/)  
""")