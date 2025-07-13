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
    border-radius: 5px;
    padding: 8px;
    background-color: #2e3a55;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease;
}
body.light-mode .stMetric {
    background-color: #e0e7f0;
    color: #333333;
}
.stMetric:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
}

.emergency-alert {
    background-color: #ff4444;
    color: #ffffff;
    padding: 8px;
    border-radius: 3px;
    text-align: center;
    font-weight: 600;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.01); }
    100% { transform: scale(1); }
}

.plot-container {
    border-radius: 5px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    transition: transform 0.2s ease;
}
.plot-container:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}

.stButton>button {
    background-color: #4a90e2;
    color: #ffffff;
    border: none;
    padding: 8px 16px;
    border-radius: 3px;
    font-size: 0.95em;
    transition: background-color 0.2s ease;
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
    border-radius: 3px;
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
    refresh_rate = st.slider("Refresh Rate (seconds)", min_value=1, max_value=5, value=2, step=1)

    st.divider()
    st.markdown(f"""
    **Developed by:** Varun  
    **Version:** 2.8.0  
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M IST')}  
    **Links:** [GitHub](https://github.com/SaiVarunPappla) | [LinkedIn](https://www.linkedin.com/in/pappla-sai-varun-874902200/)  
    **Experience:** 2+ Years in AI/ML, Certified Cloud Architect, Published Researcher
    """)

# Main header
st.header("InoSense | Smart Monitoring of Chemical Processes")
st.caption("Advanced real-time analytics and optimization for industrial efficiency")

# Plant simulator with dynamic parameters and logs
class PlantSimulator:
    def __init__(self):
        self.state = {
            "temperature": 347.0, "flow_rate": 42.0, "energy_usage": 4.2,
            "risk_score": 45.0, "pressure": 15.0, "vibration": 0.5,
            "efficiency": 85.0
        }
        self.history = pd.DataFrame(columns=[
            "time", "temperature", "flow_rate", "energy_usage", "risk_score",
            "pressure", "vibration", "efficiency"
        ])
        self.progress = 0.0
        self.logs = []

    def update(self):
        t = time.time()
        self.state["temperature"] = 347 + 5 * np.sin(t * 0.1) + np.random.normal(0, 1.5)
        self.state["flow_rate"] = 42 + 2 * np.cos(t * 0.15) + np.random.normal(0, 0.7)
        self.state["energy_usage"] = 4.2 - 0.1 * np.sin(t * 0.2) + np.random.normal(0, 0.05)
        self.state["risk_score"] = min(100, max(0, 45 + 15 * np.random.normal(0, 1)))
        self.state["pressure"] = 15 + 1 * np.sin(t * 0.12) + np.random.normal(0, 0.3)
        self.state["vibration"] = 0.5 + 0.1 * np.cos(t * 0.18) + np.random.normal(0, 0.05)
        self.state["efficiency"] = max(50, min(95, 85 + 5 * np.cos(t * 0.1) + np.random.normal(0, 2)))
        self.progress = min(1.0, self.progress + np.random.uniform(0.005, 0.01))
        self.history.loc[len(self.history)] = [datetime.now(), *self.state.values()]
        self.logs.append(f"{datetime.now().strftime('%H:%M:%S')}: Updated - Temp: {self.state['temperature']:.1f}°K, Risk: {self.state['risk_score']:.0f}%")
        return self.state

plant = PlantSimulator()

# AI predictor with predictive alerts
class AIPredictor:
    def __init__(self):
        self.model = IsolationForest(n_estimators=200, contamination=0.1)
        self._fit_initial_model()

    def _fit_initial_model(self):
        np.random.seed(42)
        temp_data = np.random.normal(347, 15, 200)
        flow_data = np.random.normal(42, 3, 200)
        energy_data = np.random.normal(4.2, 0.2, 200)
        pressure_data = np.random.normal(15, 1, 200)
        training_data = np.column_stack((temp_data, flow_data, energy_data, pressure_data))
        self.model.fit(training_data)

    def predict(self, state):
        features = np.array([state["temperature"], state["flow_rate"], state["energy_usage"], state["pressure"]]).reshape(1, -1)
        anomaly_score = self.model.score_samples(features)[0]
        risk_score = abs(anomaly_score) * 100
        return {
            "risk_score": risk_score,
            "is_anomaly": anomaly_score < -0.6,
            "recommendation": self._get_recommendation(risk_score),
            "impacts": {
                "temperature": self._calculate_impact(state["temperature"], 347),
                "flow_rate": self._calculate_impact(state["flow_rate"], 42),
                "pressure": self._calculate_impact(state["pressure"], 15),
                "efficiency": self._calculate_impact(100 - state["efficiency"], 15)
            },
            "predictive_alert": self._generate_predictive_alert(risk_score, state)
        }

    def _get_recommendation(self, risk_score):
        if risk_score > 1.8:
            return "Critical: Immediate shutdown and inspection required"
        elif risk_score > 1.2:
            return "Warning: Adjust parameters and schedule maintenance"
        else:
            return "Optimal: Maintain current operations"

    def _calculate_impact(self, value, baseline):
        return abs(value - baseline) / baseline * 100

    def _generate_predictive_alert(self, risk_score, state):
        if risk_score > 70 and state["temperature"] > 355:
            return "Predictive Alert: Potential overheating in 2-3 cycles. Initiate cooling."
        elif risk_score > 60 and state["pressure"] > 16:
            return "Predictive Alert: Pressure buildup detected. Check seals."
        return "No predictive alerts at this time."

ai_predictor = AIPredictor()

# Real-time metrics with dynamic updates
metrics_container = st.empty()
progress_container = st.empty()

def update_display():
    plant.update()
    with metrics_container.container():
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            st.metric("Temperature", f"{plant.state['temperature']:.1f}°K", f"{np.random.uniform(-1, 1):.1f}°C")
        with col2:
            st.metric("Flow Rate", f"{plant.state['flow_rate']:.1f} L/s", f"{np.random.uniform(-0.5, 0.5):.1f}%")
        with col3:
            st.metric("Energy Usage", f"{plant.state['energy_usage']:.1f} MW", f"{np.random.uniform(-0.2, 0.2):.1f} MW")
        with col4:
            st.metric("Risk Score", f"{plant.state['risk_score']:.0f}%", delta_color="inverse")
        with col5:
            st.metric("Pressure", f"{plant.state['pressure']:.1f} bar", f"{np.random.uniform(-0.3, 0.3):.1f} bar")
        with col6:
            st.metric("Vibration", f"{plant.state['vibration']:.2f} mm/s", f"{np.random.uniform(-0.03, 0.03):.2f} mm/s")
        with col7:
            st.metric("Efficiency", f"{plant.state['efficiency']:.1f}%", f"{np.random.uniform(-1, 1):.1f}%")
    with progress_container.container():
        st.progress(plant.progress)

# Professional 3D visualization
def create_3d_plant(state, zoom):
    x, y = np.meshgrid(np.linspace(0, 10, 30), np.linspace(0, 10, 30))
    z = np.sin(x) * np.cos(y) * (state["temperature"] / 500)
    fig = go.Figure(data=[go.Surface(z=z, colorscale='Blues', opacity=0.7, showscale=False)])
    fig.add_trace(go.Scatter3d(
        x=[5], y=[5], z=[z[15, 15]], mode='markers+text',
        marker=dict(size=8, color='red'),
        text=[f"Temp: {state['temperature']:.1f}°K"],
        textposition="top center",
        hovertemplate="Core Temp: %{z:.1f}°K<extra></extra>"
    ))
    fig.update_layout(
        scene=dict(xaxis_title='X-Axis', yaxis_title='Y-Axis', zaxis_title='Z-Axis',
                   camera=dict(eye=dict(x=zoom, y=zoom, z=zoom)), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), zaxis=dict(showgrid=False)),
        margin=dict(l=20, r=20, b=20, t=40),
        title="3D Plant Thermal Profile",
        template="plotly_white"
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
    fig.update_layout(margin=dict(l=20, r=20, b=20, t=40), height=250, template="plotly_white")
    return fig

def create_risk_heatmap(prediction):
    fig = go.Figure(data=go.Heatmap(
        z=[[prediction["risk_score"], prediction["impacts"]["temperature"]],
           [prediction["impacts"]["flow_rate"], prediction["impacts"]["pressure"]]],
        x=['Risk Score', 'Temp Impact'],
        y=['Flow Impact', 'Pressure Impact'],
        colorscale='RdYlGn',
        zmin=0, zmax=100,
        hovertemplate="%{x}: %{z:.1f}%<extra></extra>"
    ))
    fig.update_layout(
        title="Risk Factor Distribution",
        margin=dict(l=20, r=20, b=20, t=40),
        height=250,
        template="plotly_white"
    )
    return fig

def create_multi_trend_chart(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["temperature"].rolling(5).mean(), mode='lines', name='Temperature (°K)', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(y=history["flow_rate"].rolling(5).mean(), mode='lines', name='Flow Rate (L/s)', line=dict(color='#ff7f0e')))
    fig.add_trace(go.Scatter(y=history["risk_score"].rolling(5).mean(), mode='lines', name='Risk Score (%)', line=dict(color='#2ca02c')))
    fig.add_trace(go.Scatter(y=history["pressure"].rolling(5).mean(), mode='lines', name='Pressure (bar)', line=dict(color='#d62728')))
    fig.add_trace(go.Scatter(y=history["efficiency"].rolling(5).mean(), mode='lines', name='Efficiency (%)', line=dict(color='#9467bd')))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        title="Multi-Parameter Trend Analysis",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        height=400
    )
    return fig

# Tab layout with detailed content
tabs = st.tabs(["Live Monitoring", "AI Insights", "Reports & Trends"])

with tabs[0]:  # Live Monitoring
    st.subheader("Live Monitoring Dashboard")
    st.plotly_chart(create_3d_plant(plant.state, zoom_level), use_container_width=True, key="live_3d_chart")
    if st.session_state.get('emergency', False):
        st.markdown('<div class="emergency-alert">Warning: Critical Failure Detected</div>', unsafe_allow_html=True)
        if 'emergency_time' in st.session_state and time.time() - st.session_state.emergency_time > 10:
            st.session_state.emergency = False
            del st.session_state.emergency_time
            st.success("Emergency resolved automatically.")
    update_display()
    st.write("""
    **Live Monitoring Overview:**  
    This dashboard provides a real-time 3D thermal profile of the chemical plant, updated every few seconds based on sensor data. The model highlights the core temperature with a red marker, and metrics below reflect dynamic changes in plant conditions.  
    - **Critical Thresholds:** Temperature (>360°K), Pressure (14-16 bar), Risk Score (>70%).  
    - **Best Practices:** Use the zoom slider to inspect thermal gradients; monitor metrics for immediate action.
    """)

with tabs[1]:  # AI Insights
    st.subheader("AI-Driven Process Insights")
    prediction = ai_predictor.predict(plant.state)
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        st.plotly_chart(create_risk_gauge(prediction["risk_score"]), use_container_width=True, key="risk_gauge")
    with col2:
        st.plotly_chart(create_risk_heatmap(prediction), use_container_width=True, key="risk_heatmap")
    with col3:
        st.plotly_chart(create_multi_trend_chart(plant.history.tail(20)), use_container_width=True, key="trend_chart")
    st.info(f"""
    **Action Plan:** {prediction["recommendation"]}  
    - Anomaly Status: {prediction['is_anomaly']}  
    - Confidence Level: {min(95, max(50, 100 - prediction['risk_score'])):.0f}%  
    - **Impact Analysis:**  
      - Temperature Deviation: {prediction['impacts']['temperature']:.1f}% (Threshold: 5%)  
      - Flow Rate Variation: {prediction['impacts']['flow_rate']:.1f}% (Threshold: 10%)  
      - Pressure Deviation: {prediction['impacts']['pressure']:.1f}% (Threshold: 5%)  
      - Efficiency Impact: {prediction['impacts']['efficiency']:.1f}% (Target: <15%)  
    - **Predictive Alert:** {prediction['predictive_alert']}  
    **Detailed Recommendations:**  
    - Temperature: Activate cooling if >360°K.  
    - Flow Rate: Adjust pumps if variation >10%.  
    - Pressure: Inspect seals if >16 bar.  
    - Efficiency: Schedule maintenance if <80%.  
    """)
    if st.button("Execute Optimization", use_container_width=True, key="optimize_button"):
        with st.spinner("Running optimization algorithm..."):
            time.sleep(1.5)
            st.success(f"Optimization successful! Efficiency improved by {np.random.uniform(10, 20):.1f}%.")

with tabs[2]:  # Reports & Trends
    st.subheader("Reports & Comprehensive Trends")
    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="InoSense AI Operational Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M IST')}", ln=1)
        for key, value in plant.state.items():
            pdf.cell(200, 10, txt=f"{key.replace('_', ' ').title()}: {value:.1f} {'' if key == 'risk_score' else key.split('_')[-1]}", ln=1)
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
        mime="application/pdf",
        key="pdf_download"
    )

    def export_data():
        csv = plant.history.to_csv(index=False)
        return csv.encode()

    st.download_button(
        label="Export Historical Data (CSV)",
        data=export_data(),
        file_name=f"inosense_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        key="csv_download"
    )

    st.subheader("Comprehensive Trend Analysis")
    fig = create_multi_trend_chart(plant.history)
    st.plotly_chart(fig, use_container_width=True, key="trend_plot")
    st.write("""
    **Trend Analysis Overview:**  
    This section provides smoothed trends for all key parameters, enabling long-term performance evaluation.  
    - **Temperature (°K):** Stable range 340-355°K; spikes indicate overheating.  
    - **Flow Rate (L/s):** Optimal 40-44 L/s; drops suggest blockages.  
    - **Risk Score (%):** Action required >70%; monitor trends for escalation.  
    - **Pressure (bar):** Target 14-16 bar; deviations signal leaks.  
    - **Efficiency (%):** Maintain >80%; declines suggest wear.  
    **Advanced Usage:** Export data for regression analysis or machine learning to forecast maintenance needs.
    """)

    st.subheader("System Logs")
    st.text_area("Recent Updates", "\n".join(plant.logs[-10:]), height=150, disabled=True)

# Real-time update with session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

if time.time() - st.session_state.last_update > refresh_rate:
    update_display()
    st.session_state.last_update = time.time()

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