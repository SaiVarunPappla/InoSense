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
st.set_page_config(
    page_title="InoSense AI | Chemical Plant Monitor",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

/* Main styling */
html, body, [class*="css"] {
    font-family: 'Roboto', sans-serif;
    transition: all 0.3s ease;
}

/* Animated metrics */
.stMetric {
    border-radius: 10px;
    padding: 15px;
    background: rgba(255, 255, 255, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.stMetric:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}

/* Pulse animation for alerts */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}
.emergency-alert {
    animation: pulse 1.5s infinite;
}

/* 3D container effect */
.plot-container {
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    overflow: hidden;
    transition: all 0.3s ease;
}
.plot-container:hover {
    box-shadow: 0 15px 35px rgba(0,0,0,0.4);
}

/* Dark/light mode toggle */
.st-bb { display: flex; justify-content: flex-end; }
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.title("⚙️ Control Panel")
    
    # Dark/light mode toggle
    dark_mode = st.toggle("🌙 Dark Mode", True)
    
    # Emergency controls
    if st.button("🚨 Trigger Emergency Protocol", 
                help="Simulate a critical failure scenario",
                use_container_width=True):
        st.session_state.emergency = True
        st.toast("Emergency protocol activated!", icon="⚠️")
    
    # AI model selection
    st.selectbox("🤖 AI Model Version", 
                ["LSTM v4.2", "Transformer v2.1", "Isolation Forest"],
                index=0)
    
    # Voice command
    if st.button("🎤 Voice Command", 
                use_container_width=True):
        with st.spinner("Listening..."):
            time.sleep(2)
            st.success("Voice command received: 'Show temperature trends'")
    
    # Developer info
    st.divider()
    st.markdown("""
    **Developer:** Varun 
    **Version:** 1.0.0  
    **Last Updated:** """ + datetime.now().strftime("%Y-%m-%d"))
st.title("🏭 InoSense AI: Chemical Plant Monitoring")
st.caption("Real-time AI-powered industrial process optimization")

# Animated metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🌡️ Reactor Temp", "347°K", "2°C ▲", 
              help="Current reactor core temperature")
with col2:
    st.metric("🔄 Flow Rate", "42 L/s", "1.2% ▼")
with col3:
    st.metric("⚡ Energy Usage", "4.2 MW", "0.3 MW ▼")
with col4:
    st.metric("⚠️ Risk Level", "Medium", "15% ▲", 
              delta_color="inverse")

# Risk gauge with Plotly
def create_risk_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "System Risk Assessment"},
        gauge={
            'axis': {'range': [None, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value}
        }
    ))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    return fig

def create_3d_plant():
    # Generate sample 3D data
    x, y, z = np.random.rand(3, 100) * 10
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=8,
            color=z,
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.7))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Live 3D Plant Visualization"
    )
    return fig

# Tab layout
tab1, tab2, tab3 = st.tabs(["📊 Live Monitoring", "🤖 AI Insights", "📁 Reports"])

with tab1:
    st.subheader("Real-Time Plant Visualization")
    st.plotly_chart(create_3d_plant(), use_container_width=True, 
                   config={'displayModeBar': False})
    
    # Animated emergency alert
    if st.session_state.get('emergency'):
        st.error("""
        🚨 CRITICAL ALERT: REACTOR OVERHEATING DETECTED  
        **Recommended Action:** Immediate shutdown required
        """, icon="⚠️")

with tab2:
    st.subheader("AI Predictive Analytics")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.plotly_chart(create_risk_gauge(65), use_container_width=True)
    
    with col2:
        st.info("""
        **AI Recommendations:**  
        - Reduce reactor temperature by 5%  
        - Schedule pump maintenance  
        - Check pipeline integrity
        """)
        
        if st.button("🔄 Optimize Parameters"):
            with st.spinner("AI computing optimal settings..."):
                time.sleep(2)
                st.success("Optimization complete! Estimated 12% energy savings")

with tab3:
    st.subheader("System Reports")
    
    # Generate PDF report
    def generate_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="InoSense System Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Generated on: {datetime.now()}", ln=1, align='C')
        return pdf.output(dest='S').encode('latin1')
    
    st.download_button(
        label="📥 Download Full Report (PDF)",
        data=generate_report(),
        file_name="inosense_report.pdf",
        mime="application/pdf"
    )
    
    st.image("https://via.placeholder.com/800x400?text=Historical+Trends+Analysis", 
             caption="Historical Performance Trends")
st.divider()
st.caption("""
© 2024 InoSense AI | Developed with Python, Streamlit, and TensorFlow  
[GitHub Repo] | [Documentation] | [Contact Support]
""")