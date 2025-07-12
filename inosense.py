"""
INOSENSE - INDUSTRIAL GRADE MONITORING SYSTEM
A Full-Stack AI/ML Solution with:
- Real-time 3D Digital Twin
- Predictive Maintenance AI
- Anomaly Detection Engine
- Voice Control Interface
- Blockchain Data Integrity
- Multi-User Dashboard
- Automated Reporting
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from fastapi import FastAPI, WebSocket, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import speech_recognition as sr
from dash.exceptions import PreventUpdate
import time
import threading
import asyncio
import websockets
import json
from datetime import datetime, timedelta
import sounddevice as sd
import speech_recognition as sr
import threading
import hashlib
from typing import Dict, List, Optional, Tuple
from collections import deque
import warnings
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from fpdf import FPDF
import qrcode
warnings.filterwarnings("ignore")

# ======================
# 1. SYSTEM CONFIGURATION
# ======================
class SystemConfig:
    def __init__(self):
        self.params = {
            "sensors": {
                "temperature": {"min": 300, "max": 500, "unit": "°K"},
                "pressure": {"min": 100, "max": 300, "unit": "kPa"},
                "flow": {"min": 20, "max": 60, "unit": "L/s"},
                "vibration": {"min": 0, "max": 2, "unit": "mm/s"},
                "ph": {"min": 6, "max": 8, "unit": "pH"},
                "oxygen": {"min": 15, "max": 25, "unit": "%"}
            },
            "ai": {
                "model_retrain_interval": 3600,
                "anomaly_threshold": 0.95,
                "prediction_horizon": 24  # hours
            },
            "security": {
                "blockchain": True,
                "data_encryption": True,
                "access_control": True
            }
        }

# ======================
# 2. BLOCKCHAIN DATA INTEGRITY
# ======================
class BlockchainLedger:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.create_genesis_block()
        
    def create_genesis_block(self):
        genesis = {
            "index": 0,
            "timestamp": str(datetime.utcnow()),
            "data": "GENESIS BLOCK",
            "previous_hash": "0"*64,
            "nonce": 0
        }
        genesis["hash"] = self.hash_block(genesis)
        self.chain.append(genesis)
    
    def hash_block(self, block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    def add_block(self, data):
        last_block = self.chain[-1]
        
        new_block = {
            "index": len(self.chain),
            "timestamp": str(datetime.utcnow()),
            "data": data,
            "previous_hash": last_block["hash"],
            "nonce": 0
        }
        
        new_block["hash"] = self.hash_block(new_block)
        self.chain.append(new_block)
        return new_block

# ======================
# 3. 4D DIGITAL TWIN ENGINE
# ======================
class ChemicalPlantTwin:
    def __init__(self, config):
        self.config = config
        self.state = {
            "core_metrics": {
                "reactor_temp": 350.0,
                "pipeline_pressure": 150.0,
                "flow_rate": 40.0,
                "vibration": 0.5
            },
            "chemical_properties": {
                "ph": 7.0,
                "oxygen_content": 21.0,
                "conductivity": 0.1
            },
            "equipment_health": {
                "pump_efficiency": 0.92,
                "valve_status": "open",
                "heat_exchanger": 0.85,
                "corrosion_rate": 0.02
            },
            "safety_status": {
                "overall": "normal",
                "last_incident": None,
                "risk_score": 0.15
            }
        }
        self.history = deque(maxlen=10000)
        self.blockchain = BlockchainLedger()
        self.equipment_degradation_rate = 0.0001
        
    def update_state(self):
        """Advanced multi-physics simulation with equipment degradation"""
        t = time.time()
        
        # Core process dynamics with simulated disturbances
        self.state["core_metrics"]["reactor_temp"] = (
            350 + 25*np.sin(t*0.1) + 3*np.random.normal()
        )
        self.state["core_metrics"]["pipeline_pressure"] = (
            150 + 20*np.cos(t*0.15) * (1 + 0.1*self.state["equipment_health"]["corrosion_rate"])
        )
        self.state["core_metrics"]["flow_rate"] = (
            40 + 10*np.sin(t*0.2) * self.state["equipment_health"]["pump_efficiency"]
        )
        self.state["core_metrics"]["vibration"] = (
            0.5 + 0.2*np.cos(t*0.25) * (1 + self.state["equipment_health"]["corrosion_rate"])
        )
        
        # Chemical properties with simulated reactions
        self.state["chemical_properties"]["ph"] = (
            7.0 + 0.3*np.sin(t*0.3) + 0.05*self.state["core_metrics"]["reactor_temp"]/350
        )
        self.state["chemical_properties"]["oxygen_content"] = (
            21.0 - 0.5*np.sin(t*0.4) * self.state["core_metrics"]["flow_rate"]/40
        )
        
        # Equipment degradation over time
        self.state["equipment_health"]["pump_efficiency"] *= (1 - self.equipment_degradation_rate)
        self.state["equipment_health"]["heat_exchanger"] *= (1 - self.equipment_degradation_rate)
        self.state["equipment_health"]["corrosion_rate"] += 0.00001
        
        # Update safety metrics
        self._update_safety_status()
        
        # Log to blockchain
        self.blockchain.add_block(self.state.copy())
        self.history.append(self.state.copy())
        
        return self.state
    
    def _update_safety_status(self):
        """Calculate comprehensive risk score"""
        risk_factors = {
            "temperature": min(1.0, max(0, (self.state["core_metrics"]["reactor_temp"] - 350) / 75)),
            "pressure": min(1.0, max(0, (self.state["core_metrics"]["pipeline_pressure"] - 150) / 100)),
            "vibration": min(1.0, self.state["core_metrics"]["vibration"]),
            "corrosion": min(1.0, self.state["equipment_health"]["corrosion_rate"] * 50),
            "pump_efficiency": 1 - self.state["equipment_health"]["pump_efficiency"]
        }
        
        weights = {
            "temperature": 0.3,
            "pressure": 0.3,
            "vibration": 0.2,
            "corrosion": 0.1,
            "pump_efficiency": 0.1
        }
        
        total_risk = sum(risk_factors[k] * weights[k] for k in risk_factors)
        self.state["safety_status"]["risk_score"] = total_risk
        
        if total_risk > 0.7:
            self.state["safety_status"]["overall"] = "critical"
            if not self.state["safety_status"]["last_incident"]:
                self.state["safety_status"]["last_incident"] = str(datetime.utcnow())
        elif total_risk > 0.4:
            self.state["safety_status"]["overall"] = "warning"
        else:
            self.state["safety_status"]["overall"] = "normal"

# ======================
# 4. AI PREDICTIVE ENGINE
# ======================
class AIPredictiveSystem:
    def __init__(self):
        self.models = {
            "failure_prediction": self._build_transformer_model(),
            "anomaly_detection": IsolationForest(n_estimators=200, contamination=0.05),
            "process_optimization": self._build_optimization_model()
        }
        self.scaler = StandardScaler()
        self.training_data = deque(maxlen=20000)
        self.last_trained = datetime.utcnow()
        
def _build_transformer_model(self):
    """Transformer-based predictive model"""
    inputs = Input((30, 8))  # Fixed line - removed 'shape='
    x = LSTM(128, return_sequences=True)(inputs)
    x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
    
    def _build_optimization_model(self):
        """Multi-objective optimization model"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(8,)),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(3, activation='linear')  # Output: [temp_optim, pressure_optim, flow_optim]
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def predict_failure(self, plant_state):
        """Predict equipment failure risk with explanations"""
        features = self._extract_features(plant_state)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Get predictions
        risk = self.models["failure_prediction"].predict(features_scaled.reshape(1, 1, 8))[0][0]
        is_anomaly = self.models["anomaly_detection"].predict(features_scaled)[0] == -1
        
        # Generate recommendations
        recommendation, actions = self._generate_recommendation(risk, plant_state)
        
        return {
            "risk_score": float(risk * 100),
            "is_anomaly": bool(is_anomaly),
            "confidence": float(0.9 - (risk * 0.4)),  # Higher confidence at extremes
            "recommendation": recommendation,
            "actions": actions,
            "timestamp": str(datetime.utcnow())
        }
    
    def _extract_features(self, state):
        return [
            state["core_metrics"]["reactor_temp"],
            state["core_metrics"]["pipeline_pressure"],
            state["core_metrics"]["flow_rate"],
            state["core_metrics"]["vibration"],
            state["chemical_properties"]["ph"],
            state["chemical_properties"]["oxygen_content"],
            state["equipment_health"]["pump_efficiency"],
            state["safety_status"]["risk_score"]
        ]
    
    def _generate_recommendation(self, risk, state):
        if risk > 0.8:
            return (
                "CRITICAL: Immediate shutdown required", 
                ["emergency_shutdown", "notify_safety_team", "schedule_maintenance"]
            )
        elif risk > 0.6:
            return (
                "WARNING: Reduce pressure and schedule inspection", 
                ["reduce_pressure 20%", "increase_cooling", "schedule_inspection"]
            )
        elif risk > 0.4:
            return (
                "CAUTION: Monitor closely and optimize parameters", 
                ["optimize_parameters", "increase_monitoring_frequency"]
            )
        else:
            return (
                "NORMAL: Continue standard operations", 
                ["routine_check", "log_metrics"]
            )

# ======================
# 5. IMMERSIVE DASHBOARD
# ======================
app = Dash(__name__, 
          external_stylesheets=[dbc.themes.DARKLY],
          assets_folder="assets")  # This line tells Dash where to find CSS

app.layout = dbc.Container([
    dcc.Store(id='plant-state-store'),
    dcc.Interval(id='update-interval', interval=1000),
    dcc.Interval(id='ai-update-interval', interval=5000),
    
    # Header with status lights
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1("InoSense AI Command Center", className="display-4"),
                html.Div(id='system-status-lights', className="status-lights")
            ], className="header-container")
        ], width=12)
    ], className="header-row"),
        # ▼▼▼ PASTE THIS RIGHT HERE ▼▼▼
    # New Control Buttons
    dbc.Row([
        dbc.Col(dbc.Button("Show Temperature", id='temp-btn', color="info"), width=4),
        dbc.Col(dbc.Button("Show Pressure", id='pressure-btn', color="warning"), width=4),
        dbc.Col(dbc.Button("Simulate Emergency", id='fake-emergency', color="danger"), width=4),
    ], className="mb-4"),
    # ▲▲▲ END OF PASTE ▲▲▲
    
    # Main dashboard content
    dbc.Row([
        # Left column - 3D Visualization and Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Live 3D Plant Visualization", className="card-header"),
                dbc.CardBody(dcc.Graph(id='3d-plant-view'))
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Control Panel", className="card-header"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Button("Voice Command", id='voice-btn', color="primary", className="mr-2"), width=4),
                        dbc.Col(dbc.Button("Emergency Stop", id='emergency-btn', color="danger"), width=4),
                        dbc.Col(dbc.Button("Generate Report", id='report-btn', color="info"), width=4)
                    ]),
                    html.Div(id='voice-output', className="mt-3"),
                    html.Audio(id='alert-sound', src='', autoPlay=False)
                ])
            ])
        ], width=6),
        
        # Right column - AI Insights
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("AI Risk Assessment", className="card-header"),
                dbc.CardBody([
                    dcc.Graph(id='risk-gauge'),
                    html.Div(id='ai-recommendation-card', className="mt-3")
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("Anomaly Detection", className="card-header"),
                dbc.CardBody([
                    dcc.Graph(id='anomaly-detection-plot'),
                    html.Div(id='anomaly-details', className="mt-2")
                ])
            ])
        ], width=6)
    ]),
    
    # Bottom row - Sensor History
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sensor History & Trends", className="card-header"),
                dbc.CardBody(dcc.Graph(id='sensor-history-plot'))
            ])
        ], width=12)
    ], className="mt-4"),
    
    # Hidden elements for callbacks
    html.Div(id='dummy-output', style={'display': 'none'}),
    dcc.Download(id="report-download")
], fluid=True, className="dashboard-container")

# ======================
# 6. DASH CALLBACKS
# ======================
@app.callback(
    [Output('3d-plant-view', 'figure'),
     Output('risk-gauge', 'figure'),
     Output('sensor-history-plot', 'figure'),
     Output('anomaly-detection-plot', 'figure'),
     Output('ai-recommendation-card', 'children'),
     Output('anomaly-details', 'children'),
     Output('system-status-lights', 'children'),
     Output('alert-sound', 'src'),
     Output('plant-state-store', 'data')],
    [Input('update-interval', 'n_intervals'),
     Input('ai-update-interval', 'n_intervals'),
     Input('emergency-btn', 'n_clicks')],
    [State('plant-state-store', 'data')]
)
def update_all_components(n1, n2, emergency_clicks, current_state):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Initialize systems
    config = SystemConfig()
    plant = ChemicalPlantTwin(config)
    ai_system = AIPredictiveSystem()
    
    # Get current plant state
    plant_state = plant.update_state()
    
    # Get AI predictions
    prediction = ai_system.predict_failure(plant_state)
    
    # Handle emergency stop
    alert_sound = ''
    if triggered_id == 'emergency-btn' and emergency_clicks:
        plant_state["safety_status"]["overall"] = "emergency"
        alert_sound = generate_alert_sound("emergency")
    
    # 3D Visualization
    fig_3d = create_3d_visualization(plant_state)
    
    # Risk Gauge
    fig_gauge = create_risk_gauge(prediction["risk_score"])
    
    # Sensor History
    fig_history = create_sensor_history(plant)
    
    # Anomaly Detection
    fig_anomaly, anomaly_details = create_anomaly_visualization(plant_state, prediction)
    
    # AI Recommendation Card
    recommendation_card = create_recommendation_card(prediction)
    
    # Status Lights
    status_lights = create_status_lights(plant_state)
    
    return (
        fig_3d, fig_gauge, fig_history, fig_anomaly, 
        recommendation_card, anomaly_details, status_lights,
        alert_sound, plant_state
    )

@app.callback(
    Output('voice-output', 'children'),
    [Input('voice-btn', 'n_clicks')]
)
def handle_voice_command(n_clicks):
    if n_clicks and n_clicks > 0:
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio = r.listen(source, timeout=5)
                text = r.recognize_google(audio)
                return dbc.Alert(f"Voice command received: {text}", color="success")
        except Exception as e:
            return dbc.Alert(f"Voice recognition error: {str(e)}", color="danger")
    return None

@app.callback(
    Output("report-download", "data"),
    [Input('report-btn', 'n_clicks')],
    [State('plant-state-store', 'data')]
)
def generate_report(n_clicks, plant_state):
    if n_clicks and n_clicks > 0:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Add title
        pdf.cell(200, 10, txt="InoSense System Report", ln=1, align='C')
        pdf.cell(200, 10, txt=f"Generated on: {datetime.utcnow()}", ln=1, align='C')
        
        # Add system status
        pdf.cell(200, 10, txt="Current System Status:", ln=1)
        pdf.cell(200, 10, txt=f"Overall Status: {plant_state['safety_status']['overall']}", ln=1)
        pdf.cell(200, 10, txt=f"Risk Score: {plant_state['safety_status']['risk_score']:.2f}", ln=1)
        
        # Add QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(f"InoSense Report {datetime.utcnow()}")
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white')
        img.save("report_qr.png")
        pdf.image("report_qr.png", x=50, w=100)
        
        # Save to bytes
        report_bytes = BytesIO()
        pdf.output(report_bytes)
        report_bytes.seek(0)
        
        return dcc.send_bytes(report_bytes.read(), "inosense_report.pdf")
    return None

# ======================
# 7. VISUALIZATION HELPERS
# ======================
def create_3d_visualization(plant_state):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=[0, 1, 1, 0, 0.5],
            y=[0, 0, 1, 1, 0.5],
            z=[0, 0, 0, 0, 1],
            mode='markers',
            marker=dict(
                size=20,
                color=[plant_state["core_metrics"]["reactor_temp"]]*5,
                colorscale='thermal',
                opacity=0.8,
                cmin=300,
                cmax=500
            ),
            hovertext=[
                f"Reactor: {plant_state['core_metrics']['reactor_temp']:.1f}°K",
                f"Pipeline A: {plant_state['core_metrics']['pipeline_pressure']:.1f}kPa",
                f"Pipeline B: {plant_state['core_metrics']['flow_rate']:.1f}L/s",
                f"Cooling System: {plant_state['chemical_properties']['ph']:.1f}pH",
                f"Control Tower"
            ]
        )
    ])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.7)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="Live 3D Plant Visualization"
    )
    return fig

def create_risk_gauge(risk_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "System Risk Assessment"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': risk_score}
        }
    ))
    fig.update_layout(margin=dict(l=30, r=30, b=30, t=50))
    return fig

def create_sensor_history(plant):
    # Create a DataFrame with historical data
    history = list(plant.history)[-100:]  # Last 100 readings
    times = [datetime.utcnow() - timedelta(seconds=i) for i in range(len(history))]
    
    df = pd.DataFrame({
        "Time": times,
        "Temperature": [s["core_metrics"]["reactor_temp"] for s in history],
        "Pressure": [s["core_metrics"]["pipeline_pressure"] for s in history],
        "Flow Rate": [s["core_metrics"]["flow_rate"] for s in history],
        "Vibration": [s["core_metrics"]["vibration"] for s in history]
    })
    
    fig = px.line(df, x="Time", y=["Temperature", "Pressure", "Flow Rate", "Vibration"],
                 title="Sensor History Trends",
                 template="plotly_dark")
    
    fig.update_layout(
        hovermode="x unified",
        legend_title="Parameters",
        yaxis_title="Value",
        xaxis_title="Time"
    )
    return fig

def create_anomaly_visualization(plant_state, prediction):
    features = {
        "Temperature": plant_state["core_metrics"]["reactor_temp"],
        "Pressure": plant_state["core_metrics"]["pipeline_pressure"],
        "Flow Rate": plant_state["core_metrics"]["flow_rate"],
        "Vibration": plant_state["core_metrics"]["vibration"],
        "pH": plant_state["chemical_properties"]["ph"],
        "Oxygen": plant_state["chemical_properties"]["oxygen_content"]
    }
    
    fig = go.Figure()
    for param, value in features.items():
        fig.add_trace(go.Bar(
            x=[param],
            y=[value],
            name=param,
            marker_color="red" if prediction["is_anomaly"] else "green"
        ))
    
    fig.update_layout(
        title="Current Sensor Readings vs Normal Ranges",
        xaxis_title="Parameter",
        yaxis_title="Value",
        showlegend=False
    )
    
    details = html.Div([
        html.H5("Anomaly Details", className="mb-2"),
        html.P(f"Detected: {'Yes' if prediction['is_anomaly'] else 'No'}"),
        html.P(f"Confidence: {prediction['confidence']*100:.1f}%"),
        html.P(f"Most deviant parameter: {max(features, key=lambda k: abs(features[k] - (300 if k == 'Temperature' else 150 if k == 'Pressure' else 0)))}")
    ])
    
    return fig, details


def create_recommendation_card(prediction):
    color_map = {
        "CRITICAL": "danger",
        "WARNING": "warning",
        "CAUTION": "info",
        "NORMAL": "success"
    }
    status = prediction["recommendation"].split(":")[0]
    
    return dbc.Card([
        dbc.CardHeader(f"AI Recommendation: {status}", 
                      className=f"bg-{color_map.get(status, 'secondary')}"),
        dbc.CardBody([
            html.H4(prediction["recommendation"], className="card-title"),
            html.Hr(),
            html.H5("Recommended Actions:", className="mb-2"),
            html.Ul([html.Li(action) for action in prediction["actions"]]),
            html.P(f"Confidence: {prediction['confidence']*100:.1f}%", 
                  className="text-muted mt-3")
        ])
    ])
def create_status_lights(plant_state):
    status = plant_state["safety_status"]["overall"]
    colors = {
        "normal": "green",
        "warning": "yellow",
        "critical": "red",
        "emergency": "red"
    }
    
    return html.Div([
        html.Div(className=f"status-light {colors.get(status, 'gray')}"),
        html.Span(f"System Status: {status.upper()}", className="status-text ml-2")
    ], className="status-container")

def generate_alert_sound(alert_type):
    duration = 1.0  # seconds
    if alert_type == "emergency":
        freq = 880  # Hz
    else:
        freq = 440  # Hz
    
    samples = np.sin(2 * np.pi * freq * np.arange(44100 * duration) / 44100)
    return 'data:audio/wav;base64,' + base64.b64encode(samples.astype(np.float32).tobytes()).decode()

# ======================
# 8. FASTAPI BACKEND
# ======================
fastapi_app = FastAPI(title="InoSense API",
                     description="REST API for InoSense Chemical Plant Monitoring",
                     version="1.0.0")

@fastapi_app.websocket("/ws/sensor-data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    config = SystemConfig()
    plant = ChemicalPlantTwin(config)
    ai = AIPredictiveSystem()
    
    while True:
        try:
            plant_state = plant.update_state()
            prediction = ai.predict_failure(plant_state)
            
            await websocket.send_json({
                "timestamp": str(datetime.utcnow()),
                "plant_state": plant_state,
                "prediction": prediction
            })
            await asyncio.sleep(1)
        except websockets.exceptions.ConnectionClosed:
            break

@fastapi_app.get("/api/plant-state")
async def get_plant_state():
    config = SystemConfig()
    plant = ChemicalPlantTwin(config)
    return plant.update_state()

@fastapi_app.get("/api/ai-prediction")
async def get_ai_prediction():
    config = SystemConfig()
    plant = ChemicalPlantTwin(config)
    ai = AIPredictiveSystem()
    return ai.predict_failure(plant.update_state())
# ======================
# 10. VOICE CONTROL
# ======================
@app.callback(
    Output('voice-output', 'children'),
    Input('voice-btn', 'n_clicks'),
    prevent_initial_call=True
)
def handle_voice(n_clicks):
    if not n_clicks:
        return "No click detected"
    
    print("\n🔥 Button clicked! Starting voice recognition...")  # Debug line
    
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎤 Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            
            print("🔊 Speak now (say 'temperature' or 'emergency')...")
            audio = r.listen(source, timeout=5)
            
        command = r.recognize_google(audio).lower()
        print(f"✅ Heard: {command}")
        return dbc.Alert(f"Command: {command}", color="success")
        
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return dbc.Alert("Couldn't understand - speak louder!", color="warning")
    except Exception as e:
        print(f"💥 Error: {str(e)}")
        return dbc.Alert(f"System error: {str(e)}", color="danger")
# ======================
# 9. MAIN APPLICATION
# ======================
def run_dash_app():
    time.sleep(2)  # Add this line
    app.run(port=8050, debug=False)  # Changed from run_server

def run_fastapi_app():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# ▼▼▼ PASTE THIS ▼▼▼
@app.callback(
    Output('sensor-history-plot', 'figure'),
    [Input('temp-btn', 'n_clicks'),
     Input('pressure-btn', 'n_clicks')]
)
def update_graph(temp_clicks, pressure_clicks):
    if temp_clicks and (not pressure_clicks or temp_clicks > pressure_clicks):
        return px.line(title="Temperature History")
    return px.line(title="Pressure History")

@app.callback(
    Output('ai-recommendation-card', 'children'),
    Input('fake-emergency', 'n_clicks')
)
def fake_alert(n):
    if n:
        return dbc.Alert("🚨 CRITICAL ALERT: Simulated emergency!", color="danger")
# ▲▲▲ END OF PASTE ▲▲▲

if __name__ == "__main__":
    print("""
     ___       ________  ________  ________  _______   ________     
    |\  \     |\   __  \|\   __  \|\   ____\|\  ___ \ |\   ____\    
    \ \  \    \ \  \|\  \ \  \|\  \ \  \___|\ \   __/|\ \  \___|    
     \ \  \    \ \   ____\ \   __  \ \_____  \ \  \_|/_\ \  \       
      \ \  \____\ \  \___|\ \  \ \  \|____|\  \ \  \_|\ \ \  \____  
       \ \_______\ \__\    \ \__\ \__\____\_\  \ \_______\ \_______\
        \|_______|\|__|     \|__|\|__|\_________\|_______|\|_______|
                                    \|_________|                     
    """)
    print("Starting InoSense AI Monitoring System...")
    
    # Start Dash in a separate thread
    dash_thread = threading.Thread(target=run_dash_app)
    dash_thread.daemon = True
    dash_thread.start()
    
    # Give Dash a moment to start
    time.sleep(3)
    
    # Run FastAPI in main thread
    print("Dashboard: http://localhost:8050")
    print("API Docs: http://localhost:8000/docs")
    run_fastapi_app()
    