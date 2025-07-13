InoSense : Smart Monitorting of Chemmical Processes
[Dashboard1](assets/l1.png) [Dashboard2](assets/l2.png) [Dashboard3](assets/l3.png)
Cutting-edge AI-driven solution for real-time chemical process monitoring and optimization.

Overview :

Welcome to InoSense, a full-stack AI/ML-powered industrial monitoring system designed to revolutionize chemical plant management. This project integrates a 4D digital twin, predictive maintenance, anomaly detection, voice control, blockchain data integrity, and a multi-user dashboard with automated reporting. Built with Python, TensorFlow, Dash, and FastAPI, InoSense delivers real-time insights, optimizes efficiency (up to 15-25%), and ensures data security—making it a standout solution for industrial intelligence.

Features :

Real-Time 4D Digital Twin: Visualizes plant thermal profiles with dynamic updates.
Predictive Maintenance AI: Utilizes Transformer-based models and Isolation Forest for failure prediction.
Anomaly Detection Engine: Identifies deviations with confidence scoring and actionable recommendations.
Voice Control Interface: Enables hands-free operation with speech recognition.
Blockchain Data Integrity: Ensures tamper-proof data logging with timestamp validation.
Multi-User Dashboard: Interactive Dash-based UI with customizable risk thresholds.
Automated Reporting: Generates PDF reports with QR codes for traceability.

Tech Stack :

=======
InoSense: Smart Monitoring of CHemical Processes
[Dashboard1](assets/l1.png) [Dashboard2](assets/l2.png) [Dashboard3](assets/l3.png)
A sophisticated AI-driven solution for real-time chemical process monitoring and optimization.

Overview:
InoSense is a full-stack AI/ML-powered industrial monitoring system designed to enhance chemical plant management. It features a 4D digital twin, predictive maintenance, anomaly detection, voice control, blockchain data integrity, and a multi-user dashboard with automated reporting. Built using Python, TensorFlow, Dash, and FastAPI, InoSense provides real-time insights and optimizes operational efficiency.


Features:
Real-time 4D digital twin for plant visualization.
Predictive maintenance using advanced AI models.
Anomaly detection with actionable recommendations.
Voice control interface for hands-free operation.
Blockchain-based data integrity for secure logging.
Interactive multi-user dashboard with customizable settings.
Automated PDF reporting with QR code traceability. 

Tech Stack : 
Frontend: Dash, Plotly, Dash Bootstrap Components
Backend: FastAPI, WebSockets, Uvicorn
AI/ML: TensorFlow, scikit-learn, LSTM, MultiHeadAttention
Data Handling: pandas, numpy, qrcode, fpdf
Voice: speech_recognition, pyaudio

Security: Blockchain (custom implementation)

Installation:

Prerequisites
Python 3.9+
pip  

Setup :
Clone the repository:
git clone https://github.com/SaiVarunPappla/inosense.git
cd inosense  

Install dependencies:
pip install -r requirements.txt

Run the application: 
python inosense.py
Access the dashboard at http://localhost:8050
Access API documentation at http://localhost:8000/docs

requirements.txt:

Security: Custom blockchain implementation

Installation:
Prerequisites
Python 3.9 or higher
pip (Python package manager)

Steps : 
1)Download the project:
git clone https://github.com/SaiVarunPappla/inosense.git
cd inosense
2)Install required packages:
pip install -r requirements.txt
3)Create two folders:
Make an assets folder (for pictures and files).
Make a templates folder (can be empty).
4)Run the program:
python inosense.py
Open your browser and go to http://localhost:8050 for the dashboard.
Go to http://localhost:8000/docs for API details.

requirements.txt Content :

numpy
pandas
tensorflow
scikit-learn
plotly
dash
dash-bootstrap-components
fastapi
uvicorn
speechrecognition
pyaudio
sounddevice
matplotlib
fpdf
qrcode
websockets

Usage :


Dashboard: Navigate the interactive UI to monitor plant metrics, adjust risk thresholds, and generate reports.
Voice Commands: Click "Voice Command" to issue instructions (e.g., "temperature" or "emergency").
API: Use WebSocket (/ws/sensor-data) or REST endpoints (/api/plant-state, /api/ai-prediction) for integration.
Reports: Download PDF reports with current system status and QR-coded traceability. [voice command](assets/dashboard5.png)

Project Structure :
inosense/
├── assets/           # Static files (images, CSS)
├── templates/        # Jinja2 templates (optional)
├── inosense.py      # Main application file
├── requirements.txt # Dependency list
└── README.md        # This file

Contributing
Contributions are welcome! Please fork the repository and submit pull requests. For major changes, open an issue first to discuss.

1)Fork the repo.
2)Create a feature branch (git checkout -b feature/awesome-feature).
3)Commit changes (git commit -m "Add awesome feature").
4)Push to the branch (git push origin feature/awesome-feature).
5)Open a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Developed by Varun Pappla
Links:
GitHub
LinkedIn 
Use the dashboard to watch plant data, change settings, and create reports.
Click "Voice Command" to give voice instructions (e.g., "temperature").
Use the API with WebSocket (/ws/sensor-data) or endpoints (/api/plant-state, /api/ai-prediction). dashboard5.png
Download PDF reports from the dashboard.

Screenshots : 

[ScreenShot1](assets/demo1.png)
[ScreenShot2](assets/demo2.png)
[ScreenShot3](assets/demo3.png)
[ScreenShot4](assets/demo4.png)
[ScreenShot5](assets/demo5.png)
[ScreenShot6](assets/demo6.png)

Project Structure:
inosense/
├── assets/           # Folder for pictures and files
├── templates/        # Folder for templates (can be empty)
├── inosense.py      # Main program file
├── requirements.txt # List of needed packages
└── README.md        # This file

Contributing : 

You can help improve this project! Here’s how:
1)Make a copy of this project (fork it) on GitHub.
2)Create a new branch for your changes:
git checkout -b my-new-feature
3)Save your changes:
git commit -m "Add my new feature"
4)Send your changes back:
git push origin my-new-feature
5)Ask to add your changes by making a pull request on GitHub. 

License
This project uses the MIT License. See the LICENSE file for more information.

Acknowledgments
Created by Varun Pappla
Links : 
Linkedin : https://www.linkedin.com/in/pappla-sai-varun-874902200/

Future Plans:
-Add support for monitoring multiple plants.
-Create a mobile app version.
-Improve AI with advanced optimization techniques.
