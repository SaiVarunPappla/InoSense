InoSense: Smart Monitoring of CHemical Processes
l1.png, l2.png, l3.png
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
Use the dashboard to watch plant data, change settings, and create reports.
Click "Voice Command" to give voice instructions (e.g., "temperature").
Use the API with WebSocket (/ws/sensor-data) or endpoints (/api/plant-state, /api/ai-prediction). dashboard5.png
Download PDF reports from the dashboard.

Screenshots : 
demo1.png
demo2.png
demo3.png
demo4.png
demo5.png
demo6.png

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
