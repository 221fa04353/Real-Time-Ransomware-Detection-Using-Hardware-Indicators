Real-Time Ransomware Detection Using Hardware Performance Counters

SETUP:

1. python -m venv venv
2. venv\Scripts\activate
3. pip install -r requirements.txt

Place HPC dataset folder:
dataset/raw/data/train/*.csv

RUN:

cd scripts
python train_model.py
python evaluate_model.py

Start API:
uvicorn backend.api_server:app --reload

Dashboard:
streamlit run frontend/dashboard.py

Realtime Detection:
python backend/realtime_monitor.py