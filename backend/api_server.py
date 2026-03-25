from fastapi import FastAPI, Body
import joblib
from pathlib import Path

# ---------- CREATE APP FIRST ----------
app = FastAPI()

# ---------- LOAD MODEL ----------
BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "models" / "random_forest.pkl"
scaler_path = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ---------- HOME ROUTE ----------
@app.get("/")
def home():
    return {"message": "HPC Ransomware Detection API Running"}

# ---------- PREDICTION ROUTE ----------
@app.post("/predict")
def predict(data: list = Body(...)):

    expected = model.n_features_in_

    if len(data) != expected:
        return {
            "error": f"Model expects {expected} features but received {len(data)}"
        }

    x = scaler.transform([data])
    prob = model.predict_proba(x)[0][1]

    return {"ransomware_probability": float(prob)}