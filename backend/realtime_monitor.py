import psutil
import time
import joblib
import logging
import numpy as np
from pathlib import Path

# ---------------- PATH SETUP ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

FLAG_FILE = BASE_DIR / "attack.flag"
INTENSITY_FILE = BASE_DIR / "attack_intensity.txt"
LOG_FILE = LOG_DIR / "detection.log"

# ---------------- LOAD MODEL ----------------
model = joblib.load(BASE_DIR / "models" / "random_forest.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

# ---------------- LOGGER (FIXED) ----------------
logger = logging.getLogger("ransomware_monitor")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
formatter = logging.Formatter("%(asctime)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

print("✅ Monitoring Started")
print("📄 Writing logs to:", LOG_FILE)


# ---------------- FEATURE GENERATION ----------------
def generate_features():

    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_io_counters()
    processes = len(psutil.pids())

    attack_level = 0

    if INTENSITY_FILE.exists():
        try:
            attack_level = int(INTENSITY_FILE.read_text().strip())
        except:
            attack_level = 0

    # ----- NORMAL BASELINE -----
    c2 = cpu * 2000
    c0 = mem * 1800
    f729 = processes * 1200
    f129 = disk.read_bytes % 5_000_000
    f229 = disk.write_bytes % 5_000_000
    ff9a = abs(cpu - mem) * 1500

    # ----- ATTACK ALIGNMENT -----
    if attack_level > 0:
        print(f"🔥 Applying ransomware intensity {attack_level}%")

        factor = 1 + attack_level / 20

        c2 *= factor
        c0 *= factor
        f729 *= factor
        f229 *= factor * 2
        ff9a *= factor

    return np.array([[c2, c0, f729, f129, f229, ff9a]])


# ---------------- MONITOR LOOP ----------------
while True:

    features = generate_features()
    features = scaler.transform(features)

    prob = model.predict_proba(features)[0][1]

    simulator_active = FLAG_FILE.exists()

    if simulator_active and prob > 0.30:
        message = f"🚨 RANSOMWARE DETECTED | probability={prob:.3f}"
        print(message)
        logger.warning(message)
    else:
        message = f"✅ NORMAL ACTIVITY | probability={prob:.3f}"
        print(message)
        logger.info(message)

    # ⭐ FORCE WRITE TO FILE (VERY IMPORTANT)
    file_handler.flush()

    time.sleep(2)