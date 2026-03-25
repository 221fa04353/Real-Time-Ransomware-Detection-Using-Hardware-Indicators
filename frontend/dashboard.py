import streamlit as st
import time
from pathlib import Path
import pandas as pd
import re

# ---------------- PAGE ----------------
st.set_page_config(layout="wide")
st.title("🛡 Enterprise Ransomware Detection SOC Dashboard")

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "logs" / "detection.log"
ALARM_FILE = BASE_DIR / "frontend" / "assets" / "alarm.mp3"

# ---------------- UI AREAS ----------------
banner = st.empty()
risk_meter = st.empty()
chart_area = st.empty()
timeline_area = st.empty()
log_area = st.empty()

probabilities = []
timeline = []
alerts_sent = 0


# ---------------- HELPERS ----------------
def extract_probability(line):
    match = re.search(r"probability=(\d+\.\d+)", line)
    return float(match.group(1)) if match else None


def risk_level(prob):
    if prob < 0.4:
        return "LOW", "green"
    elif prob < 0.7:
        return "MEDIUM", "orange"
    else:
        return "CRITICAL", "red"


# ---------------- LOOP ----------------
while True:

    if LOG_FILE.exists():

        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if len(lines) == 0:
            banner.success("🟢 SYSTEM SECURE — NO ACTIVITY")
            time.sleep(2)
            continue

        # -------- CHECK ONLY LAST LINE --------
        last_line = lines[-1]

        latest_ransomware = "RANSOMWARE DETECTED" in last_line

        output = ""
        latest_prob = 0

        for line in lines[-40:]:

            prob = extract_probability(line)
            if prob is not None:
                probabilities.append(prob)
                latest_prob = prob

            if "RANSOMWARE DETECTED" in line:
                output += f"<p style='color:red;font-weight:bold'>🚨 {line}</p>"
            else:
                output += f"<p style='color:lightgreen'>✅ {line}</p>"

        # -------- ALERT BANNER --------
        if latest_ransomware:

            banner.markdown(
                """
                <div style="
                    background-color:red;
                    padding:15px;
                    text-align:center;
                    color:white;
                    font-size:28px;
                    font-weight:bold;
                    animation: blinker 1s linear infinite;">
                    🚨 RANSOMWARE ATTACK ACTIVE 🚨
                </div>

                <style>
                @keyframes blinker {50% {opacity: 0;}}
                </style>
                """,
                unsafe_allow_html=True
            )

            if ALARM_FILE.exists():
                st.audio(str(ALARM_FILE), autoplay=True)

            st.toast("📧 SOC EMAIL ALERT SENT!", icon="🚨")

        else:
            banner.success("🟢 SYSTEM SECURE — NO ACTIVE THREAT")

        # -------- GRAPH --------
        if len(probabilities) > 5:
            df = pd.DataFrame(
                {"Probability": probabilities[-60:]}
            )
            chart_area.line_chart(df)

        log_area.markdown(output, unsafe_allow_html=True)

    else:
        banner.warning("Waiting for monitoring system...")

    time.sleep(2)