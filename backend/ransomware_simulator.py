import os
import time
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FLAG = BASE_DIR / "attack.flag"
INTENSITY = BASE_DIR / "attack_intensity.txt"

print("🚨 Ransomware Simulation Started")

FLAG.touch()

try:
    level = 0
    while True:
        # gradually increase attack intensity
        level = min(level + random.randint(5,10), 100)

        with open(INTENSITY, "w") as f:
            f.write(str(level))

        print(f"🔥 Attack intensity: {level}%")
        time.sleep(2)

except KeyboardInterrupt:
    print("Stopping simulator...")
    if FLAG.exists():
        FLAG.unlink()
    if INTENSITY.exists():
        INTENSITY.unlink()