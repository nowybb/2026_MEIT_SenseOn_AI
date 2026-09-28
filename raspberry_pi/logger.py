import csv
import os
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "senseon_log.csv")


def save_log(hazard, e2e_latency_ms=None):
    file_exists = os.path.exists(LOG_FILE)

    # AI 결과가 None이면 SAFE 상태로 기록
    if hazard is None:
        hazard = {
            "object": "none",
            "direction": "CENTER",
            "risk": "SAFE",
            "ttc": None
        }

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "object",
                "direction",
                "risk",
                "ttc",
                "e2e_latency_ms"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            hazard["object"],
            hazard["direction"],
            hazard["risk"],
            hazard["ttc"],
            e2e_latency_ms
        ])