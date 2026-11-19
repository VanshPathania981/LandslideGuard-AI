
import os
import json
import uuid
from datetime import datetime

HISTORY_DIR = "data/history"
HISTORY_FILE = os.path.join(HISTORY_DIR, "prediction_history.json")


def initialize_history():
    os.makedirs(HISTORY_DIR, exist_ok=True)

    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w") as file:
            json.dump([], file, indent=4)


def save_prediction(image_name, threshold, total_pixels,
                    predicted_pixels, coverage):

    initialize_history()

    history = load_history()

    record = {
        "id": str(uuid.uuid4()),
        "image_name": image_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": float(threshold),
        "total_pixels": int(total_pixels),
        "predicted_pixels": int(predicted_pixels),
        "coverage": float(coverage)
    }

    history.append(record)

    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file, indent=4)

    return record


def load_history():
    initialize_history()

    with open(HISTORY_FILE, "r") as file:
        return json.load(file)
