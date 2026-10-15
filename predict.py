import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

decision_tree_model = joblib.load(MODEL_PATH)

def predict_land_water(latitude, longitude):
    """
    Latitude aur longitude se Land ya Water predict karein.
    """

    # Input validation
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    coordinates = pd.DataFrame(
    [[latitude, longitude]],
    columns=["latitude", "longitude"]
   )

    prediction = int(
        decision_tree_model.predict(coordinates)[0]
    )

    labels = {
        0: "Water",
        1: "Land"
    }

    return labels[prediction]