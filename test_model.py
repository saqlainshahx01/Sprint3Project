
import joblib
from pathlib import Path

# Project ke models folder ka path
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

# Saved Decision Tree model load karo
decision_tree_model = joblib.load(MODEL_PATH)

print("Model successfully loaded!")

# Example input: latitude and longitude
latitude = 31.5204
longitude = 74.3587

# Prediction ke liye coordinates
coordinates = [[latitude, longitude]]

prediction = decision_tree_model.predict(coordinates)[0]

# Numeric label ko readable class mein convert karo
labels = {
    0: "Water",
    1: "Land"
}

print("Latitude:", latitude)
print("Longitude:", longitude)
print("Predicted class:", labels[int(prediction)])