# Land vs Water Classification and Image Segmentation

A machine-learning project that predicts whether a geographic coordinate represents **Land** or **Water**, and performs pixel-level land/water segmentation on satellite imagery.

The application provides a **FastAPI backend** and a **Streamlit frontend**.

## Features

### Part 1 — Coordinate Classification
- Enter latitude and longitude manually or select a location on the map.
- Send the coordinates to the FastAPI backend.
- Display the predicted class: `Land` or `Water`.
- The coordinate model uses two input features: `latitude` and `longitude`.

### Part 2 — Image Segmentation
Two U-Net segmentation models are available:

1. **Improved RGB U-Net**
   - Accepts `.jpg`, `.jpeg`, and `.png` images through the frontend/API.
   - Uses a three-channel RGB input.
   - Intended for RGB imagery similar to its training data. Predictions on ordinary phone photographs may be unreliable because the model was trained on satellite imagery.

2. **Six-Band U-Net**
   - Accepts `.tif` and `.tiff` satellite imagery.
   - Expects six Sentinel-2 bands in this order: **B2, B3, B4, B8, B11, B12**.
   - The model preprocessing normalizes reflectance values by dividing by `10000`.
   - The displayed mask uses white for predicted water and black for predicted land/non-water, according to the current application implementation.

## Project Structure

```text
Land-Water-Classification/
├── app.py
├── main.py
├── predict.py
├── models/
│   ├── best_model.joblib
│   ├── model_metadata.json
│   └── segmentation/
│       ├── water_unet_6band.pth
│       └── water_unet_rgb_improved.pth
├── test_images/
│   └── S2A_L2A_20190318_N0211_R061_6Bands_S2.tif
└── .venv312/
```

*Note: `test_images/` and `.venv312/` are local working directories. You do not need to commit the virtual environment or large model/test files to GitHub. If the files are excluded from the repository, document how to obtain them separately.*

## Requirements

- Python 3.12 environment (the current local setup uses `.venv312`)
- FastAPI and Uvicorn
- Streamlit
- Requests
- NumPy
- Pillow
- Rasterio for GeoTIFF processing
- PyTorch for U-Net inference
- Folium and `streamlit-folium` for the interactive map
- Any additional packages imported by `main.py`, `app.py`, and `predict.py`

Install the project dependencies from the repository's dependency file if available. If there is no dependency file yet, review the imports in the source files and create a `requirements.txt` for the exact packages and versions used by your environment.

## Setup and Run

Open a terminal and move to the project directory:

```bash
cd ~/ml_projects/Land-Water-Classification
```

Activate the existing virtual environment:

```bash
source .venv312/bin/activate
```

### 1. Start the FastAPI backend

Run this command in the first terminal:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

- API base URL: `http://127.0.0.1:8000`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Keep this terminal running while using the frontend.

### 2. Start the Streamlit frontend

Open a second terminal:

```bash
cd ~/ml_projects/Land-Water-Classification
source .venv312/bin/activate
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## API Endpoints

The application currently includes these POST endpoints:

| Endpoint | Purpose | Expected input |
|---|---|---|
| `/predict/coordinates` | Predict Land or Water from coordinates | Latitude and longitude in the request body |
| `/predict/segmentation/rgb` | Run RGB U-Net segmentation | JPG, JPEG, or PNG image |
| `/predict/segmentation` | Run six-band U-Net segmentation | Six-band TIFF/GeoTIFF image |

For the exact request schema and response format, open `http://127.0.0.1:8000/docs` after starting the backend.

## Testing Performed

The following basic integration checks have been completed in the local application:

- Coordinate prediction returned **Land** for Lahore (`31.5204, 74.3587`).
- Coordinate prediction returned **Water** for a test coordinate in the Arabian Sea (`20, 65`).
- Coordinate prediction returned **Water** for a test coordinate in the central Pacific (`0, -140`).
- The RGB U-Net frontend test completed successfully.
- The six-band U-Net test completed successfully using a TIFF image verified to contain six bands (760 × 711 pixels).

These are basic functional tests, not a guarantee that every prediction is geographically or visually correct.

## Model Metadata

The project's current `models/model_metadata.json` records the coordinate model as a Decision Tree and includes these previously recorded metrics:

- Original test accuracy: **0.9881**
- Geographic reference accuracy: **0.98**
- Geographic reference F1 score: **0.9643**

These are existing project metadata values; they have not been recalculated by this README.

## Important Limitations

- Coordinate predictions depend on the training dataset and geographic coverage. They should not be treated as authoritative geographic data.
- Segmentation output is a model prediction, not a verified map or scientific measurement.
- The six-band model requires compatible six-band satellite data with the expected band order and value scaling. An arbitrary six-band TIFF may not be compatible.
- The RGB model was trained on satellite imagery. Although the endpoint accepts common image formats, results on regular photographs or images from a different sensor may be inaccurate.
- Confirm the predicted mask visually before using it for analysis or decision-making.

