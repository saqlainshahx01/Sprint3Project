
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict_land_water


app = FastAPI(
    title="Land-Water Classification API",
    description="Coordinate prediction and satellite image segmentation",
    version="1.0.0",
)


class CoordinateInput(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


@app.get("/")
def home():
    return {
        "message": "Land-Water Classification API is running",
        "available_tasks": [
            "Coordinate-based land/water prediction",
            "Satellite image segmentation (coming next)",
        ],
    }


@app.post("/predict/coordinates")
def predict_coordinates(data: CoordinateInput):
    try:
        result = predict_land_water(
            latitude=data.latitude,
            longitude=data.longitude,
        )

        return {
            "latitude": data.latitude,
            "longitude": data.longitude,
            "prediction": result,
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
