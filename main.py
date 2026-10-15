
import io
from pathlib import Path

import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp

from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from predict import predict_land_water


# --------------------------------------------------
# 1. PATHS AND DEVICE
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Existing six-band model
UNET_6BAND_PATH = (
    BASE_DIR
    / "models"
    / "segmentation"
    / "water_unet_6band.pth"
)

# New improved RGB model
UNET_RGB_PATH = (
    BASE_DIR
    / "models"
    / "segmentation"
    / "water_unet_rgb_improved.pth"
)

DEVICE = torch.device("cpu")


# --------------------------------------------------
# 2. LOAD SIX-BAND U-NET MODEL
# --------------------------------------------------

unet_model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=6,
    classes=2,
    activation=None,
)

checkpoint_6band = torch.load(
    UNET_6BAND_PATH,
    map_location=DEVICE,
    weights_only=True,
)

unet_model.load_state_dict(checkpoint_6band)
unet_model.to(DEVICE)
unet_model.eval()


# --------------------------------------------------
# 3. LOAD IMPROVED RGB U-NET MODEL
# --------------------------------------------------

rgb_model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=2,
    activation=None,
)

checkpoint_rgb = torch.load(
    UNET_RGB_PATH,
    map_location=DEVICE,
    weights_only=True,
)

rgb_model.load_state_dict(checkpoint_rgb)
rgb_model.to(DEVICE)
rgb_model.eval()

print("Six-band U-Net loaded successfully!")
print("Improved RGB U-Net loaded successfully!")
print("Device:", DEVICE)


# --------------------------------------------------
# 4. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Land-Water Classification API",
    description=(
        "Coordinate-based land/water prediction, "
        "six-band satellite segmentation, "
        "and RGB image segmentation."
    ),
    version="2.0.0",
)


# --------------------------------------------------
# 5. COORDINATE INPUT
# --------------------------------------------------

class CoordinateInput(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


# --------------------------------------------------
# 6. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Land-Water Classification API is running",
        "available_tasks": [
            "Coordinate-based land/water prediction",
            "Six-band satellite GeoTIFF segmentation",
            "RGB JPG/PNG image segmentation",
        ],
        "endpoints": {
            "coordinates": "/predict/coordinates",
            "six_band_segmentation": "/predict/segmentation",
            "rgb_segmentation": "/predict/segmentation/rgb",
            "documentation": "/docs",
        },
    }


# --------------------------------------------------
# 7. COORDINATE-BASED PREDICTION
# --------------------------------------------------

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
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )


# --------------------------------------------------
# 8. SIX-BAND SATELLITE IMAGE SEGMENTATION
# --------------------------------------------------

@app.post("/predict/segmentation")
async def predict_segmentation(
    file: UploadFile = File(...)
):

    filename = (file.filename or "").lower()

    if not filename.endswith((".tif", ".tiff")):
        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a six-band Sentinel-2 "
                "GeoTIFF image (.tif or .tiff)."
            ),
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # Read GeoTIFF from memory
        with rasterio.MemoryFile(contents) as memory_file:
            with memory_file.open() as src:
                image = src.read()
                height = src.height
                width = src.width

        # Six-band model expects [6, H, W]
        if image.ndim != 3 or image.shape[0] != 6:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Expected exactly 6 bands, "
                    f"but received shape {image.shape}."
                ),
            )

        # Match six-band training preprocessing
        image = image.astype(np.float32) / 10000.0

        full_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        patch_size = 256

        with torch.no_grad():

            for row in range(0, height, patch_size):
                for col in range(0, width, patch_size):

                    row_end = min(row + patch_size, height)
                    col_end = min(col + patch_size, width)

                    actual_height = row_end - row
                    actual_width = col_end - col

                    patch = np.zeros(
                        (6, patch_size, patch_size),
                        dtype=np.float32,
                    )

                    patch[
                        :, :actual_height, :actual_width
                    ] = image[
                        :, row:row_end, col:col_end
                    ]

                    tensor = torch.from_numpy(
                        patch
                    ).unsqueeze(0).to(DEVICE)

                    logits = unet_model(tensor)

                    predicted_patch = (
                        torch.argmax(logits, dim=1)[0]
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                    full_mask[
                        row:row_end, col:col_end
                    ] = predicted_patch[
                        :actual_height, :actual_width
                    ]

        # Black = non-water; white = water
        mask_image = Image.fromarray(full_mask * 255)

        output = io.BytesIO()
        mask_image.save(output, format="PNG")
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="image/png",
            headers={
                "Content-Disposition":
                    'inline; filename="water_mask_6band.png"'
            },
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Six-band segmentation failed: {str(error)}",
        )

    finally:
        await file.close()


# --------------------------------------------------
# 9. RGB JPG/PNG IMAGE SEGMENTATION
# --------------------------------------------------

@app.post("/predict/segmentation/rgb")
async def predict_rgb_segmentation(
    file: UploadFile = File(...)
):

    filename = (file.filename or "").lower()

    if not filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG, or PNG image.",
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # Open the image and convert it to RGB
        with Image.open(io.BytesIO(contents)) as img:
            image = np.array(img.convert("RGB"))

        height, width, channels = image.shape

        # The RGB model was trained on values divided by 10000.
        # JPG/PNG pixel values are 0..255, so this puts them
        # into the 0..1 range. Data-domain differences remain.
        image = image.astype(np.float32) / 255.0

        full_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        patch_size = 256

        with torch.no_grad():

            for row in range(0, height, patch_size):
                for col in range(0, width, patch_size):

                    row_end = min(row + patch_size, height)
                    col_end = min(col + patch_size, width)

                    actual_height = row_end - row
                    actual_width = col_end - col

                    # Patch format: [256, 256, 3]
                    patch = np.zeros(
                        (patch_size, patch_size, 3),
                        dtype=np.float32,
                    )

                    patch[
                        :actual_height, :actual_width, :
                    ] = image[
                        row:row_end, col:col_end, :
                    ]

                    # HWC -> CHW, then add batch dimension.
                    # Model input shape: [1, 3, 256, 256]
                    patch_chw = patch.transpose(2, 0, 1).copy()

                    tensor = torch.from_numpy(
                        patch_chw
                    ).unsqueeze(0).to(DEVICE)

                    logits = rgb_model(tensor)

                    predicted_patch = (
                        torch.argmax(logits, dim=1)[0]
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )

                    # Copy only the unpadded region
                    full_mask[
                        row:row_end, col:col_end
                    ] = predicted_patch[
                        :actual_height, :actual_width
                    ]

        # Black = non-water; white = water
        mask_image = Image.fromarray(full_mask * 255)

        output = io.BytesIO()
        mask_image.save(output, format="PNG")
        output.seek(0)

        return StreamingResponse(
            output,
            media_type="image/png",
            headers={
                "Content-Disposition":
                    'inline; filename="rgb_water_mask.png"'
            },
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"RGB segmentation failed: {str(error)}",
        )

    finally:
        await file.close()
