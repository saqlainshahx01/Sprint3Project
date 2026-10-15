
import io
import requests
import numpy as np
import streamlit as st
import folium

from PIL import Image
from streamlit_folium import st_folium


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Land-Water Classification System",
    page_icon="🌍",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"


# ==========================================
# 2. HEADER
# ==========================================
st.title("🌍 Land-Water Classification System")

st.write(
    "This application supports coordinate-based "
    "classification and satellite image segmentation."
)


# ==========================================
# 3. MAIN NAVIGATION
# ==========================================
part = st.sidebar.radio(
    "Choose Project Part",
    [
        "Part 1: Coordinate Classification",
        "Part 2: Image Segmentation"
    ]
)


# ==========================================
# PART 1: COORDINATE CLASSIFICATION
# ==========================================
if part == "Part 1: Coordinate Classification":

    st.header("Part 1 — Land-Water Classification")

    input_method = st.radio(
        "Choose location input method:",
        [
            "Manual Coordinates",
            "Select on Map"
        ],
        horizontal=True
    )

    latitude = None
    longitude = None

    if input_method == "Manual Coordinates":

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=31.5204,
                step=0.0001,
                format="%.6f"
            )

        with col2:
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=74.3587,
                step=0.0001,
                format="%.6f"
            )

    else:

        if "selected_latitude" not in st.session_state:
            st.session_state.selected_latitude = 31.5204

        if "selected_longitude" not in st.session_state:
            st.session_state.selected_longitude = 74.3587

        selected_lat = st.session_state.selected_latitude
        selected_lon = st.session_state.selected_longitude

        map_object = folium.Map(
            location=[selected_lat, selected_lon],
            zoom_start=5
        )

        folium.Marker(
            [selected_lat, selected_lon],
            tooltip="Selected location"
        ).add_to(map_object)

        map_result = st_folium(
            map_object,
            height=450,
            width=None,
            key="classification_map"
        )

        if map_result and map_result.get("last_clicked"):

            clicked = map_result["last_clicked"]

            new_lat = clicked["lat"]
            new_lon = clicked["lng"]

            if (
                new_lat != st.session_state.selected_latitude
                or new_lon != st.session_state.selected_longitude
            ):
                st.session_state.selected_latitude = new_lat
                st.session_state.selected_longitude = new_lon
                st.rerun()

        latitude = st.session_state.selected_latitude
        longitude = st.session_state.selected_longitude

        st.write(f"**Latitude:** {latitude:.6f}")
        st.write(f"**Longitude:** {longitude:.6f}")

    if st.button(
        "Predict Land or Water",
        type="primary",
        key="coordinate_predict"
    ):

        try:
            with st.spinner("Classifying selected location..."):

                response = requests.post(
                    f"{API_URL}/predict/coordinates",
                    json={
                        "latitude": float(latitude),
                        "longitude": float(longitude)
                    },
                    timeout=30
                )

            if response.ok:

                result = response.json()
                prediction = result.get("prediction", result)

                st.subheader("Classification Result")

                if isinstance(prediction, str):

                    if prediction.strip().lower() == "water":
                        st.info("🌊 Prediction: Water")
                    elif prediction.strip().lower() == "land":
                        st.success("🌍 Prediction: Land")
                    else:
                        st.write(f"Prediction: {prediction}")

                else:
                    st.json(prediction)

                st.caption(
                    f"Coordinates: {latitude:.6f}, {longitude:.6f}"
                )

            else:
                st.error(f"API error ({response.status_code})")
                st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error(
                "Backend connection failed. "
                "Make sure FastAPI is running on port 8000."
            )

        except requests.exceptions.Timeout:
            st.error("The backend request timed out.")

        except Exception as error:
            st.error(f"Error: {error}")


# ==========================================
# PART 2: IMAGE SEGMENTATION
# ==========================================
else:

    st.header("Part 2 — Land-Water Image Segmentation")

    model_choice = st.selectbox(
        "Choose U-Net Model",
        [
            "Improved RGB U-Net",
            "Six-Band U-Net"
        ]
    )

    if model_choice == "Improved RGB U-Net":

        endpoint = f"{API_URL}/predict/segmentation/rgb"

        uploaded_file = st.file_uploader(
            "Upload an RGB image",
            type=["jpg", "jpeg", "png"],
            key="rgb_upload"
        )

        st.caption(
            "Accepted: JPG, JPEG, PNG. "
            "The model was trained on RGB Sentinel-2 satellite patches."
        )

    else:

        endpoint = f"{API_URL}/predict/segmentation"

        uploaded_file = st.file_uploader(
            "Upload a six-band Sentinel-2 GeoTIFF",
            type=["tif", "tiff"],
            key="six_band_upload"
        )

        st.caption(
            "Accepted: six-band GeoTIFF containing "
            "B2, B3, B4, B8, B11 and B12 in the expected order."
        )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        st.subheader("Input Image")

        if model_choice == "Improved RGB U-Net":

            try:
                original_image = Image.open(
                    io.BytesIO(file_bytes)
                ).convert("RGB")

                st.image(
                    original_image,
                    caption="Uploaded RGB image",
                    use_container_width=True
                )

            except Exception as error:
                st.error(f"Could not read image: {error}")

        else:
            st.write(f"File: {uploaded_file.name}")
            st.write(
                f"Size: {len(file_bytes) / (1024 * 1024):.2f} MB"
            )

        if st.button(
            "Generate Segmentation Mask",
            type="primary",
            key="segmentation_predict"
        ):

            try:
                with st.spinner(
                    "U-Net is processing the image..."
                ):

                    response = requests.post(
                        endpoint,
                        files={
                            "file": (
                                uploaded_file.name,
                                file_bytes,
                                uploaded_file.type
                                or "application/octet-stream"
                            )
                        },
                        timeout=300
                    )

                if response.ok:

                    mask_image = Image.open(
                        io.BytesIO(response.content)
                    ).convert("L")

                    mask = np.array(mask_image)

                    water_pixels = np.count_nonzero(mask > 127)
                    total_pixels = mask.size
                    land_pixels = total_pixels - water_pixels

                    water_percentage = (
                        water_pixels / total_pixels
                    ) * 100

                    land_percentage = (
                        land_pixels / total_pixels
                    ) * 100

                    st.success("Segmentation completed!")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Predicted Mask")

                        st.image(
                            mask_image,
                            caption=(
                                "White = Water | "
                                "Black = Land/Non-water"
                            ),
                            use_container_width=True
                        )

                    with col2:
                        st.subheader("Mask Statistics")

                        st.metric(
                            "Predicted Water",
                            f"{water_percentage:.2f}%"
                        )

                        st.metric(
                            "Predicted Land / Non-water",
                            f"{land_percentage:.2f}%"
                        )

                    st.caption(
                        "Percentages are calculated from the predicted "
                        "mask; they are not ground-truth accuracy metrics."
                    )

                    st.download_button(
                        "Download Segmentation Mask",
                        data=response.content,
                        file_name="water_mask.png",
                        mime="image/png"
                    )

                else:
                    st.error(
                        f"Segmentation API error: {response.status_code}"
                    )
                    st.code(response.text)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Backend connection failed. "
                    "Make sure FastAPI is running on port 8000."
                )

            except requests.exceptions.Timeout:
                st.error(
                    "Segmentation timed out. Try a smaller image."
                )

            except Exception as error:
                st.error(f"Error: {error}")
