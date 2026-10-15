
import requests
import streamlit as st
import folium

from streamlit_folium import st_folium


# ----------------------------------
# 1. Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Land-Water Classification",
    page_icon="🌍",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"


# ----------------------------------
# 2. Page Title
# ----------------------------------
st.title("🌍 Land-Water Classification")
st.write(
    "Enter coordinates manually or select a location "
    "on the map to predict Land or Water."
)


# ----------------------------------
# 3. Choose Input Method
# ----------------------------------
input_method = st.radio(
    "Choose how to select a location:",
    [
        "Enter Coordinates Manually",
        "Select Location on Map"
    ],
    horizontal=True
)


# ----------------------------------
# 4. Manual Coordinate Input
# ----------------------------------
latitude = None
longitude = None

if input_method == "Enter Coordinates Manually":

    st.subheader("Enter Coordinates")

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


# ----------------------------------
# 5. Interactive Map
# ----------------------------------
else:

    st.subheader("Select a Location on the Map")

    # Default map center: Lahore
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
        key="land_water_map"
    )

    if map_result and map_result.get("last_clicked"):

        clicked = map_result["last_clicked"]

        st.session_state.selected_latitude = clicked["lat"]
        st.session_state.selected_longitude = clicked["lng"]

    latitude = st.session_state.selected_latitude
    longitude = st.session_state.selected_longitude

    st.write(f"**Selected Latitude:** {latitude:.6f}")
    st.write(f"**Selected Longitude:** {longitude:.6f}")


# ----------------------------------
# 6. Predict Button
# ----------------------------------
st.divider()

if st.button("Predict Land or Water", type="primary"):

    payload = {
        "latitude": float(latitude),
        "longitude": float(longitude)
    }

    try:
        with st.spinner("Predicting location..."):

            response = requests.post(
                f"{API_URL}/predict/coordinates",
                json=payload,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction completed!")

            st.subheader("Prediction Result")

            st.write(f"**Latitude:** {latitude:.6f}")
            st.write(f"**Longitude:** {longitude:.6f}")

            prediction = result.get("prediction", result)

            if isinstance(prediction, dict):
                st.json(prediction)
            else:
                st.markdown(f"### 🌍 {prediction}")

        else:
            st.error(
                f"Backend returned error {response.status_code}"
            )
            st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            "Backend se connection nahi ho raha. "
            "Check karo ke FastAPI server chal raha hai."
        )

    except requests.exceptions.Timeout:
        st.error("Backend response mein timeout ho gaya.")

    except Exception as error:
        st.error(f"Error: {error}")
