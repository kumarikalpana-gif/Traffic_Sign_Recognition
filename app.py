
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from src.config import RECOMMENDED_MODEL_PATH
from src.preprocessing import prepare_image_for_prediction


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="centered"
)

st.title("🚦 Traffic Sign Recognition")
st.write(
    "Upload a traffic sign image and let the trained "
    "CNN model predict its class."
)


# --------------------------------------------------
# Traffic sign class names (GTSRB: 43 classes)
# --------------------------------------------------

CLASS_NAMES = {
    0: "Speed limit (20 km/h)",
    1: "Speed limit (30 km/h)",
    2: "Speed limit (50 km/h)",
    3: "Speed limit (60 km/h)",
    4: "Speed limit (70 km/h)",
    5: "Speed limit (80 km/h)",
    6: "End of speed limit (80 km/h)",
    7: "Speed limit (100 km/h)",
    8: "Speed limit (120 km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_trained_model():
    if not RECOMMENDED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {RECOMMENDED_MODEL_PATH}"
        )

    return tf.keras.models.load_model(RECOMMENDED_MODEL_PATH)


try:
    model = load_trained_model()
    st.success("Trained Baseline CNN loaded successfully.")
except Exception as error:
    st.error(f"Could not load model: {error}")
    st.stop()


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded image")
    st.image(image, caption="Input traffic sign", width=300)

    if st.button("Predict Traffic Sign", type="primary"):

        with st.spinner("Analyzing image..."):

            processed_image = prepare_image_for_prediction(image)

            predictions = model.predict(
                processed_image,
                verbose=0
            )

            top_3_indices = np.argsort(predictions[0])[-3:][::-1]

            predicted_class = int(top_3_indices[0])
            confidence = float(predictions[0][predicted_class]) * 100

            predicted_name = CLASS_NAMES[predicted_class]

        st.subheader("Prediction result")

        st.write("**Predicted class ID:**", predicted_class)
        st.write("**Traffic sign:**", predicted_name)

        st.metric(
            label="Model confidence",
            value=f"{confidence:.2f}%"
        )

        st.subheader("Top 3 predictions")

        for index in top_3_indices:
            class_id = int(index)
            probability = float(predictions[0][class_id]) * 100

            st.write(
                f"{CLASS_NAMES[class_id]} "
                f"(Class {class_id})"
            )

            st.progress(probability / 100)

            st.caption(f"{probability:.2f}%")

        st.caption(
            "Confidence is the model's predicted probability, "
            "not a guarantee that the sign was identified correctly."
        )

st.divider()

st.warning(
    "This application is an educational demonstration. "
    "Do not use its predictions for real-world driving decisions."
)
