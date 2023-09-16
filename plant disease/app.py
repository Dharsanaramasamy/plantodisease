# Import necessary libraries
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

# Set Streamlit page title and icon, layout, and sidebar state
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Define the Streamlit app title with a background image (you can uncomment this if you have a background image)
# st.markdown(
#     """
#     <style>
#     .title {
#         background-image: url("https://example.com/background.jpg");
#         background-size: cover;
#         text-align: center;
#         color: white;
#         padding: 20px;
#         font-size: 36px;
#         border-radius: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# Title
st.markdown("<h1 class='title'>Plant Disease Detection App</h1>", unsafe_allow_html=True)

# Load your pre-trained model (replace "model.h5" with the actual model file)
model = tf.keras.models.load_model("model2.h5")

# File uploader widget
uploaded_file = st.file_uploader("Choose a plant image (JPG format)", type=["jpg", "jpeg"])

# Dictionary mapping class indices to disease labels based on your model's predictions
map_dict = {
    0: 'Healthy',
    1: 'Powdery',
    2: 'Rust',
}

# Function to preprocess and classify the uploaded image
def classify_image(image_data):
    try:
        # Convert the file to an OpenCV image
        nparr = np.frombuffer(image_data.read(), np.uint8)
        opencv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(opencv_image, (64, 64))

        # Display the uploaded image
        st.image(opencv_image, use_column_width=True, caption="Uploaded Image")

        # Preprocess the image and make a prediction
        resized = mobilenet_v2_preprocess_input(resized)
        img_reshape = resized[np.newaxis, ...]
        prediction = model.predict(img_reshape).argmax()
        predicted_label = map_dict.get(prediction, "Unknown")

        st.title(f"Predicted Disease: {predicted_label}")
    except Exception as e:
        st.error(f"Error: {e}")

# Generate Prediction button if an image is uploaded
if uploaded_file is not None:
    if st.button("Generate Prediction"):
        classify_image(uploaded_file)
