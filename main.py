import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Title of the app
st.title("Color to Grayscale and Back to Color using Autoencoder")

# Load pre-trained autoencoder model
@st.cache_resource
def load_autoencoder_model():
    model = load_model('autoencoder_model.keras')  # Ganti dengan nama file model autoencoder kamu
    return model

model = load_autoencoder_model()

# Allow the user to upload a color image
uploaded_file = st.file_uploader("Choose a color image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Convert the PIL image to a numpy array (RGB format)
    image_array = np.array(image)
    
    # Get the original dimensions of the uploaded image
    original_shape = image_array.shape[:2]  # Menyimpan tinggi dan lebar asli
    
    # Display the original color image with its original dimensions
    st.subheader("Original Color Image")
    st.image(image_array, caption="Original Color Image", use_column_width=False)
    
    # Convert the image to grayscale using OpenCV
    grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Display the grayscale image with original dimensions
    st.subheader("Grayscale Image")
    st.image(grayscale_image, caption="Grayscale Image", use_column_width=False, channels='GRAY')
    
    # Resize the grayscale image to the required size for the model input (e.g., 160x160)
    SIZE = 160  # Sesuaikan dengan ukuran input model autoencoder kamu
    grayscale_resized = cv2.resize(grayscale_image, (SIZE, SIZE))
    
    # Replicate grayscale image to have 3 channels for the model input
    grayscale_3channel = np.stack((grayscale_resized,)*3, axis=-1)  # Bentuknya sekarang (160, 160, 3)
    
    # Normalize the image to [0, 1] range and reshape to (1, SIZE, SIZE, 3) for model input
    grayscale_input = np.expand_dims(grayscale_3channel, axis=0)  # Bentuk (1, 160, 160, 3)
    grayscale_input = grayscale_input.astype('float32') / 255.0

    # Use the autoencoder model to predict the colorized image
    predicted_color_image = model.predict(grayscale_input)
    
    # Clip the predicted values to [0, 1] range
    predicted_color_image = np.clip(predicted_color_image, 0.0, 1.0)
    
    # Reshape the predicted image to (SIZE, SIZE, 3) for RGB output
    predicted_color_image_reshaped = predicted_color_image.reshape(SIZE, SIZE, 3)
    
    # Convert the predicted image to uint8 format (0-255) for display
    predicted_color_image_uint8 = (predicted_color_image_reshaped * 255).astype(np.uint8)
    
    # Resize the predicted color image back to the original dimensions
    predicted_color_image_resized = cv2.resize(predicted_color_image_uint8, original_shape[::-1])  # Balik tinggi dan lebar
    
    # Display the predicted colorized image with original dimensions
    st.subheader("Predicted Colorized Image")
    st.image(predicted_color_image_resized, caption="Colorized Image by Autoencoder", use_column_width=False)
    
    # Provide the option to download the colorized image
    st.subheader("Download Colorized Image")
    
    # Convert the predicted image to PIL format for download
    color_image_pil = Image.fromarray(predicted_color_image_resized)
    
    # Convert PIL image to byte array for download
    color_image_bytes = cv2.imencode('.png', predicted_color_image_resized)[1].tobytes()
    
    # Create a download button
    st.download_button(
        label="Download Colorized Image",
        data=color_image_bytes,
        file_name="colorized_image.png",
        mime="image/png"
    )
