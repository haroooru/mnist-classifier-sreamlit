import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from mnist_classifier import predict_digit

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

st.title("🧠 MNIST Handwritten Digit Classifier")

st.markdown(
    """
    Draw a digit (0-9) in the box below. 
    Your drawing will be classified using a simple neural net built from scratch (no TensorFlow or PyTorch!).
    """
)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=15,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype(np.uint8))

    # Process the image only if the user draws something
    if np.max(img) > 0:
        img_resized = img.resize((28, 28)).convert("L")  # Grayscale
        img_array = np.asarray(img_resized).astype(np.float32) / 255.0
        img_flat = img_array.flatten().reshape(1, -1)

        pred = predict_digit(img_flat)

        st.image(img_resized, caption="Model Input (28x28 Grayscale)", width=150)
        st.markdown(f"### Predicted Digit: `{pred}`")
    else:
        st.info("Draw something in the canvas above to get a prediction!")
else:
    st.info("Waiting for your drawing...")

