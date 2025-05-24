import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from mnist_classifier import SimpleMNISTClassifier

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")

st.title("🖌️ Draw a digit (0–9)")
st.markdown("Draw a digit below and click **Predict** to see what the model thinks.")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="#000000",  # Black ink
    stroke_width=15,
    stroke_color="#FFFFFF",  # White stroke
    background_color="#000000",  # Black background
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# When the user clicks the button
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Preprocess the drawn image
        from PIL import Image
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))  # Get 2D image
        img = img.resize((28, 28)).convert('L')  # Resize and convert to grayscale
        img_array = np.array(img)

        # Load model and predict
        model = SimpleMNISTClassifier()
        pred, probs = model.predict(img_array)

        st.success(f"Predicted Digit: {pred}")
        st.bar_chart(probs)
    else:
        st.warning("Please draw something first!")

