import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from mnist_classifier import SimpleMNISTClassifier

st.title("MNIST Digit Classifier")

model = SimpleMNISTClassifier()

# Draw input area
canvas = st.canvas(height=280, width=280, stroke_width=15, stroke_color="#FFFFFF", background_color="#000000")

if canvas.image_data is not None:
    # Convert to grayscale and resize to 28x28
    img = Image.fromarray(canvas.image_data.astype('uint8'), 'RGBA').convert('L')
    img = ImageOps.invert(img)  # invert colors for MNIST style: white digit on black bg
    img = img.resize((28, 28))

    st.image(img, caption="Processed Input", width=100)

    img_array = np.array(img)

    digit, probs = model.predict(img_array)
    st.write(f"Predicted digit: {digit}")
    st.bar_chart(probs)
