# -*- coding: utf-8 -*-
# mnist_classifier.py

import joblib
import numpy as np

# Load trained model
model = joblib.load("mnist_model.pkl")

def predict_digit(image):
    """Preprocess and predict digit from a 28x28 image (numpy array)."""
    # Normalize and flatten the image
    image = image.flatten().reshape(1, -1) / 255.0
    prediction = model.predict(image)[0]
    return prediction

