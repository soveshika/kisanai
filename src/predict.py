print("Script started")

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'kisanai_model.h5')
CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'models', 'class_indices.json')

IMG_SIZE = 224

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
    return model, index_to_class

def predict(image_path):
    model, index_to_class = load_model()
    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index] * 100
    disease_name = index_to_class[predicted_index]
    return disease_name, confidence

if __name__ == '__main__':
    import sys
    image_path = sys.argv[1]
    disease, confidence = predict(image_path)
    print(f"Disease: {disease}")
    print(f"Confidence: {confidence:.2f}%")