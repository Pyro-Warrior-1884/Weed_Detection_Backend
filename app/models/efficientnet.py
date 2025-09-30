import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import load_model
from pathlib import Path
from PIL import Image

MODEL_PATH = Path("models/efficientnetb0.h5")

model = None
try:
    try:
        model = load_model(MODEL_PATH)
    except Exception:
        model = EfficientNetB0(weights=None, input_shape=(224, 224, 3), include_top=True)
        model.load_weights(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] EfficientNet model could not be loaded: {e}")

def preprocess(image: Image.Image, input_size: int = 224):
    arr = image.resize((input_size, input_size))
    arr = np.array(arr).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

def predict(image: Image.Image):
    if model is None:
        print("[ERROR] Cannot run prediction — EfficientNet model not loaded.")
        return None, None
    arr = preprocess(image)
    preds = model.predict(arr)
    top_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    return top_idx, confidence

if model is not None:
    predict.framework = "tensorflow"
    predict.input_size = 224
