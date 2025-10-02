import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import traceback
import os

MODEL_PATH = Path("models/unet.h5")
IMG_SIZE = 128
model = None

try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file does not exist: {MODEL_PATH}")
    print(f"[INFO] Loading U-Net model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[SUCCESS] Model loaded successfully.")
except (OSError, IOError, ValueError, FileNotFoundError) as e:
    print(f"[ERROR] Failed to load model: {e}")
    traceback.print_exc()

def preprocess(image: Image.Image, input_size: int = IMG_SIZE):
    try:
        image = image.convert("RGB")
        image = image.resize((input_size, input_size))
        arr = np.array(image).astype(np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        traceback.print_exc()
        raise

def weed_status(segmentation_map: np.ndarray, threshold: float = 0.01):
    try:
        fraction = np.mean(segmentation_map)
        return "Weeds" if fraction >= threshold else "No Weeds"
    except Exception as e:
        traceback.print_exc()
        raise

def predict(image: Image.Image):
    try:
        if model is None:
            raise RuntimeError("U-Net model is not loaded.")
        arr = preprocess(image, input_size=IMG_SIZE)
        preds = model.predict(arr)
        if preds.ndim != 4 or preds.shape[-1] != 1:
            raise ValueError(f"Unexpected prediction shape: {preds.shape}")
        preds = preds[0]
        segmentation_map = (preds[..., 0] > 0.5).astype(np.uint8)
        confidence = float(np.mean(preds[..., 0]))
        status = weed_status(segmentation_map)
        return status, confidence
    except Exception as e:
        traceback.print_exc()
        raise

predict.framework = "tensorflow"
predict.input_size = IMG_SIZE
