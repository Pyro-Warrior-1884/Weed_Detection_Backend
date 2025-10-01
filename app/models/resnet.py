"""
ResNet-50 (TensorFlow) with DeepWeeds preprocessing
"""
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from pathlib import Path
import numpy as np
from PIL import Image

MODEL_PATH = Path("models/resnet50.h5")

class_labels = [
    'ChineeApple', 'Lantana', 'Parkinsonia', 'Parthenium', 
    'PricklyAcacia', 'Rubbervine', 'SiamWeed', 'Snakeweed', 'Negatives'
]

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.trainable = False
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load ResNet model: {e}")

# =========================
# Preprocessing function
# =========================
def preprocess(img: Image.Image, target_size: int = 224):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((target_size, target_size))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # scales to [-1,1] and channel order
    return x

# =========================
# Prediction function
# =========================
def predict(img: Image.Image):
    """
    Predict DeepWeeds label and confidence for a single image.
    Returns (label_name, confidence)
    """
    if model is None:
        print("[ERROR] Cannot run prediction — model not loaded.")
        return None, None

    x = preprocess(img)
    with tf.device(device):
        preds = model(x, training=False)
        pred_idx = tf.argmax(preds, axis=1).numpy()[0]
        confidence = tf.reduce_max(tf.nn.softmax(preds, axis=1)).numpy()
        label = class_labels[pred_idx]

    return label, float(confidence)

if model is not None:
    predict.framework = "tensorflow"
    predict.input_size = 224
