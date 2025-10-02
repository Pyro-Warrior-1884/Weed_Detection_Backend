import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input
from pathlib import Path
from PIL import Image

MODEL_PATH = Path("models/efficientnetb0.h5")

class_names = [
    "Chinee apple", "Lantana", "Parkinsonia", "Parthenium",
    "Prickly acacia", "Rubber vine", "Siam weed",
    "Snake weed", "Negatives"
]

try:
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,   
        input_shape=(224, 224, 3),
        pooling="avg"
    )

    x = base_model.output
    x = Dropout(0.3)(x) 
    outputs = Dense(len(class_names), activation="softmax")(x)
    print(f"[SUCCESS] EfficientNet loaded successfully.")
    model = Model(inputs=base_model.input, outputs=outputs)
    model.load_weights(MODEL_PATH, by_name=True)

except Exception as e:
    model = None
    print(f"[ERROR] Could not build/load model: {e}")

def preprocess(image: Image.Image, input_size: int = 224):
    image = image.convert("RGB")
    image = image.resize((input_size, input_size))
    arr = np.array(image).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def predict(image: Image.Image):
    if model is None:
        raise RuntimeError("Model not loaded!")

    arr = preprocess(image)
    preds = model.predict(arr, verbose=0)
    top_idx = int(np.argmax(preds[0]))
    confidence = float(np.max(preds[0]))
    label = class_names[top_idx] if top_idx < len(class_names) else str(top_idx)
    return label, confidence
