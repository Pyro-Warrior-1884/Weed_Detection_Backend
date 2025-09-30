"""
ResNet-50 (PyTorch)
"""
import torch
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np

MODEL_PATH = Path("models/resnet50.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Try loading the model, log error if missing
model = None
if MODEL_PATH.exists():
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
else:
    print(f"[ERROR] ResNet model file not found at {MODEL_PATH.resolve()}")

def preprocess(image: Image.Image, input_size: int = 224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize
    ])
    return transform(image).unsqueeze(0)

def predict(image: Image.Image):
    """ResNet-50"""
    if model is None:
        print("[ERROR] Cannot run prediction — ResNet model not loaded.")
        return None, None

    tensor = preprocess(image).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_idx = torch.max(probs, dim=1)
        idx = int(top_idx.cpu().numpy()[0])
        confidence = float(top_prob.cpu().numpy()[0])
    return idx, confidence

if model is not None:
    predict.framework = "pytorch"
    predict.input_size = 224
