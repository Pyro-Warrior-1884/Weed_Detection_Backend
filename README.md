# Weed Detection Backend

An intelligent **Weed Detection Backend** built using **FastAPI**, **TensorFlow**, and **PyTorch**, designed to classify and segment images for automated weed identification.  
This repository serves as the backend API powering the weed detection system, capable of handling image uploads, preprocessing, and model inference across multiple deep learning architectures.

---

## Features

- Supports multiple model architectures: **U-Net**, **ResNet50**, **EfficientNetB0**, and **custom TensorFlow models**  
- Fast inference with **FastAPI** and **Uvicorn**  
- Handles image uploads and validation (JPEG/PNG)  
- Configurable model registry and environment  
- Built-in Pydantic schemas for request/response validation  

---

## Tech Stack

| Category | Technologies |
|-----------|---------------|
| **Backend Framework** | FastAPI |
| **Deep Learning** | TensorFlow, PyTorch, TorchVision |
| **Data Handling** | Pillow, NumPy, Pydantic |
| **Server** | Uvicorn |
| **Environment** | Python 3.9+ |

---

## Project Structure

```
Weed_Detection_Backend/
│
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration (paths, constants, etc.)
│   ├── schemas.py              # Pydantic models for API validation
│   ├── utils/
│   │   ├── registry.py         # Handles model loading and registration
│   │   └── __init__.py
│   └── models/
│       ├── resnet.py           # ResNet50 model handler
│       ├── efficientnet.py     # EfficientNetB0 model handler
│       ├── unet.py             # U-Net segmentation model handler
│       └── model4.py           # Custom TensorFlow model
│
├── .github/
│   └── workflows/
│       └── main_weeddetection.yml   # Azure CI/CD pipeline
│
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation (you are here)
```

---

## Installation & Setup

### Clone the repository
```bash
git clone https://github.com/Pyro-Warrior-1884/Weed_Detection_Backend.git
cd Weed_Detection_Backend
```

### Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # For Linux/macOS
venv\Scripts\activate         # For Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the FastAPI server
```bash
uvicorn app.main:app --reload
```

### Access the API
Visit:
```
http://127.0.0.1:8000
```
---

## Model Overview

| Model | Framework | Input Size | Purpose |
|--------|------------|-------------|----------|
| **U-Net** | TensorFlow | 128×128 | Image segmentation & weed area detection |
| **ResNet50** | TensorFlow | 224×224 | Image classification |
| **EfficientNetB0** | TensorFlow | 224×224 | Lightweight classification |
| **Model4** | TensorFlow | 224×224 | Custom model template |

---

## API Schemas

### ModelInfo
```python
{
  "id": "string",
  "name": "ResNet50",
  "framework": "tensorflow",
  "input_size": 224
}
```

### PredictionResponse
```python
{
  "model": "unet",
  "prediction": "weed_detected",
  "confidence": 0.93
}
```

---

## Testing

You can test predictions using **cURL** or **Postman**:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@sample.jpg"
```

Expected response:
```json
{
  "model": "unet",
  "prediction": "weed_detected",
  "confidence": 0.94
}
```

---

## Configuration

Located in `app/config.py`

```python
BASE_DIR = "/path/to/project/base/dir"
MODEL_DIR = BASE_DIR + "/models"
MAX_UPLOAD_SIZE = 1024 * 1024  # 1 MB
ACCEPTED_CONTENT_TYPES = ["image/jpeg", "image/png"]
ALLOWED_ORIGINS = ["*"]
```

You can modify these constants to adjust:
- Upload limits  
- Accepted MIME types  
- Allowed CORS origins  

---

## Notes & Considerations

- Ensure model files (e.g., `unet.h5`, `resnet50.h5`) are present in the `models/` directory.  
- Use consistent input image sizes (224x224 or 128x128).  
- Fine-tune threshold values in segmentation for better weed detection accuracy.  

---

## Contributing

Contributions are welcome!  
To contribute:
1. Fork the repository  
2. Create a new branch (`feature/my-feature`)  
3. Commit your changes  
4. Submit a Pull Request  

---

## Author

**Pyro Warrior**  
GitHub: [@Pyro-Warrior-1884](https://github.com/Pyro-Warrior-1884)  
Reach out for collaborations or ML-based backend integrations.

---

### Don’t forget to star this repo if you find it useful!
