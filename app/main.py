import io
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .config import MAX_UPLOAD_SIZE, ACCEPTED_CONTENT_TYPES
from .schemas import ModelInfo, PredictionResponse
from .utils.registry import get_model_handler, list_available_models

app = FastAPI(title="ML Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    models = []
    for name in list_available_models():
        handler = get_model_handler(name)  # loads on demand
        models.append(ModelInfo(
            id=name,
            name=handler.__doc__ or name,
            framework=handler.framework,
            input_size=handler.input_size
        ))
    return models

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model_name: str = Form(...)):
    try:
        handler = get_model_handler(model_name)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid model_name")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    if file.content_type not in ACCEPTED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read image file")

    try:
        prediction, confidence = handler(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    return PredictionResponse(
        model=model_name,
        prediction=str(prediction),
        confidence=round(float(confidence), 4)
    )
