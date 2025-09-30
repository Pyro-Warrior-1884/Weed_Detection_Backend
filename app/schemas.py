from pydantic import BaseModel

class ModelInfo(BaseModel):
    id: str
    name: str
    framework: str
    input_size: int

class PredictionResponse(BaseModel):
    model: str
    prediction: str
    confidence: float
