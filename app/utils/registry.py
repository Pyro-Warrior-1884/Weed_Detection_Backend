import importlib

# cache for loaded models
LOADED_MODELS = {}

# list of available models (just names, no heavy imports yet)
AVAILABLE_MODELS = ["resnet", "efficientnet", "unet", "model4"]

def get_model_handler(model_name: str):
    """Load and return the predict handler for a given model."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name in LOADED_MODELS:
        return LOADED_MODELS[model_name]

    module = importlib.import_module(f"app.models.{model_name}")
    handler = getattr(module, "predict")

    LOADED_MODELS[model_name] = handler
    return handler

def list_available_models():
    """Return the list of model names (lightweight)."""
    return AVAILABLE_MODELS
