from app.models import resnet, efficientnet, unet, model4

MODEL_REGISTRY = {
    "resnet": resnet.predict,
    "efficientnet": efficientnet.predict,
    "unet": unet.predict,
    "model4": model4.predict,
}
