import os

from .model import MXModel
from .generated.tflite.Model import Model

DEBUG = os.getenv("DEBUG", "0") == "1"

def load_model(path: str) -> MXModel:
    with open(path, "rb") as f:
        return MXModel(Model.GetRootAs(f.read(), 0))
