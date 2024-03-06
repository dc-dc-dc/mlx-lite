from mlxlite import load_model
import mlx.core as mx
from PIL import Image
import os
import numpy as np

model="./ResNet50.tflite"

img = os.getenv("IMG", "car.jpg")
img = np.array(Image.open(img).resize((224, 224))).astype(np.float32)
img = (img - 127.5) / 127.5
img = np.expand_dims(img, axis=0)
img = mx.array(img)

m = load_model(model)
sub = m.get_subgraph()
sub.init_arrays()
res = sub(img)
mxpred = mx.argmax(res[0]).item()

labels = []
with open("./imagenet_labels.txt", "r") as f:
    labels = [l.strip() for l in f.readlines()]

print(f"Prediction: {labels[mxpred]}")