from mlxlite.load import load_model
import mlx.core as mx

model="./experiments/ResNet50.tflite"

m = load_model(model)
sub = m.get_subgraph()
sub.init_arrays()
res = sub(mx.ones((1, 224, 224, 3)))
print(res[0].shape)