from mlxlite import load_model
import mlx.core as mx
import tensorflow as tf

model = load_model("./FFNet-78S.tflite")
g = model.get_subgraph()
g.init_arrays()
res = g(mx.ones((1, 1024, 2048, 3)))

tf_model = tf.lite.Interpreter(model_path="./FFNet-78S.tflite")
tf_model.allocate_tensors()
tf_model.set_tensor(tf_model.get_input_details()[0]['index'], tf.ones((1, 1024, 2048, 3)))
tf_model.invoke()
tf_res = tf_model.get_tensor(tf_model.get_output_details()[0]['index'])

print(mx.allclose(res[0], mx.array(tf_res), atol=1e-3, rtol=1e-3))