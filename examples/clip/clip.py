from mlxlite import load_model
import mlx.core as mx
import tensorflow as tf
import numpy as np

encoder = load_model("./CLIPImageEncoder.tflite")

tfm_encoder = tf.lite.Interpreter(model_path="./CLIPImageEncoder.tflite")
tfm_encoder.allocate_tensors()
tfm_encoder.set_tensor(tfm_encoder.get_input_details()[0]['index'], tf.convert_to_tensor(np.ones((1, 224, 224, 3), dtype=np.float32)))
tfm_encoder.invoke()

textEncoder = load_model("./CLIPTextEncoder.tflite")

encoderGraph = encoder.get_subgraph()
textGraph = textEncoder.get_subgraph()

encoderGraph.init_arrays()
encoderres = encoderGraph(mx.ones((1, 224, 224, 3)))

textGraph.init_arrays()
text_res = textGraph(mx.ones((1, 77), dtype=mx.int32))

tfm_text_encoder = tf.lite.Interpreter(model_path="./CLIPTextEncoder.tflite")
tfm_text_encoder.allocate_tensors()
tfm_text_encoder.set_tensor(tfm_text_encoder.get_input_details()[0]['index'], tf.convert_to_tensor(np.ones((1, 77), dtype=np.int32)))
tfm_text_encoder.invoke()

np.testing.assert_allclose(np.array(text_res[0]), tfm_text_encoder.get_tensor(tfm_text_encoder.get_output_details()[0]["index"]), atol=1e-4)
np.testing.assert_allclose(np.array(encoderres[0]), tfm_encoder.get_tensor(tfm_encoder.get_output_details()[0]["index"]), atol=1e-4)