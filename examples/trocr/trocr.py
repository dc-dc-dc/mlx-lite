from mlxlite import load_model
import mlx.core as mx
import numpy as np
import tensorflow as tf

encoder = load_model("./TrOCREncoder.tflite")
encgraph = encoder.get_subgraph()
encgraph.init_arrays()
decres = encgraph(mx.ones((1, 384, 384, 3)))

tfenc = tf.lite.Interpreter(model_path="./TrOCREncoder.tflite")
tfenc.allocate_tensors()
tfenc.set_tensor(tfenc.get_input_details()[0]['index'], tf.ones((1, 384, 384, 3)))
tfenc.invoke()
for i in range(len(tfenc.get_output_details())):
    tfres = tfenc.get_tensor(tfenc.get_output_details()[i]['index'])
    np.testing.assert_allclose(tfres, np.array(decres[i]), atol=1e-2, rtol=1e-3)


decoder = load_model("./TrOCRDecoder.tflite")
decgraph = decoder.get_subgraph()
decgraph.init_arrays()
decres = decgraph([mx.ones(x.shape, x.dtype) for x in decgraph.get_inputs()])

tfdec = tf.lite.Interpreter(model_path="./TrOCRDecoder.tflite")
tfdec.allocate_tensors()
for i in range(len(tfdec.get_input_details())):
    tfdec.set_tensor(tfdec.get_input_details()[i]['index'], tf.ones(tfdec.get_input_details()[i]['shape'], tfdec.get_input_details()[i]['dtype']))

tfdec.invoke()
for i in range(len(tfdec.get_output_details())):
    tfres = tfdec.get_tensor(tfdec.get_output_details()[i]['index'])
    np.testing.assert_allclose(tfres, np.array(decres[i]), atol=1e-2, rtol=1e-3)