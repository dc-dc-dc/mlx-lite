import tensorflow as tf
import numpy as np
import mlx.core as mx
from mlxlite import load_model
import keras
import argparse
import os

from typing import List, Tuple

np.random.seed(0)

def generate_op_file(op: str, input_shapes: List[Tuple[int]]) -> str:
    if getattr(keras.layers, op, None) is None:
        print(f"Invalid operation {op}")
        exit(1)

    in_tensors = [keras.layers.Input(shape=x[1:], batch_size=x[0]) for x in input_shapes]
    out = getattr(keras.layers, op)()(
        in_tensors[0] if len(in_tensors) == 1 else in_tensors
    )
    model = keras.models.Model(inputs=in_tensors, outputs=out)
    out = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    filename = f"./op-{op}.tflite"  
    with open(filename, "wb") as f:
        f.write(out)
    return filename

def compare_tf(path: str, shapes: List[Tuple[int]]):
    np_ins = [np.random.random(shape).astype(np.float32) for shape in shapes]
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    for t, d in zip(interpreter.get_input_details(), np_ins):
        interpreter.set_tensor(t["index"], tf.convert_to_tensor(d))
    interpreter.invoke()
    tf_out = [interpreter.get_tensor(t["index"]) for t in interpreter.get_output_details()]
    model = load_model(path)
    s = model.get_subgraph()
    s.init_arrays()
    mxout = s([mx.array(d) for d in np_ins])
    for t, m in zip(tf_out, mxout):
        np.testing.assert_allclose(t, np.array(m), rtol=1e-5, atol=1e-5)
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--input",
        help="shape of the input tensor ex: 224x224x3",
        type=str,
        default=[],
        action="append",
        required=True,
    )
    args.add_argument("op", help="operation to perform", type=str)

    args = args.parse_args()
    shapes = [tuple(map(int, i.split("x"))) for i in args.input]
    filename = generate_op_file(args.op, shapes)
    compare_tf(filename, shapes)

    os.remove(filename)
