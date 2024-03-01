import tflite
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import flatbuffers
from typing import Dict, List
import tensorflow as tf
import math, os
from PIL import Image

GPU = os.getenv("GPU", "0") == "1"
DEBUG = os.getenv("DEBUG", "0") == "1"
COMPARE = os.getenv("COMPARE", "0") == "1"

mx.set_default_device(mx.gpu if GPU else mx.cpu)


op_map = {
    getattr(tflite.BuiltinOperator, p): p
    for p in dir(tflite.BuiltinOperator)
    if not p.startswith("__")
}

tensor_type_map: Dict[int, np.dtype] = {
    tflite.TensorType.BOOL: np.bool_,
    tflite.TensorType.UINT8: np.uint8,
    tflite.TensorType.UINT16: np.uint16,
    tflite.TensorType.UINT32: np.uint32,
    tflite.TensorType.UINT64: np.uint64,
    tflite.TensorType.INT8: np.int8,
    tflite.TensorType.INT16: np.int16,
    tflite.TensorType.INT32: np.int32,
    tflite.TensorType.INT64: np.int64,
    tflite.TensorType.FLOAT16: np.float16,
    tflite.TensorType.FLOAT32: np.float32,
}

tensors = {-1: None}


def init_tensor(model: tflite.Model, tensor: tflite.Tensor):
    buffer = model.Buffers(tensor.Buffer())
    o = flatbuffers.number_types.UOffsetTFlags.py_type(buffer._tab.Offset(4))
    shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]
    np_type = np.dtype(tensor_type_map[tensor.Type()])
    if (qt := tensor.Quantization()) is not None:
        if not qt.ScaleIsNone(): 
            print(f"\tQUANTIZATION: max={qt.MaxAsNumpy()} min={qt.MinAsNumpy()} scale={qt.ScaleAsNumpy()} dim={qt.QuantizedDimension()} zero={qt.ZeroPointAsNumpy()}")
            exit(0)

    return mx.array(
        np.frombuffer(
            buffer._tab.Bytes,
            offset=buffer._tab.Vector(o),
            count=buffer._tab.VectorLen(o) // np_type.itemsize,
            dtype=np_type,
        )
        if o != 0
        else np.zeros(shape, dtype=np_type)
    ).reshape(shape)


# 0 = None, 1 = Relu
def handleFusedActivationFunction(a: mx.array, func: int):
    assert func in [0, 1], f"unsupported activation function={func}"

    if func == 1:
        return nn.relu(a)
    return a


def rsqrt(inputs: List[mx.array]):
    if DEBUG:
        print(f"\tRSQRT input={inputs[0].shape}")
    return mx.rsqrt(inputs[0])


def mean(inputs: List[mx.array]):
    if DEBUG:
        print(f"\tMEAN input={inputs[0].shape} axis={inputs[1].tolist()}")
    return mx.mean(inputs[0], axis=inputs[1].tolist())

def conv2d(inputs: List[mx.array], options: tflite.Conv2DOptions):
    temp = inputs[0]
    filter = inputs[1]
    expanded = False
    assert options.DilationHFactor() == 1
    assert options.DilationWFactor() == 1
    if inputs[0].ndim != 4:
        expanded = True
        temp = mx.expand_dims(temp, axis=0)
    padding = [0, 0]
    if options.Padding() == 0:
        _is = temp.shape
        _fs = filter.shape
        padding = [0, 0]
        if (off := _is[1] % _fs[1]) != 0:
            padding[0] = math.ceil(off / 2)
        if (off := _is[2] % _fs[1]) != 0:
            padding[1] = math.ceil(off / 2)
        # if previous doesn't work, try this
        if padding == [0, 0] and (options.StrideH() == 1 and options.StrideW() == 1):
            padding[0] = math.ceil((_fs[1] - 1) / 2)
            padding[1] = math.ceil((_fs[2] - 1) / 2)

    if DEBUG:
        print(
            f"\tCONV_2D: in={temp.shape} filter={filter.shape} expanded={expanded} stride=({options.StrideH()}, {options.StrideW()}) padding={padding} dilation={options.DilationHFactor(), options.DilationWFactor()} func={options.FusedActivationFunction()}"
        )
    temp = mx.conv2d(
        temp, filter, stride=(options.StrideH(), options.StrideW()), padding=padding
    )
    if len(inputs) == 3:
        temp = mx.add(temp, inputs[2])

    return handleFusedActivationFunction(temp, options.FusedActivationFunction())


def pad(inputs: List[mx.array], options: tflite.PadOptions):
    if DEBUG:
        print(f"\tPAD: in={inputs[0].shape}, pad={inputs[1].tolist()}")
    return mx.pad(inputs[0], inputs[1].tolist())


def max_pool2d(inputs: List[mx.array], options: tflite.Pool2DOptions):
    # TODO: Support padding
    if DEBUG:
        print(
            f"\tMAX_POOL_2D: in={inputs[0].shape}, pool=({options.FilterHeight()}, {options.FilterWidth()}) stride=({options.StrideH()},{options.StrideW()}) padding={options.Padding()}"
        )
    temp = nn.MaxPool2d(
        (options.FilterHeight(), options.FilterWidth()),
        stride=(options.StrideH(), options.StrideW()),
    )(inputs[0])
    return handleFusedActivationFunction(temp, options.FusedActivationFunction())


def add(inputs: List[mx.array], options: tflite.AddOptions):
    assert len(inputs) == 2
    x, y = inputs[0], inputs[1]
    if DEBUG:
        print(f"\tADD: in={x.shape} {y.shape} func={options.FusedActivationFunction()}")
    return handleFusedActivationFunction(x + y, options.FusedActivationFunction())


def fully_connected(inputs: List[mx.array], options: tflite.FullyConnectedOptions):
    if DEBUG:
        print(
            f"\tFULLY_CONNECTED: inputs={len(inputs)} format={options.WeightsFormat()} keepdims={options.KeepNumDims()} func={options.FusedActivationFunction()}"
        )
    if len(inputs) == 3 and isinstance(inputs[2], mx.array):
        temp = mx.addmm(inputs[2], inputs[0], inputs[1].T)
    else:
        temp = inputs[0] @ inputs[1].T
    return handleFusedActivationFunction(temp, options.FusedActivationFunction())


def reshape(ins: List[mx.array], options: tflite.ReshapeOptions):
    if DEBUG:
        print(f"\tRESHAPE: in={ins[0].shape} shape={ins[1].tolist()}")
    return mx.reshape(ins[0], ins[1].tolist())


def concat(ins: List[mx.array], options: tflite.ConcatenationOptions):
    if DEBUG:
        print(f"\tCONCAT: in={len(ins)} axis={options.Axis()}")
    return mx.concatenate(ins, axis=options.Axis())


def squared_difference(ins: List[mx.array], options: tflite.SquaredDifferenceOptions):
    assert len(ins) == 2
    x, y = ins[0], ins[1] 
    if DEBUG:
        print(f"\tSQUARED_DIFFERENCE: in={x.shape} {y.shape}")
    return mx.square(x - y)


def mul(inputs: List[mx.array], options: tflite.MulOptions):
    assert len(inputs) == 2
    x, y = inputs[0], inputs[1]
    if DEBUG:
        print(f"\tMUL: in={x.shape} {y.shape}")
    return handleFusedActivationFunction(x * y, options.FusedActivationFunction())


def sub(inputs: List[mx.array], options: tflite.SubOptions):
    assert len(inputs) == 2
    x, y = inputs[0], inputs[1]
    if DEBUG:
        print(f"\tSUB: in={x.shape} {y.shape}")
    return handleFusedActivationFunction(x - y, options.FusedActivationFunction())

def strided_slice(ins: List[mx.array], options: tflite.StridedSliceOptions):
    if DEBUG:
        print(f"\tSTRIDED_SLICE: in={ins[0].shape} begin={ins[1].tolist()} end={ins[2].tolist()} strides={ins[3].tolist()} begin_mask={options.BeginMask()} end_mask={options.EndMask()} ellipsis_mask={options.EllipsisMask()} new_axis={options.NewAxisMask()} shrink_axis={options.ShrinkAxisMask()} ")
    assert options.EllipsisMask() == 0, "Ellipsis not supported"
    assert options.NewAxisMask() == 0, "NewAxis not supported"
    assert options.ShrinkAxisMask() == 0, "ShrinkAxis not supported"
    slices = [slice(0, d) for d in ins[0].shape]
    i = 0
    shape = ins[0].shape
    begin_mask = options.BeginMask()
    end_mask = options.EndMask()
    for b, e, s in zip(ins[1].tolist(), ins[2].tolist(), ins[3].tolist()):
        if (begin_mask >> i) & 1 == 1:
            b = 0
        if (end_mask >> i) & 1 == 1:
            e = shape[i]
        slices[i] = slice(b, e, s)
        i += 1
    return ins[0][tuple(slices)]

def transpose(ins: List[mx.array], options: tflite.TransposeOptions):
    assert len(ins) == 2
    if DEBUG:
        print(f"\tTRANSPOSE: in={ins[0].shape} order={ins[1].shape}")
    return mx.transpose(ins[0], ins[1].tolist())

def batch_matmul(ins: List[mx.array], options: tflite.BatchMatMulOptions):
    assert len(ins) == 2
    if DEBUG:
        print(f"\tBATCH_MATMUL: x={ins[0].shape} y={ins[1].shape} adj_x={options.AdjX()} adj_y={options.AdjY()} asymmetric={options.AsymmetricQuantizeInputs()}")
    assert not options.AdjX(), "AdjX not supported"
    assert not options.AdjY(), "AdjY not supported"
    assert not options.AsymmetricQuantizeInputs(), "AsymmetricQuantizeInputs not supported"
    return mx.matmul(ins[0], ins[1])

def softmax(ins: List[mx.array], options: tflite.SoftmaxOptions):
    assert len(ins) == 1
    assert options.Beta() == 1.0, "Beta not supported"
    if DEBUG:
        print(f"\tSOFTMAX: in={ins[0].shape} beta={options.Beta()}")
    return mx.softmax(ins[0], axis=-1)

def gelu(ins: List[mx.array], options: tflite.GeluOptions):
    assert len(ins) == 1
    if DEBUG:
        print(f"\tGELU: len={len(ins)}")
    return nn.gelu_approx(ins[0]) if options.Approximate() else nn.gelu(ins[0]) 

def resize_nearest_neighbor(ins: List[mx.array], options: tflite.ResizeNearestNeighborOptions):
    assert len(ins) == 2
    if DEBUG:
        print(f"\tRESIZE_NEAREST_NEIGHBOR: in={ins[0].shape} new_shape={ins[1].tolist()} align_corners={options.AlignCorners()} half_pixel_centers={options.HalfPixelCenters()}")

    scale_factor = [n // i for i, n in zip(ins[0].shape[1:-1], ins[1].tolist())]
    return nn.Upsample(scale_factor=scale_factor, mode="nearest")(ins[0])

def leaky_relu(ins: List[mx.array], options: tflite.LeakyReluOptions):
    if DEBUG:
        print(f"\tLEAKY_RELU: in={ins[0].shape} alpha={options.Alpha()}")
    return nn.leaky_relu(ins[0], options.Alpha())

def tile(ins: List[mx.array], options: tflite.TileOptions):
    assert len(ins) == 2
    if DEBUG:
        print(f"\tTILE: in={ins[0].shape} tile={ins[1].tolist()}")
    return mx.tile(ins[0], ins[1].tolist())

def broadcast_to(ins: List[mx.array], options: tflite.BroadcastToOptions):
    assert len(ins) == 2
    if DEBUG:
        print(f"\tBROADCAST_TO: in={ins[0].shape} target_shape={ins[1].tolist()}")
    return mx.broadcast_to(ins[0], ins[1].tolist())

def _sum(ins: List[mx.array]):
    assert len(ins) == 2
    if DEBUG:
        print(f"\tSUM: in={ins[0].shape} axis={ins[1].tolist()}")
    return mx.sum(ins[0], axis=ins[1].tolist())

def minimum(ins: List[mx.array], options: tflite.MaximumMinimumOptions):
    assert len(ins) == 2
    if DEBUG:
        print(f"\tMINIMUM: in={ins[0].shape} {ins[1].shape}")
    return mx.minimum(ins[0], ins[1])

def maximum(ins: List[mx.array], options: tflite.MaximumMinimumOptions):
    assert len(ins) == 2
    if DEBUG:
        print(f"\tMAXIMUM: in={ins[0].shape} {ins[1].shape}")
    return mx.maximum(ins[0], ins[1])

def gather(ins: List[mx.array], options: tflite.GatherOptions):
    print(f"\tGATHER: in={ins[0].shape} indices={ins[1].tolist()} axis={options.Axis()} batch_dims={options.BatchDims()}")
    return mx.take(ins[0], ins[1], axis=options.Axis())

op_funcs = {
    tflite.BuiltinOperator.MUL: mul,
    tflite.BuiltinOperator.SUB: sub,
    tflite.BuiltinOperator.CONV_2D: conv2d,
    tflite.BuiltinOperator.RSQRT: rsqrt,
    tflite.BuiltinOperator.MEAN: mean,
    tflite.BuiltinOperator.PAD: pad,
    tflite.BuiltinOperator.MAX_POOL_2D: max_pool2d,
    tflite.BuiltinOperator.ADD: add,
    tflite.BuiltinOperator.FULLY_CONNECTED: fully_connected,
    tflite.BuiltinOperator.RESHAPE: reshape,
    tflite.BuiltinOperator.CONCATENATION: concat,
    tflite.BuiltinOperator.SQUARED_DIFFERENCE: squared_difference,
    tflite.BuiltinOperator.STRIDED_SLICE: strided_slice,
    tflite.BuiltinOperator.TRANSPOSE: transpose,
    tflite.BuiltinOperator.BATCH_MATMUL: batch_matmul,
    tflite.BuiltinOperator.SOFTMAX: softmax,
    tflite.BuiltinOperator.GELU: gelu,
    tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: resize_nearest_neighbor,
    tflite.BuiltinOperator.LEAKY_RELU: leaky_relu,
    tflite.BuiltinOperator.TILE: tile,
    tflite.BuiltinOperator.BROADCAST_TO: broadcast_to,
    tflite.BuiltinOperator.SUM: _sum,
    tflite.BuiltinOperator.MINIMUM: minimum,
    tflite.BuiltinOperator.MAXIMUM: maximum,
    tflite.BuiltinOperator.GATHER: gather,
}

op_options = {
    tflite.BuiltinOperator.FULLY_CONNECTED: tflite.FullyConnectedOptions,
    tflite.BuiltinOperator.MUL: tflite.MulOptions,
    tflite.BuiltinOperator.ADD: tflite.AddOptions,
    tflite.BuiltinOperator.SUB: tflite.SubOptions,
    tflite.BuiltinOperator.TRANSPOSE: tflite.TransposeOptions,
    tflite.BuiltinOperator.SOFTMAX: tflite.SoftmaxOptions,
    tflite.BuiltinOperator.STRIDED_SLICE: tflite.StridedSliceOptions,
    tflite.BuiltinOperator.BATCH_MATMUL: tflite.BatchMatMulOptions,
    tflite.BuiltinOperator.GELU: tflite.GeluOptions,
    tflite.BuiltinOperator.SQUARED_DIFFERENCE: tflite.SquaredDifferenceOptions,
    tflite.BuiltinOperator.CONV_2D: tflite.Conv2DOptions,
    tflite.BuiltinOperator.CONCATENATION: tflite.ConcatenationOptions,
    tflite.BuiltinOperator.RESHAPE: tflite.ReshapeOptions,
    tflite.BuiltinOperator.PAD: tflite.PadOptions,
    tflite.BuiltinOperator.MAX_POOL_2D: tflite.Pool2DOptions,
    tflite.BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: tflite.ResizeNearestNeighborOptions,
    tflite.BuiltinOperator.LEAKY_RELU: tflite.LeakyReluOptions,
    tflite.BuiltinOperator.TILE: tflite.TileOptions,
    tflite.BuiltinOperator.BROADCAST_TO: tflite.BroadcastToOptions,
    tflite.BuiltinOperator.MINIMUM: tflite.MaximumMinimumOptions,
    tflite.BuiltinOperator.MAXIMUM: tflite.MaximumMinimumOptions,
    tflite.BuiltinOperator.GATHER: tflite.GatherOptions,
}


def init_arrays(model: tflite.Model, graph: tflite.SubGraph):
    for i in range(graph.TensorsLength()):
        tensors[i] = init_tensor(model, graph.Tensors(i))


def run_mx(path: str, inputs: np.ndarray):
    with open(path, "rb") as f:
        model = tflite.Model.GetRootAs(f.read(), 0)
    for i in range(model.SubgraphsLength()):
        init_arrays(model, model.Subgraphs(i))
        sg = SubGraph(model, model.Subgraphs(i))
        # sg = mx.compile(sg)
        return sg([mx.array(x) for x in inputs])


# TODO Remove this.
tfmodel: tf.lite.Interpreter = None


def run_tf(path: str, input: np.ndarray):
    global tfmodel
    tfmodel = tf.lite.Interpreter(model_path=path)
    tfmodel.allocate_tensors()
    tfinput = tfmodel.get_input_details()
    output = tfmodel.get_output_details()
    for i, v in zip(tfinput, input):
        if math.prod(i["shape"]) != math.prod(v.shape):
            raise ValueError(f"shape mismatch: expected={i['shape']} got={v.shape}")
        tfmodel.set_tensor(i["index"], tf.convert_to_tensor(v))
    tfmodel.invoke()
    return tfmodel.get_tensor(output[0]["index"])


class SubGraph:
    def __init__(self, model: tflite.Model, graph: tflite.SubGraph) -> None:
        self.model = model
        self.graph = graph

    def __call__(self, ins: mx.array):
        for i, _in in zip(range(self.graph.InputsLength()), ins):
            if DEBUG:
                print(f"INPUT: i={i} shape={tensors[self.graph.Inputs(i)].shape}")

            t = tensors[self.graph.Inputs(i)]
            if t is not None and math.prod(t.shape) != math.prod(_in.shape):
                raise ValueError(
                    f"shape mismatch: expected={t.shape} got={_in.shape} tensor={self.graph.Inputs(i)}"
                )
            tensors[self.graph.Inputs(i)] = _in

        for i in range(self.graph.OperatorsLength()):
            op = self.graph.Operators(i)
            op_code = self.model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            in_tensors = [tensors[op.Inputs(j)] for j in range(op.InputsLength())]
            if DEBUG:
                print(
                    f"Running op={op_map[op_code]} with tensors={[t.shape if t is not None else None for t in in_tensors]}"
                )
            if op_code in op_options:
                opt = op_options[op_code]()
                if (_temp := op.BuiltinOptions()) != None:
                    opt.Init(_temp.Bytes, _temp.Pos)
                if op_code in op_funcs:
                    res = op_funcs[op_code](in_tensors, opt)
                else:
                    raise ValueError(
                        f"unsupported op_code={op_code} map_value={op_map[op_code]}"
                    )
            elif op_code in op_funcs:
                res = op_funcs[op_code](in_tensors)
            else:
                raise ValueError(
                    f"unsupported op_code={op_code} map_value={op_map[op_code]}"
                )
            if res is None:
                raise ValueError(
                    f"got no result for op_code={op_code} map_value={op_map[op_code]}"
                )

            # Update array data
            res = [res] if not isinstance(res, list) else res
            if len(res) != op.OutputsLength():
                raise ValueError(
                    f"output length mismatch: {op_map[op_code]} expected={op.OutputsLength()} got={len(res)}"
                )
            for j, v in zip(range(op.OutputsLength()), res):
                t = tensors[op.Outputs(j)]
                if DEBUG:
                    print(
                        f"Setting tensor={op.Outputs(j)} {t.shape if t is not None else None} to shape={v.shape}"
                    )
                if t is not None and math.prod(t.shape) != math.prod(v.shape):
                    raise ValueError(
                        f"shape mismatch: {op_map[op_code]} expected={t.shape} got={v.shape}"
                    )
                if t.shape != v.shape:
                    v = mx.reshape(v, t.shape)
                # TODO: This does not work with intermediate tensors as tf releases them after invoke
                # if COMPARE:
                #     tftensor = tfmodel.get_tensor(op.Outputs(j))
                #     if tftensor.shape != v.shape:
                #         print(f"shape mismatch: tf={tftensor.shape} mx={v.shape}")
                #     np.testing.assert_allclose(
                #         tftensor, np.array(v), rtol=1e-3, atol=1e-3
                #     )
                tensors[op.Outputs(j)] = v
        return [
            tensors[self.graph.Outputs(i)] for i in range(self.graph.OutputsLength())
        ]


def compare_tensors(path: str):
    tfmodel = tf.lite.Interpreter(model_path=path)
    tfmodel.allocate_tensors()
    with open(path, "rb") as f:
        mxmodel = tflite.Model.GetRootAs(f.read(), 0)

    [
        init_arrays(mxmodel, mxmodel.Subgraphs(i))
        for i in range(mxmodel.SubgraphsLength())
    ]
    for i in tensors.keys():
        if tensors[i] is not None:
            try:
                tftensor = tfmodel.get_tensor(i)
                mxtensor = tensors[i]
                if tftensor.shape != mxtensor.shape:
                    print(f"shape mismatch: tf={tftensor.shape} mx={mxtensor.shape}")
                    continue
                if not np.array_equal(tftensor, np.array(mxtensor)):
                    print(f"tensor mismatch: {i}")
            except:
                print(f"tensor not found: {i}")


np.random.seed(0)
import time

if __name__ == "__main__":
    path = os.getenv("MODEL", "./StyleGAN2.tflite")
    print(f"Running model={path}")
    # compare_tensors(path)
    # exit()
    # img = os.getenv("IMG", "car.jpg")
    # img = np.array(Image.open(img).resize((224, 224))).astype(np.float32)
    # img = (img - 127.5) / 127.5
    # img = np.expand_dims(img, axis=0)
    inputs = [
        # img
        np.ones((1, 1), dtype=np.int32)
    ]  # [np.random.random((1, 224, 224, 3)).astype(np.float32)]

    if COMPARE:
        tfstart = time.perf_counter()
        tfpred = run_tf(path, inputs)
        tfpred = tfpred[0]
        tfend = time.perf_counter() - tfstart

    mxstart = time.perf_counter()
    mxpred = run_mx(path, inputs)
    mxpred = np.array(mxpred[0])
    mxend = time.perf_counter() - mxstart
    if COMPARE:
        np.testing.assert_allclose(
            np.expand_dims(tfpred, axis=0), mxpred, rtol=1e-3, atol=1e-3
        )
        print("OUTPUTS MATCHED! 🚀")
        print(f"TF: {tfend:.4f} MX: {mxend:.4f}")
    
    print(f"{mxpred.shape}")
    if False:
        tfpred = np.argmax(tfpred)
        mxpred = np.argmax(mxpred)
        # print(f"TF: {tfpred} MX: {mxpred}")
        labels = []
        with open("./imagenet_labels.txt", "r") as f:
            labels = [l.strip() for l in f.readlines()]
        print(f"TF: {labels[tfpred]}, MLX: {labels[mxpred]}")
