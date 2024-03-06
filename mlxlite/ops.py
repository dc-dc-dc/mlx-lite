from mlxlite.generated.tflite.FullyConnectedOptions import FullyConnectedOptions
from mlxlite.generated.tflite.BuiltinOperator import BuiltinOperator
from mlxlite.generated.tflite.AddOptions import AddOptions
from mlxlite.generated.tflite.SubOptions import SubOptions
from mlxlite.generated.tflite.MulOptions import MulOptions
from mlxlite.generated.tflite.DivOptions import DivOptions
from mlxlite.generated.tflite.TransposeOptions import TransposeOptions
from mlxlite.generated.tflite.SoftmaxOptions import SoftmaxOptions
from mlxlite.generated.tflite.ReshapeOptions import ReshapeOptions
from mlxlite.generated.tflite.StridedSliceOptions import StridedSliceOptions
from mlxlite.generated.tflite.ConcatenationOptions import ConcatenationOptions
from mlxlite.generated.tflite.MaximumMinimumOptions import MaximumMinimumOptions
from mlxlite.generated.tflite.PadOptions import PadOptions
from mlxlite.generated.tflite.Pool2DOptions import Pool2DOptions
from mlxlite.generated.tflite.SquaredDifferenceOptions import SquaredDifferenceOptions
from mlxlite.generated.tflite.GeluOptions import GeluOptions
from mlxlite.generated.tflite.BatchMatMulOptions import BatchMatMulOptions
from mlxlite.generated.tflite.LeakyReluOptions import LeakyReluOptions
from mlxlite.generated.tflite.TileOptions import TileOptions
from mlxlite.generated.tflite.GatherOptions import GatherOptions
from mlxlite.generated.tflite.BroadcastToOptions import BroadcastToOptions
from mlxlite.generated.tflite.Conv2DOptions import Conv2DOptions
from mlxlite.generated.tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions
from mlxlite.generated.tflite.ActivationFunctionType import ActivationFunctionType
from mlxlite.generated.tflite.Padding import Padding
from mlxlite.generated.tflite.ArgMaxOptions import ArgMaxOptions

import mlx.core as mx
import mlx.nn as nn

from typing import List
import math


def handle_activation(x: mx.array, act: int):
    if act == ActivationFunctionType.NONE:
        return x
    elif act == ActivationFunctionType.RELU:
        return nn.relu(x)

    raise ValueError(f"unsupported activation function: {act}")


def op_add(ins: List[mx.array], options: AddOptions):
    return handle_activation(mx.add(ins[0], ins[1]), options.FusedActivationFunction())


def op_sub(ins: List[mx.array], options: SubOptions):
    return handle_activation(
        mx.subtract(ins[0], ins[1]), options.FusedActivationFunction()
    )


def op_mul(ins: List[mx.array], options: MulOptions):
    return handle_activation(
        mx.multiply(ins[0], ins[1]), options.FusedActivationFunction()
    )


def op_div(ins: List[mx.array], options: DivOptions):
    return handle_activation(
        mx.divide(ins[0], ins[1]), options.FusedActivationFunction()
    )


def op_rsqrt(ins: List[mx.array]):
    return mx.rsqrt(ins[0])


def op_mean(ins: List[mx.array]):
    return mx.mean(ins[0], axis=ins[1].tolist())


def conv2d(ins: List[mx.array], options: Conv2DOptions):
    x = ins[0]
    filter = ins[1]
    if x.ndim != 4:
        x = mx.expand_dims(x, axis=0)
    padding = [0, 0]
    if options.Padding() == Padding.SAME:
        if (off := x.shape[1] % filter.shape[1]) != 0:
            padding[0] = math.ceil(off / 2)
        if (off := x.shape[2] % filter.shape[2]) != 0:
            padding[1] = math.ceil(off / 2)
        if padding == [0, 0] and (options.StrideH() == 1 and options.StrideW() == 1):
            padding = [
                math.ceil((filter.shape(1) - 1) / 2),
                math.ceil((filter.shape(2) - 1) / 2),
            ]

    x = mx.conv2d(
        x,
        filter,
        stride=(options.StrideH(), options.StrideW()),
        padding=padding,
        dilation=(options.DilationHFactor(), options.DilationWFactor()),
    )
    if len(ins) == 3:
        x = mx.add(x, ins[2])

    return handle_activation(x, options.FusedActivationFunction())


def op_pad(ins: List[mx.array], options: PadOptions):
    return mx.pad(ins[0], ins[1].tolist())


def max_pool2d(ins: List[mx.array], options: Pool2DOptions):
    assert options.Padding() == Padding.VALID, "only SAME padding is supported"
    x = nn.MaxPool2d(
        (options.FilterHeight(), options.FilterWidth()),
        (options.StrideH(), options.StrideW()),
    )(ins[0])
    return handle_activation(x, options.FusedActivationFunction())


def op_fully_connected(ins: List[mx.array], options: FullyConnectedOptions):
    x = (
        mx.addmm(ins[2], ins[0], ins[1].T)
        if len(ins) == 3 and isinstance(ins[2], mx.array)
        else mx.matmul(ins[0], ins[1].T)
    )
    return handle_activation(x, options.FusedActivationFunction())

def op_reshape(ins: List[mx.array], options: ReshapeOptions):
    return mx.reshape(ins[0], ins[1].tolist())

def op_concat(ins: List[mx.array], options: ConcatenationOptions):
    return mx.concatenate(ins, axis=options.Axis())

def op_squared_difference(ins: List[mx.array], options: SquaredDifferenceOptions):
    return mx.square(mx.subtract(ins[0], ins[1]))

def op_strided_slice(ins: List[mx.array], options: StridedSliceOptions):
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

def op_transpose(ins: List[mx.array], options: TransposeOptions):
    return mx.transpose(ins[0], ins[1].tolist())

def op_batch_matmul(ins: List[mx.array], options: BatchMatMulOptions):
    assert not options.AdjX(), "adj_x not supported"
    assert not options.AdjY(), "adj_y not supported"
    assert not options.AsymmetricQuantizeInputs(), "asymmetric_quantize_inputs not supported"
    return mx.matmul(ins[0], ins[1])

def op_softmax(ins: List[mx.array], options: SoftmaxOptions):
    assert options.Beta() == 1.0, "only beta of 1.0 supported"
    return mx.softmax(ins[0], axis=-1)

def op_gelu(ins: List[mx.array], options: GeluOptions):
    return nn.gelu_approx(ins[0]) if options.Approximate() else nn.gelu(ins[0])

def op_resize_nearest_neighbor(ins: List[mx.array], options: ResizeNearestNeighborOptions):
    scale_factor = [n // i for i, n in zip(ins[0].shape[1:-1], ins[1].tolist())]
    return nn.Upsample(scale_factor=scale_factor, mode="nearest")(ins[0])

def op_leaky_relu(ins: List[mx.array], options: LeakyReluOptions):
    return nn.leaky_relu(ins[0], options.Alpha())

def op_tile(ins: List[mx.array], options: TileOptions):
    return mx.tile(ins[0], ins[1].tolist())

def op_broadcast_to(ins: List[mx.array], options: BroadcastToOptions):
    return mx.broadcast_to(ins[0], ins[1].tolist())

def op_sum(ins: List[mx.array]):
    return mx.sum(ins[0], axis=ins[1].tolist())

def op_maximum(ins: List[mx.array], options: MaximumMinimumOptions):
    return mx.maximum(ins[0], ins[1])

def op_minimum(ins: List[mx.array], options: MaximumMinimumOptions):
    return mx.minimum(ins[0], ins[1])

def op_gather(ins: List[mx.array], options: GatherOptions):
    return mx.take(ins[0], ins[1], axis=options.Axis())

def op_logistic(ins: List[mx.array]):
    return nn.sigmoid(ins[0])

def op_argmax(ins: List[mx.array], options: ArgMaxOptions):
    return mx.argmax(ins[0])

op_map = {
    BuiltinOperator.ADD: (op_add, AddOptions),
    BuiltinOperator.LOGISTIC: (op_logistic, None),
    BuiltinOperator.ARG_MAX: (op_argmax, ArgMaxOptions),
    BuiltinOperator.SUB: (op_sub, SubOptions),
    BuiltinOperator.MUL: (op_mul, MulOptions),
    BuiltinOperator.DIV: (op_div, DivOptions),
    BuiltinOperator.TRANSPOSE: (op_transpose, TransposeOptions),
    BuiltinOperator.SOFTMAX: (op_softmax, SoftmaxOptions),
    BuiltinOperator.RESHAPE: (op_reshape, ReshapeOptions),
    BuiltinOperator.STRIDED_SLICE: (op_strided_slice, StridedSliceOptions),
    BuiltinOperator.CONCATENATION: (op_concat, ConcatenationOptions),
    BuiltinOperator.MAXIMUM: (op_maximum, MaximumMinimumOptions),
    BuiltinOperator.MINIMUM: (op_minimum, MaximumMinimumOptions),
    BuiltinOperator.PAD: (op_pad, PadOptions),
    BuiltinOperator.MAX_POOL_2D: (max_pool2d, Pool2DOptions),
    BuiltinOperator.SQUARED_DIFFERENCE: (op_squared_difference, SquaredDifferenceOptions),
    BuiltinOperator.GELU: (op_gelu, GeluOptions),
    BuiltinOperator.BATCH_MATMUL: (op_batch_matmul, BatchMatMulOptions),
    BuiltinOperator.LEAKY_RELU: (op_leaky_relu, LeakyReluOptions),
    BuiltinOperator.TILE: (op_tile, TileOptions),
    BuiltinOperator.GATHER: (op_gather, GatherOptions),
    BuiltinOperator.BROADCAST_TO: (op_broadcast_to, BroadcastToOptions),
    BuiltinOperator.CONV_2D: (conv2d, Conv2DOptions),
    BuiltinOperator.RESIZE_NEAREST_NEIGHBOR: (op_resize_nearest_neighbor, ResizeNearestNeighborOptions),
    BuiltinOperator.FULLY_CONNECTED: (op_fully_connected, FullyConnectedOptions),
    BuiltinOperator.RSQRT: (op_rsqrt, None),
    BuiltinOperator.MEAN: (op_mean, None),
    BuiltinOperator.SUM: (op_sum, None),
}
