import mlx.core as mx
from typing import Any, Dict, List, Tuple, Callable
import math
import os
import flatbuffers
from mlxlite.ops import op_map

DEBUG = os.getenv("DEBUG", "0") == "1"

from mlxlite.generated.tflite.Model import Model
from mlxlite.generated.tflite.SubGraph import SubGraph
from mlxlite.generated.tflite.Tensor import Tensor
from mlxlite.generated.tflite.TensorType import TensorType

tensor_type_map = {
    TensorType.BOOL: mx.bool_,
    TensorType.UINT8: mx.uint8,
    TensorType.UINT16: mx.uint16,
    TensorType.UINT32: mx.uint32,
    TensorType.UINT64: mx.uint64,
    TensorType.INT8: mx.int8,
    TensorType.INT16: mx.int16,
    TensorType.INT32: mx.int32,
    TensorType.INT64: mx.int64,
    TensorType.FLOAT16: mx.float16,
    TensorType.FLOAT32: mx.float32,
}

tensor_type_cast = {
    TensorType.BOOL: "?",
    TensorType.UINT8: "B",
    TensorType.UINT16: "H",
    TensorType.UINT32: "I",
    TensorType.UINT64: "L",
    TensorType.INT8: "b",
    TensorType.INT16: "h",
    TensorType.INT32: "i",
    TensorType.INT64: "l",
    TensorType.FLOAT16: "e",
    TensorType.FLOAT32: "f",
}

class MXModel:
    def __init__(self, model: Model):
        self.model = model
        self.op_map: Dict[int, Tuple[Callable[[List[mx.array]], mx.array], Any]] = {}
        self.arrays: Dict[int, mx.array] = {}

    def get_subgraph(self, index: int = 0):
        if index >= self.model.SubgraphsLength():
            raise ValueError(f"subgraph {index} does not exist")
        return MXSubGraph(self, self.model.Subgraphs(index))

    def get_array(self, index: int) -> mx.array:
        if index == -1:
            return None
        if index not in self.arrays:
            raise ValueError(f"array {index} does not exist")
        return self.arrays[index]
    
    def init_array(self, tensor: Tensor):
        buf = self.model.Buffers(tensor.Buffer())
        o = flatbuffers.number_types.UOffsetTFlags.py_type(buf._tab.Offset(4))
        shape = [tensor.Shape(i) for i in range(tensor.ShapeLength())]
        if tensor.Type() not in tensor_type_map:
            raise ValueError(f"tensor type={tensor.Type()} not supported")
        # TODO: add quantization support
        
        dtype = tensor_type_map[tensor.Type()]
        if o == 0:
            return mx.zeros(shape, dtype)
        offset = buf._tab.Vector(o)
        count = buf._tab.VectorLen(o)
        return mx.array(memoryview(buf._tab.Bytes[offset : offset + count]).cast(tensor_type_cast[tensor.Type()], shape=shape), dtype)

    def set_array(self, index: int, value: mx.array, try_reshape=False):
        if index not in self.arrays:
            self.arrays[index] = value
        t = self.arrays[index]
        if t is not None and math.prod(t.shape) != math.prod(value.shape):
            raise ValueError(
                f"shape mismatch: expected={t.shape} got={value.shape} tensor={index}"
            )
        self.arrays[index] = mx.reshape(value, t.shape) if try_reshape else value

    def num_graphs(self) -> int:
        return self.model.SubgraphsLength()

    def get_op(self, code: int, options: Any = None):
        op_code = self.model.OperatorCodes(code).BuiltinCode()
        if op_code not in op_map:
            raise ValueError(f"[get_op] op_code={op_code} not supported")
        (func, opt) = op_map[op_code]
        if options is not None and opt is not None:
            opt = opt()
            opt.Init(options.Bytes, options.Pos)
        return func, opt


class MXSubGraph:
    def __init__(self, model: MXModel, graph: SubGraph) -> None:
        self.model = model
        self.graph = graph

    def get_inputs(self):
        return [
            self.model.get_array(self.graph.Inputs(i))
            for i in range(self.graph.InputsLength())
        ]

    def init_arrays(self):
        for i in range(self.graph.TensorsLength()):
            self.model.set_array(i, self.model.init_array(self.graph.Tensors(i)))

    def __call__(self, ins: List[mx.array]) -> List[mx.array]:
        ins = ins if isinstance(ins, List) else [ins] 
        for i, _in in zip(range(self.graph.InputsLength()), ins):
            if DEBUG:
                print(f"INPUT: i={i} shape={self.tensors[self.graph.Inputs(i)].shape}")
            self.model.set_array(self.graph.Inputs(i), _in)

        for i in range(self.graph.OperatorsLength()):
            op = self.graph.Operators(i)
            in_tensors = [self.model.get_array(op.Inputs(j)) for j in range(op.InputsLength())]
            func, opt = self.model.get_op(op.OpcodeIndex(), op.BuiltinOptions())
            res = func(in_tensors, opt) if opt is not None else func(in_tensors)
            if res is None:
                raise ValueError(f"got no result for op_code={op.OpcodeIndex()}")

            # Update array data
            res = [res] if not isinstance(res, list) else res
            if len(res) != op.OutputsLength():
                raise ValueError(
                    f"output length mismatch: {op.OpcodeIndex()} expected={op.OutputsLength()} got={len(res)}"
                )
            for j, v in zip(range(op.OutputsLength()), res):
                self.model.set_array(op.Outputs(j), v, True)

        return [
            self.model.get_array(self.graph.Outputs(i))
            for i in range(self.graph.OutputsLength())
        ]
