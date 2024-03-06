# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Model(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Model()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsModel(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ModelBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # Model
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Model
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Model
    def OperatorCodes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from tflite.OperatorCode import OperatorCode
            obj = OperatorCode()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def OperatorCodesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def OperatorCodesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # Model
    def Subgraphs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from tflite.SubGraph import SubGraph
            obj = SubGraph()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def SubgraphsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def SubgraphsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Model
    def Description(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Model
    def Buffers(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from tflite.Buffer import Buffer
            obj = Buffer()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def BuffersLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def BuffersIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

    # Model
    def MetadataBuffer(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Model
    def MetadataBufferAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Model
    def MetadataBufferLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def MetadataBufferIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        return o == 0

    # Model
    def Metadata(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from tflite.Metadata import Metadata
            obj = Metadata()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def MetadataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def MetadataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        return o == 0

    # Model
    def SignatureDefs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from tflite.SignatureDef import SignatureDef
            obj = SignatureDef()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Model
    def SignatureDefsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Model
    def SignatureDefsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        return o == 0

def ModelStart(builder):
    builder.StartObject(8)

def Start(builder):
    ModelStart(builder)

def ModelAddVersion(builder, version):
    builder.PrependUint32Slot(0, version, 0)

def AddVersion(builder, version):
    ModelAddVersion(builder, version)

def ModelAddOperatorCodes(builder, operatorCodes):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(operatorCodes), 0)

def AddOperatorCodes(builder, operatorCodes):
    ModelAddOperatorCodes(builder, operatorCodes)

def ModelStartOperatorCodesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOperatorCodesVector(builder, numElems: int) -> int:
    return ModelStartOperatorCodesVector(builder, numElems)

def ModelAddSubgraphs(builder, subgraphs):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(subgraphs), 0)

def AddSubgraphs(builder, subgraphs):
    ModelAddSubgraphs(builder, subgraphs)

def ModelStartSubgraphsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartSubgraphsVector(builder, numElems: int) -> int:
    return ModelStartSubgraphsVector(builder, numElems)

def ModelAddDescription(builder, description):
    builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(description), 0)

def AddDescription(builder, description):
    ModelAddDescription(builder, description)

def ModelAddBuffers(builder, buffers):
    builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(buffers), 0)

def AddBuffers(builder, buffers):
    ModelAddBuffers(builder, buffers)

def ModelStartBuffersVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartBuffersVector(builder, numElems: int) -> int:
    return ModelStartBuffersVector(builder, numElems)

def ModelAddMetadataBuffer(builder, metadataBuffer):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(metadataBuffer), 0)

def AddMetadataBuffer(builder, metadataBuffer):
    ModelAddMetadataBuffer(builder, metadataBuffer)

def ModelStartMetadataBufferVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartMetadataBufferVector(builder, numElems: int) -> int:
    return ModelStartMetadataBufferVector(builder, numElems)

def ModelAddMetadata(builder, metadata):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(metadata), 0)

def AddMetadata(builder, metadata):
    ModelAddMetadata(builder, metadata)

def ModelStartMetadataVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartMetadataVector(builder, numElems: int) -> int:
    return ModelStartMetadataVector(builder, numElems)

def ModelAddSignatureDefs(builder, signatureDefs):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(signatureDefs), 0)

def AddSignatureDefs(builder, signatureDefs):
    ModelAddSignatureDefs(builder, signatureDefs)

def ModelStartSignatureDefsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartSignatureDefsVector(builder, numElems: int) -> int:
    return ModelStartSignatureDefsVector(builder, numElems)

def ModelEnd(builder):
    return builder.EndObject()

def End(builder):
    return ModelEnd(builder)
