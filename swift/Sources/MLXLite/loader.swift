import MLX
import FlatBuffers
import Foundation

public typealias Model = tflite_Model
public typealias Subgraph = tflite_SubGraph


public func runSubgraph(model: Model, subgraphIndex: Int) {
    
}

public func loadModel(at data: Data) throws -> Model  {
    var buf = ByteBuffer(data: data)
    var verifier = try Verifier(buffer: &buf)
    try ForwardOffset<Model>.verify(&verifier, at: 0, of: Model.self)
    return Model.init(buf, o: Int32(buf.read(def: UOffset.self, position: buf.reader)) + Int32(buf.reader))
}