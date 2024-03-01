import MLXLite
import Foundation

let url = FileManager.default.currentDirectoryPath
let path = URL(fileURLWithPath: url).appendingPathComponent("op.tflite")
let data = try! Data(contentsOf: path)
let model = try loadModel(at: data)

print(model.version)
print(model.hasSubgraphs)
print(model.subgraphsCount)
print(model.subgraphsCount)