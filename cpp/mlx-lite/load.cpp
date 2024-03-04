#include <fstream>
#include <iostream>
#include <string>

#include <flatbuffers/flatbuffers.h>

#include "load.h"

namespace mlx::lite {
const tflite::Model *load(const std::string &path) {
  std::ifstream infile;
  infile.open(path, std::ios::binary | std::ios::in);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  infile.seekg(0, std::ios::end);
  int length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  char *data = new char[length];
  infile.read(data, length);
  infile.close();

  return tflite::GetModel(data);
}

Dtype getDtype(tflite::TensorType _type) {
  switch (_type) {
  case tflite::TensorType::TensorType_FLOAT16:
    return float16;
  case tflite::TensorType::TensorType_FLOAT32:
    return float32;
  case tflite::TensorType::TensorType_INT8:
    return int8;
  case tflite::TensorType::TensorType_INT16:
    return int16;
  case tflite::TensorType::TensorType_INT32:
    return int32;
  case tflite::TensorType::TensorType_INT64:
    return int64;
  case tflite::TensorType::TensorType_UINT8:
    return uint8;
  case tflite::TensorType::TensorType_UINT16:
    return uint16;
  case tflite::TensorType::TensorType_UINT32:
    return uint32;
  case tflite::TensorType::TensorType_UINT64:
    return uint64;
  case tflite::TensorType::TensorType_BOOL:
    return bool_;
  default:
    throw std::runtime_error("Unsupported type");
  }
}

array load_buffer(const tflite::Buffer *buffer, Dtype type, std::vector<int> shape) {
  switch(type) {
    case float32:
      return array(reinterpret_cast<const float *>(buffer->data()->data()), shape, type);
    case float16:
      return array(reinterpret_cast<const float16_t *>(buffer->data()->data()), shape, type);
    case int8:
      return array(reinterpret_cast<const int8_t *>(buffer->data()->data()), shape, type);
    case int16:
      return array(reinterpret_cast<const int16_t *>(buffer->data()->data()), shape, type);
    case int32:
      return array(reinterpret_cast<const int32_t *>(buffer->data()->data()), shape, type);
    case int64:
      return array(reinterpret_cast<const int64_t *>(buffer->data()->data()), shape, type);
    case uint8:
      return array(reinterpret_cast<const uint8_t *>(buffer->data()->data()), shape, type);
    case uint16:
      return array(reinterpret_cast<const uint16_t *>(buffer->data()->data()), shape, type);
    case uint32:
      return array(reinterpret_cast<const uint32_t *>(buffer->data()->data()), shape, type);
    case uint64:
      return array(reinterpret_cast<const uint64_t *>(buffer->data()->data()), shape, type);
    case bool_:
      return array(reinterpret_cast<const bool *>(buffer->data()->data()), shape, type);
    default:
      throw std::runtime_error("Unsupported type");
  }
}

std::unordered_map<int, array> load_arrays(const tflite::Model *model,
                                           int subgraphIndex /* = 0 */) {
  std::unordered_map<int, array> arrays;
  if (subgraphIndex >= model->subgraphs()->size()) {
    throw std::runtime_error("[load_arrays] subgraphIndex out of range");
  }
  auto subgraph = model->subgraphs()->Get(subgraphIndex);
  for (int i = 0; i < subgraph->tensors()->size(); i++) {
    auto tensor = subgraph->tensors()->Get(i);
    auto shape = tensor->shape();
    auto type = getDtype(tensor->type());
    auto buffer = model->buffers()->Get(tensor->buffer());
    std::vector<int> _shape = {shape->cbegin(), shape->cend()};
    if (buffer->data() != nullptr) {
      arrays.emplace(i, load_buffer(buffer, type, _shape));
    } else {
      arrays.emplace(i, zeros(_shape, type));
    }
  }
  return arrays;
}
} // namespace mlx::lite