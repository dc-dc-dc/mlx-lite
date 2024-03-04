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

array load_buffer(const tflite::Buffer *buffer, Dtype type,
                  std::vector<int> shape) {
  switch (type) {
  case float32:
    return array(reinterpret_cast<const float *>(buffer->data()->data()), shape,
                 type);
  case float16:
    return array(reinterpret_cast<const float16_t *>(buffer->data()->data()),
                 shape, type);
  case int8:
    return array(reinterpret_cast<const int8_t *>(buffer->data()->data()),
                 shape, type);
  case int16:
    return array(reinterpret_cast<const int16_t *>(buffer->data()->data()),
                 shape, type);
  case int32:
    return array(reinterpret_cast<const int32_t *>(buffer->data()->data()),
                 shape, type);
  case int64:
    return array(reinterpret_cast<const int64_t *>(buffer->data()->data()),
                 shape, type);
  case uint8:
    return array(reinterpret_cast<const uint8_t *>(buffer->data()->data()),
                 shape, type);
  case uint16:
    return array(reinterpret_cast<const uint16_t *>(buffer->data()->data()),
                 shape, type);
  case uint32:
    return array(reinterpret_cast<const uint32_t *>(buffer->data()->data()),
                 shape, type);
  case uint64:
    return array(reinterpret_cast<const uint64_t *>(buffer->data()->data()),
                 shape, type);
  case bool_:
    return array(reinterpret_cast<const bool *>(buffer->data()->data()), shape,
                 type);
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

std::vector<array> get_inputs(std::unordered_map<int, array> &arrays,
                              const flatbuffers::Vector<int32_t> *inputs) {
  std::vector<array> ins;
  for (int i = 0; i < inputs->size(); i++) {
    auto index = inputs->Get(i);
    if (arrays.find(index) == arrays.end()) {
      throw std::runtime_error("[get_inputs] input not found");
    }
    ins.push_back(arrays.at(index));
  }
  return ins;
}

void set_inputs(const tflite::Model *model,
                std::unordered_map<int, array> &arrays, std::vector<array> ins,
                int subgraph /*= 0*/) {
  if (subgraph >= model->subgraphs()->size()) {
    throw std::runtime_error("[set_inputs] subgraph out of range");
  }
  auto _subgraph = model->subgraphs()->Get(subgraph);
  if (ins.size() != _subgraph->inputs()->size()) {
    throw std::runtime_error("[set_inputs] input size mismatch");
  }
  for (int i = 0; i < _subgraph->inputs()->size(); i++) {
    auto index = _subgraph->inputs()->Get(i);
    if (arrays.find(index) == arrays.end()) {
      throw std::runtime_error("[set_inputs] input not found");
    }
    auto fin = ins.at(i);
    if (fin.dtype() != arrays.at(index).dtype()) {
      throw std::runtime_error("[set_inputs] input dtype mismatch");
    }
    if (fin.shape() != arrays.at(index).shape()) {
      throw std::runtime_error("[set_inputs] input shape mismatch");
    }
    arrays.at(index) = fin;
  }
}

std::vector<array> get_outputs(std::unordered_map<int, array> &arrays,
                                const flatbuffers::Vector<int32_t> *outputs) {
  std::vector<array> outs;
  for (int i = 0; i < outputs->size(); i++) {
    auto index = outputs->Get(i);
    if (arrays.find(index) == arrays.end()) {
      throw std::runtime_error("[get_outputs] output not found");
    }
    outs.push_back(arrays.at(index));
  }
  return outs;
}

std::vector<array> run_op(tflite::BuiltinOperator code, std::vector<array> ins,
                          const tflite::Operator *op) {
  switch (code) {
  case tflite::BuiltinOperator_SOFTMAX:
    return {softmax(ins[0], -1)};
  default:
    throw std::runtime_error("[run_op] Unsupported operator code");
  }
}

std::vector<array> run_graph(const tflite::Model *model,
                             std::unordered_map<int, array> arrays,
                             int subgraph /* = 0 */) {

  if (subgraph >= model->subgraphs()->size()) {
    throw std::runtime_error("[run_graph] subgraph out of range");
  }
  auto _subgraph = model->subgraphs()->Get(subgraph);
  for (auto i = 0; i < _subgraph->operators()->size(); i++) {
    auto _op = _subgraph->operators()->Get(i);
    auto ins = get_inputs(arrays, _op->inputs());
    auto code =
        model->operator_codes()->Get(_op->opcode_index())->builtin_code();
    auto outs = run_op(code, ins, _op);
    for (auto j = 0; j < _op->outputs()->size(); j++) {
      auto index = _op->outputs()->Get(j);
      if (arrays.find(index) == arrays.end()) {
        throw std::runtime_error("[run_graph] output not found");
      }
      if (arrays.at(index).shape() != outs.at(j).shape()) {
        throw std::runtime_error("[run_graph] output shape mismatch");
      }
      arrays.at(index) = outs.at(j);
    }
  }
  return get_outputs(arrays, _subgraph->outputs());
}
} // namespace mlx::lite