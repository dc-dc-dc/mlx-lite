
#include <string>
#include <vector>
#include <unordered_map>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/utils.h>

#include "schema_generated.h"

using namespace mlx::core;

namespace mlx::lite {
const tflite::Model* load(const std::string &path);

std::unordered_map<int, array> load_arrays(const tflite::Model* model, int subgraph = 0);

void set_inputs(const tflite::Model* model, std::unordered_map<int, array>& arrays, std::vector<array> ins, int subgraph = 0);

std::vector<array> run_graph(const tflite::Model* model, std::unordered_map<int, array> arrays, int subgraph = 0); 
}