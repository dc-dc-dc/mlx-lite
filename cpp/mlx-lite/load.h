
#include <string>
#include <unordered_map>

#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/utils.h>

#include "schema_generated.h"

using namespace mlx::core;

namespace mlx::lite {
const tflite::Model* load(const std::string &path);

std::unordered_map<int, array> load_arrays(const tflite::Model* model, int subgraph = 0);
}