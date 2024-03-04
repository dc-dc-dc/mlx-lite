#include "doctest/doctest.h"

#include "mlx-lite/load.h"

TEST_CASE("test load") {
  auto model = mlx::lite::load("../../op.tflite");
  CHECK_EQ(model->version(), 3);
  auto arrays = mlx::lite::load_arrays(model);
  CHECK_THROWS(mlx::lite::load_arrays(model, 1));
  CHECK_EQ(arrays.size(), 2);
  auto x = arrays.at(0);
  CHECK_EQ(x.shape(), std::vector<int>({1, 8, 8, 1}));
}

TEST_CASE("test run") {
  auto model = mlx::lite::load("../../op.tflite");
  auto arrays = mlx::lite::load_arrays(model);
  CHECK_EQ(arrays.size(), 2);
  CHECK_THROWS(mlx::lite::set_inputs(model, arrays, {}));
  CHECK_THROWS(mlx::lite::set_inputs(model, arrays, {ones({16})}));
  CHECK_THROWS(mlx::lite::set_inputs(model, arrays, {ones({1, 8, 8})}));
  CHECK_THROWS(mlx::lite::set_inputs(model, arrays, {ones({1, 8, 8, 1}), ones({1, 8, 8, 1})}));
  mlx::lite::set_inputs(model, arrays, {ones({1, 8, 8, 1})});
  auto out = mlx::lite::run_graph(model, arrays);
  CHECK_EQ(out.size(), 1);
  CHECK(array_equal(out[0], ones({1, 8, 8, 1})).item<bool>());
}

TEST_CASE("test resnet") {
  auto model = mlx::lite::load("../../ResNet50.tflite");
  CHECK_EQ(model->version(), 3);
  auto arrays = mlx::lite::load_arrays(model);
  CHECK_EQ(arrays.size(), 189);
  auto x = arrays.at(1);
  auto expected = array(
      {0.2247861623764038,   0.6160263419151306,      0.01131313294172287,
       0.13409118354320526,  0.1800396591424942,      0.14786501228809357,
       0.17428146302700043,  0.1902552992105484,      0.2322368174791336,
       0.19948673248291016,  0.12878000736236572,     -0.21446888148784637,
       0.15097272396087646,  -3.9216942582243064e-08, 0.2504678964614868,
       0.20425228774547577,  0.5501607656478882,      0.21034212410449982,
       0.2246486395597458,   0.47126027941703796,     0.2375287562608719,
       0.20461787283420563,  0.21630124747753143,     0.6556571125984192,
       0.22690816223621368,  0.6599755883216858,      0.20618516206741333,
       0.19299788773059845,  0.11056351661682129,     0.33052200078964233,
       0.12406208366155624,  0.04317320138216019,     0.7731648683547974,
       0.2699756324291229,   0.33691129088401794,     0.5705695152282715,
       0.1500382274389267,   0.1746693104505539,      0.193894162774086,
       0.17255130410194397,  0.8054161667823792,      0.2374676614999771,
       -0.43538811802864075, 0.8472456336021423,      -0.3802163898944855,
       0.24828225374221802,  0.1809040755033493,      0.3296544849872589,
       -0.28314006328582764, 0.22714683413505554,     0.26178622245788574,
       0.05761786177754402,  -0.5040741562843323,     0.15764929354190826,
       0.17895302176475525,  0.2811422348022461,      0.4144947826862335,
       -0.09663743525743484, -0.31476929783821106,    -0.02535378560423851,
       0.08844863623380661,  0.18039242923259735,     0.8314655423164368,
       0.25088798999786377},
      {
          64,
      });
  CHECK(array_equal(x, expected).item<bool>());
}