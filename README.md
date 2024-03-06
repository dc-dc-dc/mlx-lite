# MLX-Lite

A package for running tflite files in MLX.

## Installation

```shell
pip install -e .
```

## Usage

```python
from mlxlite import load_model
m = load_model("./path-to-model.tflite")
sub = m.get_subgraph(0) # index of the subgraph, default is 0
sub.init_arrays() 
res = sub([mx.array(...), mx.array(...)]) # run with array inputs
```

## Examples

- [ResNet50](./examples/resnet50/README.md)

## Setup Flatbuffers

tensorflow utilizes flatbuffers to create tflite files, to enable parsing in other languages you first need to load the schema used to define the file and generate a language equivalent. To help streamline this a helper script was created called `setup_fbs.sh` which downloads the tensorflow version of the fbs used and generates the swift/cpp equivalent utilizing `flatc` compiler.

If you do not have `flatc` installed run the following command on macos

```shell
brew install flatbuffers
```

To run the setup script

This script will fetch the fbs schema from tensorflow repo and run `flatc`

```shell
./setup_fbs.sh
```
