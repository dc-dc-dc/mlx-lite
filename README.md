# MLX-Lite

A package for running tflite files in MLX.

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
