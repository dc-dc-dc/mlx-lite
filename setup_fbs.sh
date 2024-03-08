#! /bin/bash

# This script downloads the schema.fbs file from the TensorFlow repository
# and compiles it into swift using the flatc compiler.
# To install flatc use:
#   brew install flatbuffers

if ! [ -x "$(command -v flatc)" ]; then
  echo 'Error: flatc is not installed.' >&2
  exit 1
fi

version="2.15"
if [ ! -f "schema.fbs" ]; then
  echo "Info: Downloading schema.fbs from TensorFlow repository version: ${version}"
  wget https://raw.githubusercontent.com/tensorflow/tensorflow/r${version}/tensorflow/lite/schema/schema.fbs
else
    echo "Info: schema.fbs already exists, utilizing existing file"
fi

flatc --swift -o swift/Sources/MLXLite schema.fbs
flatc --cpp -o cpp/src/ schema.fbs
flatc --python -o mlxlite/generated schema.fbs
find mlxlite/generated -name "*.py" -exec sed -i'' -e 's|from tflite|from mlxlite.generated.tflite|g' {} \;  
rm mlxlite/generate/tflite/*.py-e # remove the backups created