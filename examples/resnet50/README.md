# ResNet50 Example

Note: This example requires `Pillow` and `numpy`

Download the model [here](https://huggingface.co/qualcomm/ResNet50/tree/main)

```shell
wget https://huggingface.co/qualcomm/ResNet50/resolve/main/ResNet50.tflite
```

Run the model

```shell
python ./resnet.py
```

Run against a different image

```shell
IMG="./gorilla.jpg" python ./resnet.py
```
