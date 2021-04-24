# image-classification-tensorflow
cnn models used for image classification


|  机器ip   | 模型名称  |  Total Epoch   | WarmUp Epoch  |  Init LR   | Optimizer  |  Eval Stat/Epoch |
|  ----  | :----:  |  :----:  | :----:  |  :----:  | :----:  |  :----:  |
| 192.168.42.198 | Xception | 128 | 4 | 0.045 | SGD | 0.36162 / 8 |
| 192.168.42.202 | VGG-16 | 128 | None | 0.01 | SGD | 0.5999 / 16 |
| 192.168.42.203 | ResNet-101 | 256 | None | 0.1 | SGD | 0.23991 / 16 |
| 192.168.42.204 | DenseNet-121 | 128 | 4 | 0.1 | SGD | 0.55873 / 28 |
| 192.168.42.211 | ResNet-50 | 128 | 5 | 0.01 | SGD | 0.6555 / 96 |
| 192.168.42.212 | MobileNetV2 | 256 | 5 | 0.1 | SGD | 0.43519 / 40 |
| 172.18.58.235 | MobileNetV2 | 128 | None | 0.01 | SGD | 0.61 / 36 |