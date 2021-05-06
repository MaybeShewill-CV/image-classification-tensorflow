# image-classification-tensorflow
cnn models used for image classification


|  机器ip   | 模型名称  |  Total Epoch   | WarmUp Epoch  |  Init LR   | Optimizer  |  Eval Stat/Epoch |
|  ----  | :----:  |  :----:  | :----:  |  :----:  | :----:  |  :----:  |
| .42.198 | Xception | 128 | 5 | 0.01 | SGD | 0.68660 / 48 |
| .42.202 | VGG-16 | 128 | 5 | 0.0125 | SGD | 0.71685 / 128 |
| .42.203 | ResNet-101 | 128 | 5 | 0.0125 | SGD | 0.75167 / 120 |
| .42.204 | DenseNet-121 | 128 | 5 | 0.0125 | SGD | 0.70053 / 128 |
| .42.211 | ResNet-50 | 128 | 5 | 0.0125 | SGD | 0.74660 / 128 |
| .42.212 | MobileNetV2 | 128 | 5 | 0.03 | SGD | 0.704531 / 128 |
| .58.8 | ResNet-18 | 128 | 5 | 0.0125 | SGD | 0.69476 / 128 |

|  机器ip   | 模型名称  |  Total Epoch   | WarmUp Epoch  |  Init LR   | Optimizer  |  Eval Stat/Epoch |
|  ----  | :----:  |  :----:  | :----:  |  :----:  | :----:  |  :----:  |
| .42.202 | VGG-16 | 192 | 5 | 0.0125 | SGD | 0.53103 / 16 |
| .42.204 | DenseNet-121 | 192 | 5 | 0.0125 | SGD | 0.65065 / 68 |
| .42.211 | ResNet-50 | 192 | 5 | 0.0125 | SGD | 0.59099 / 68 |
| .42.212 | MobileNetV2 | 192 | 5 | 0.03 | SGD | 0.66243 / 112 |
| .58.8 | ResNet-18 | 192 | None | 0.01 | SGD | None / None |
| .58.8 | ResNet-50 | 192 | None | 0.01 | SGD | None / None |