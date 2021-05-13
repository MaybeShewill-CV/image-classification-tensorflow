# image-classification-tensorflow
cnn models used for image classification


|  机器ip   | 模型名称  |  Total Epoch   | WarmUp Epoch  |  Init LR   |  BS  | Optimizer  |  Eval Stat/Epoch |
|  ----  | :----:  |  :----:  | :----:  |  :----:  | :----: | :----:  |  :----:  |
| .42.198 | Xception | 128 | 5 | 0.01 | 24 | SGD | 0.69868 / 96 |
| .42.202 | VGG-16 | 128 | 5 | 0.0125 | 32 | SGD | 0.71685 / 128 |
| .42.203 | ResNet-101 | 128 | 5 | 0.0125 | 32 | SGD | 0.75602 / 128 |
| .42.204 | DenseNet-121 | 128 | 5 | 0.0125 | 32 | SGD | 0.70053 / 128 |
| .42.211 | ResNet-50 | 128 | 5 | 0.0125 | 32 | SGD | 0.74660 / 128 |
| .42.212 | MobileNetV2 | 128 | 5 | 0.03 | 64 | SGD | 0.704531 / 128 |
| .58.8 | ResNet-18 | 128 | 5 | 0.0125 | 32 | SGD | 0.69476 / 128 |

|  机器ip   | 模型名称  |  Total Epoch   | WarmUp Epoch  |  Init LR   |  BS  | Optimizer  |  Eval Stat/Epoch |
|  ----  | :----:  |  :----:  | :----:  |  :----:  | :----: | :----:  |  :----:  |
| .42.202 | VGG-16 | 192 | 5 | 0.0125 | 32 | SGD | 0.63281 / 128 |
| .42.203 | ResNet-101 | 192 | 5 | 0.0125 | 32 | SGD | 0.62848/ 80 |
| .42.204 | DenseNet-121 | 192 | 5 | 0.0125 | 32 | SGD | 0.69875 / 192 |
| .42.211 | ResNet-50 | 192 | 5 | 0.0125 | 32 | SGD | 0.74475 / 192 |
| .42.212 | MobileNetV2 | 192 | 5 | 0.03 | 64 | SGD | 0.69936 / 192 |
| .58.249 | DarkNet53 | 128 | 5 | 0.01 | 32 | SGD | 0.66048 / 8 |

|  机器ip   | 模型名称  |  Total Epoch   | WarmUp Epoch  |  Init LR   |  BS  | Optimizer  |  Eval Stat/Epoch |
|  ----  | :----:  |  :----:  | :----:  |  :----:  | :----: | :----:  |  :----:  |
| .42.204 | DenseNet-121 | 96 | None | 0.0075 | 16 | SGD | 0.61759 / 20 |
| .42.211 | ResNet-50 | 128 | 5 | 0.0075 | 24 | SGD | 0.59074 / 32 |
| .42.212 | MobileNetV2 | 192 | 5 | 0.02 | 64 | SGD | 0.67799 / 148 |