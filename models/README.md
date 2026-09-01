# Model Weights

This project evaluates four trained models:

- ResNet18 trained from scratch
- ResNet18 pretrained on ImageNet and fine-tuned
- VGG16-BN trained from scratch
- VGG16-BN pretrained on ImageNet and fine-tuned

The trained model weight files are not included in this repository because of file size considerations.

The models can be reproduced using the training script in:

`src/train.py`

Expected weight filenames:

- `resnet18_scratch_weights.pth`
- `resnet18_pretrained_weights.pth`
- `vgg16_bn_scratch_weights.pth`
- `vgg16_bn_pretrained_weights.pth`
