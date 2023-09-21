<p align="center">
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white" align="center">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" align="center">
<img src="https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor "align="center">
</p>

# ResNet-9

This repository contains an implementation of a lightweight deep residual network – ResNet-9 – created from scratch in PyTorch. This model serves as a less computationally-intensive alternative to larger, deeper networks, while providing a similar level of accuracy for less complex image classification problems.

## Overview

This implementation was inspired by the need for a faster, smaller model for image classification, especially in scenarios where resources might be limited. While models like ResNet-18, ResNet-50, or larger might offer higher performance, they are often "overkill" for simpler tasks and can be more resource-demanding. ResNet-9 provides a good middle ground, maintaining the core concepts of ResNet, but shrinking down the network size and computational complexity.

## Model Details

The ResNet-9 model consists of nine layers with weights; two Residual Blocks (each containing two convolutional layers), one initial convolution layer, and a final fully connected layer. The implementation also includes Batch Normalization and Relu Activations. Don't forget to adjust the `num_classes` parameter to match your specific problem.

### Usage

To create a ResNet-9 for a classification problem with for example 12 classes, you could use:

```python
net = ResNet(ResidualBlock, num_classes=12)
```

Further training and inference would follow standard PyTorch routines.

## Authors

* **Petr Vanek** - *Initial work* - [VanekPetr](https://github.com/VanekPetr)

## Contributing
Thank you for considering contributing to this project! We welcome contributions from everyone.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/VanekPetr/ResNet-9/tags). 

## License

This repository is licensed under [MIT](LICENSE) (c) 2023 GitHub, Inc.

<div align='center'>
<a href='https://github.com/vanekpetr/ResNet-9/releases'>
<img src='https://img.shields.io/github/v/release/vanekpetr/ResNet-9?color=%23FDD835&label=version&style=for-the-badge'>
</a>
<a href='https://github.com/vanekpetr/ResNet-9/blob/main/LICENSE'>
<img src='https://img.shields.io/github/license/vanekpetr/ResNet-9?style=for-the-badge'>
</a>
</div>
