# MLP on CIFAR-10

本项目旨在测试多层感知机（MLP）在 CIFAR-10 数据集上的图像分类能力，并通过可视化与超参数搜索等方式，探索和比较不同配置对模型性能的影响。

## 1.项目目的

- **数据集**：使用 CIFAR-10 数据集，对图像进行分类任务。
- **模型**：构建可支持多种激活函数的 MLP 模型。
- **训练与分析**：在学习率、隐藏层大小、正则化强度等多个超参数的维度上，分析模型的表现并可视化结果。

## 2.目录结构

```
plaintext复制编辑├── dataset.py
├── hyperparameter_search.py
├── model.py
├── train.py
├── main.py
├── display
│   ├── activation function.ipynb
│   ├── learning rate.ipynb
│   ├── multi-layer.ipynb
│   └── hyperparameter-search.ipynb
└── script
│   ├── run_hyperparam.sh
│   ├── run_train.sh
```

## 3.代码文件说明

- **dataset.py**
  主要负责对 CIFAR-10 数据集进行加载与预处理操作。包括数据下载、解压、数据增强、标准化处理等，为后续训练提供高质量的输入。
- **hyperparameter_search.py**
  实现对学习率（learning rate）、隐藏层大小（hidden layer size）、正则化强度（regularization strength）等超参数进行搜索与调优。可使用网格搜索或其他搜索策略来比较不同参数配置下的模型表现。
- **model.py**
  用于构建多层感知机（MLP）模型，支持多种激活函数（如 ReLU、Sigmoid、Tanh 等）。可根据需求灵活定义模型层数、神经元数量等超参数。
- **train.py**
  封装训练逻辑，包括前向传播、损失计算、反向传播与优化步骤。通过定期打印或记录训练过程中的损失与准确率，帮助追踪模型收敛情况。
- **main.py**
  项目的主入口文件，可根据传入的不同参数选择执行模型训练或超参数搜索。可在此配置训练轮数、批次大小、学习率等核心参数，或者调用超参数搜索模块自动搜索最优配置。
- **display 文件夹**
  - **activation function.ipynb**
    通过可视化实验，比较不同激活函数（如 ReLU、Sigmoid、Tanh 等）对模型准确率和训练收敛速度的影响。
  - **learning rate.ipynb**
    研究不同学习率对于模型收敛速度和最终准确率的影响，帮助寻找合适的学习率范围。
  - **multi-layer.ipynb**
    测试并对比不同隐藏层数的模型表现，例如单层、两层或更多层的 MLP，在准确率和收敛速度上的差异。
  - **hyperparameter-search.ipynb**
    对超参数搜索的结果进行可视化呈现，包括不同参数组合下模型在验证集或测试集上的准确率、损失等指标。
- **script 文件夹**
  - **run_hyperparam.sh**
    参数搜索运行脚本
  - **run_train.sh**
    训练运行脚本

 `ThreeLayerNet` 模型在初始化时允许你手动调整以下几个关键参数：

1. **input_size**：输入数据的维度（例如，CIFAR-10 的图像尺寸展平后通常为 3072，即 32×32×332 \times 32 \times 332×32×3）。
2. **hidden_size**：隐藏层的大小，即隐藏层神经元的数量。
3. **output_size**：输出层的大小，通常对应分类问题中的类别数（例如，CIFAR-10 为 10）。
4. **activation**：激活函数的类型，原始实现通常支持 `'relu'` 和 `'sigmoid'`，你也可以扩展其他激活函数。
5. **std**：权重初始化的标准差，用于控制随机初始化时权重的尺度。

通过以上各组件的协同工作，可较为系统地研究 MLP 在 CIFAR-10 数据集上的性能表现，并为后续模型改进与超参数选择提供参考。

在训练文件中的 `train` 函数中，你可以调整的参数包括：

- **learning_rate**：初始学习率，控制每次参数更新的步长。
- **learning_rate_decay**：每个 epoch 后学习率的衰减因子，用于逐渐降低学习率。
- **reg**：L2 正则化强度，用于惩罚大权重，帮助防止过拟合。
- **num_iters**：迭代的总次数，决定了训练过程中进行多少次参数更新。
- **batch_size**：每次训练时使用的批量样本数。
- **verbose**：是否在训练过程中打印进度信息（如 loss、训练准确率和验证准确率）。

这些参数可以帮助你控制模型的训练速度、稳定性和正则化效果。

# 4.模型文件下载

该实验在理想情况下训练所得的模型文件可通过以下链接获得：
