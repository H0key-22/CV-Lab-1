#!/bin/bash
# run_hyperparam.sh
# 本脚本用于在 CIFAR-10 数据集上进行 MLP 模型的超参数查找

# 设置数据目录（请确保 CIFAR-10 数据已经解压到该目录中）
DATA_DIR="./cifar-10-batches-py"

# 调用 main.py 脚本，并传入超参数查找模式及其他必要参数
python main.py --mode hyperparam \
               --data_dir "$DATA_DIR" \
               --verbose
