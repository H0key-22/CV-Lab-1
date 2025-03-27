#!/bin/bash
# run_train.sh
# 本脚本用于启动 CIFAR-10 上的 MLP 模型训练

# 设置数据目录（请确保 CIFAR-10 数据已经解压到该目录中）
DATA_DIR="./cifar-10-batches-py"
# 设置模型保存路径
SAVE_PATH="./best_model.pkl"

# 调用 main.py 脚本，并传入训练所需参数
python main.py --mode train \
               --data_dir "$DATA_DIR" \
               --learning_rate 0.1 \
               --learning_rate_decay 0.95 \
               --reg 5e-6 \
               --num_iters 10000 \
               --batch_size 200 \
               --hidden_size 384 \
               --activation relu \
               --verbose \
               --save_path "$SAVE_PATH"
