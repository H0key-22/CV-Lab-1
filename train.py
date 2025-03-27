import numpy as np


def train(net, X_train, y_train, X_val, y_val,
          learning_rate=1e-3, learning_rate_decay=0.95,
          reg=5e-6, num_iters=1000, batch_size=200, verbose=False):
    """
    训练网络。
    参数：
      - net: ThreeLayerNet 实例
      - X_train, y_train: 训练数据及标签
      - X_val, y_val: 验证数据及标签
      - learning_rate: 学习率
      - learning_rate_decay: 每个 epoch 后学习率衰减因子
      - reg: L2 正则化强度
      - num_iters: 迭代次数
      - batch_size: 批量大小
      - verbose: 是否打印训练过程信息
    返回：
      - 一个包含 loss 历史、训练准确率和验证准确率的字典
    """
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_val_acc = 0
    best_params = None

    for it in range(num_iters):
        # 随机抽取小批量样本
        batch_idx = np.random.choice(num_train, batch_size, replace=True)
        X_batch = X_train[batch_idx].reshape(batch_size, -1)  # 将图像展平为向量
        y_batch = y_train[batch_idx]

        # 计算损失和梯度
        loss, grads = net.loss(X_batch, y=y_batch, reg=reg)
        loss_history.append(loss)

        # 使用 SGD 更新参数
        for param in net.params:
            net.params[param] -= learning_rate * grads[param]

        # 每个 epoch 评估准确率，并进行学习率衰减
        if it % iterations_per_epoch == 0:
            train_acc = (net.predict(X_batch) == y_batch).mean()
            # 验证集样本同样需要展平
            val_acc = (net.predict(X_val.reshape(X_val.shape[0], -1)) == y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            if verbose:
                print(f'Iteration {it}/{num_iters}: loss {loss:.4f}, train acc {train_acc:.4f}, val acc {val_acc:.4f}')

            # 保存最佳模型参数
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {k: v.copy() for k, v in net.params.items()}

            learning_rate *= learning_rate_decay

    # 将最佳参数赋回网络
    if best_params is not None:
        net.params = best_params
    return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
    }

def test(net, X_test, y_test):
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    y_pred = net.predict(X_test_flat)
    test_acc = np.mean(y_pred == y_test)
    print(f'Test accuracy: {test_acc:.4f}')
    return test_acc