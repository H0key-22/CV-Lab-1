import numpy as np


class ThreeLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, activation='relu', std=1e-4):
        """
        初始化模型参数。
        参数：
          - input_size: 输入维度（例如 3072）
          - hidden_size: 隐藏层大小
          - output_size: 输出类别数（例如 10）
          - activation: 激活函数类型，支持 'relu'、'sigmoid'、'tanh'、'leaky_relu'
          - std: 权重初始化标准差
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        self.activation = activation

    def forward(self, X):
        # 第一层：仿射变换
        hidden = X.dot(self.params['W1']) + self.params['b1']
        # 激活函数处理
        if self.activation == 'relu':
            hidden_act = np.maximum(0, hidden)
        elif self.activation == 'sigmoid':
            hidden_act = 1 / (1 + np.exp(-hidden))
        elif self.activation == 'tanh':
            hidden_act = np.tanh(hidden)
        elif self.activation == 'leaky_relu':
            hidden_act = np.where(hidden > 0, hidden, 0.01 * hidden)
        else:
            raise ValueError("Unsupported activation function: " + self.activation)
        # 输出层
        scores = hidden_act.dot(self.params['W2']) + self.params['b2']
        return hidden, hidden_act, scores

    def loss(self, X, y=None, reg=0.0):
        """
        计算损失和梯度。如果 y 为 None，则返回得分；否则返回 (loss, grads)。
        """
        hidden, hidden_act, scores = self.forward(X)
        if y is None:
            return scores

        # 计算 softmax 损失
        shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        N = X.shape[0]
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        loss += reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))

        # 反向传播
        grads = {}
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N

        # 输出层梯度
        grads['W2'] = hidden_act.T.dot(dscores) + 2 * reg * self.params['W2']
        grads['b2'] = np.sum(dscores, axis=0)

        # 反向传播到隐藏层
        dhidden = dscores.dot(self.params['W2'].T)
        # 激活函数的梯度
        if self.activation == 'relu':
            dhidden[hidden <= 0] = 0
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-hidden))
            dhidden *= s * (1 - s)
        elif self.activation == 'tanh':
            dhidden *= (1 - np.tanh(hidden) ** 2)
        elif self.activation == 'leaky_relu':
            dhidden[hidden <= 0] *= 0.01
        grads['W1'] = X.T.dot(dhidden) + 2 * reg * self.params['W1']
        grads['b1'] = np.sum(dhidden, axis=0)

        return loss, grads

    def predict(self, X):
        """
        利用当前模型参数预测标签。
        """
        _, hidden_act, scores = self.forward(X)
        return np.argmax(scores, axis=1)