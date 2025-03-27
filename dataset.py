import numpy as np
import pickle
import os

def load_CIFAR_batch(filename):
    """加载 CIFAR-10 单个 batch 数据"""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        # reshape 成 (N, 32, 32, 3)
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(root):
    """加载 CIFAR-10 所有数据"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    X_test, Y_test = load_CIFAR_batch(os.path.join(root, 'test_batch'))
    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_CIFAR10('./cifar-10-batches-py')
    print(f'Training data shape: {X_train.shape}')
    print(f'Training labels shape: {y_train.shape}')

