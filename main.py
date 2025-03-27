import argparse
import pickle
from dataset import load_CIFAR10
from model import ThreeLayerNet
from train import train, test
from hyperparameter_search import hyperparameter_tuning


def main():
    parser = argparse.ArgumentParser(description='Train an MLP on CIFAR-10')
    parser.add_argument('--data_dir', type=str, default='./cifar-10-batches-py', help='Path to CIFAR-10 data directory')
    parser.add_argument('--mode', type=str, choices=['train', 'hyperparam'], default='train',
                        help='运行模式：train 为普通训练，hyperparam 为超参数查找')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=0.95, help='Learning rate decay factor per epoch')
    parser.add_argument('--reg', type=float, default=5e-6, help='L2 regularization strength')
    parser.add_argument('--num_iters', type=int, default=1000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=200, help='Training batch size')
    parser.add_argument('--hidden_size', type=int, default=100, help='Hidden layer size')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid'],
                        help='Activation function')
    parser.add_argument('--verbose', action='store_true', help='Print training progress')
    # 新增保存模型文件的参数，指定保存路径（例如：model.pkl）
    parser.add_argument('--save_path', type=str, default=None, help='保存模型文件的路径')
    args = parser.parse_args()

    # 加载 CIFAR-10 数据
    X_train, y_train, X_test, y_test = load_CIFAR10(args.data_dir)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # 划分验证集（例如取前5000个作为验证集）
    num_val = 5000
    X_val = X_train[:num_val]
    y_val = y_train[:num_val]
    X_train_new = X_train[num_val:]
    y_train_new = y_train[num_val:]

    input_size = 32 * 32 * 3  # 3072
    output_size = 10  # 10 类

    if args.mode == 'train':
        # 普通训练模式
        net = ThreeLayerNet(input_size, args.hidden_size, output_size, activation=args.activation)
        stats = train(net, X_train_new, y_train_new, X_val, y_val,
                      learning_rate=args.learning_rate,
                      learning_rate_decay=args.learning_rate_decay,
                      reg=args.reg,
                      num_iters=args.num_iters,
                      batch_size=args.batch_size,
                      verbose=args.verbose)
        test(net, X_test, y_test)
        # 如果提供了保存路径，则将训练好的模型保存至文件
        if args.save_path is not None:
            with open(args.save_path, 'wb') as f:
                pickle.dump(net, f)
            print(f"模型已保存至 {args.save_path}")
    elif args.mode == 'hyperparam':
        # 超参数查找模式
        best_net, results = hyperparameter_tuning(X_train_new, y_train_new, X_val, y_val, input_size, output_size)
        print(test(best_net, X_test, y_test))
        # 如果提供了保存路径，则将最佳模型保存至文件
        if args.save_path is not None:
            with open(args.save_path, 'wb') as f:
                pickle.dump(best_net, f)
            print(f"最佳模型已保存至 {args.save_path}")


if __name__ == '__main__':
    main()
