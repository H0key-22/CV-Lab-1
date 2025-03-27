from model import ThreeLayerNet
from train import train

def hyperparameter_tuning(X_train, y_train, X_val, y_val, input_size, output_size):
    best_val = -1
    best_net = None
    results = {}

    learning_rates = [1e-2, 5e-2, 1e-1]
    hidden_sizes = [96, 192, 384]
    reg_strengths = [0, 1e-3, 5e-3, 1e-2]

    for lr in learning_rates:
        for hs in hidden_sizes:
            for reg in reg_strengths:
                net = ThreeLayerNet(input_size, hs, output_size, activation='relu')
                stats = train(net,
                              X_train, y_train, X_val, y_val,
                              learning_rate=lr, reg=reg,
                              num_iters=5000, batch_size=200, verbose=False)
                val_acc = (net.predict(X_val.reshape(X_val.shape[0], -1)) == y_val).mean()
                results[(lr, hs, reg)] = val_acc
                print(f'lr {lr}, hidden_size {hs}, reg {reg} => val accuracy: {val_acc:.4f}')
                if val_acc > best_val:
                    best_val = val_acc
                    best_net = net

    print(f'Best validation accuracy: {best_val:.4f}')
    return best_net, results
