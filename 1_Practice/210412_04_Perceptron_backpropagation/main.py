"""
210412
04_Perceptron
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder


# Base class
class Activation(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return self._function(x=x)

    def _function(self, x):
        return 0

    def calc_gradients(self, x):
        return 0


class LossFunction(object):

    def __init__(self):
        pass

    def __call__(self, y, t):
        return self._function(y=y, t=t)

    def _function(self, y, t):
        return 0

    def calc_gradients(self, y, t):
        return 0


# Activation functions
class Sigmoid(Activation):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def _function(self, x):
        f = 1 / (1 + np.exp(-x))

        return f

    def calc_gradients(self, x):
        f = self(x)

        return np.multiply(f, 1 - f)


class Softmax(Activation):

    def __init__(self):
        super(Softmax, self).__init__()

    def _function(self, x):
        exps = np.exp(x - np.max(x, 1, keepdims=True))
        f = exps / np.sum(exps, 1, keepdims=True)

        return f

    def calc_gradients(self, x):
        # can be more simpler....
        # diag(i=j): f_i * (1 - f_j)
        # non-diag(i!=j): -f_i*f_j
        f = self(x)
        g = np.zeros((x.shape[0], x.shape[1], x.shape[1]))  # gradient 를 담을 matrix

        diag = np.multiply(f, 1 - f)

        # Diagonal 일 경우
        for i in range(x.shape[1]):
            g[:, i, i] = diag[:, i]
        # Non-diagnomal 일 경우
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                if i == j:
                    continue
                g[:, i, j] = - np.multiply(f[:, i], f[:, j])

        return g


# Loss functions
class CrossEntropy(LossFunction):

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def _function(self, y, t):
        y = y + 1e-24
        f = - np.sum(np.multiply(t, np.log(y)), axis=1)

        return f

    def calc_gradients(self, y, t):
        return - np.multiply(t, 1 / y)


# Optimizer
class GradientDescentOptimizer(object):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_gradients(self, layer):
        for i in range(len(layer.trainable_variables)):
            layer.trainable_variables[i] = layer.trainable_variables[i] - self.learning_rate * layer.gradients[i]

        layer.gradients = None  # 계산했으므로 필요 없으므로


# Model
class DenseLayer(object):

    def __init__(self, n_in, n_out, activation=None, name=None):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation()
        self.name = name
        self.trainable_variables = None  # Weights
        self.gradients = None

        self._build()

    def __call__(self, x):  # Forward 함수
        # x = m * d
        self.x = x
        self.a = np.matmul(self.x, self.trainable_variables[0]) + self.trainable_variables[1]
        return self.activation(self.a)

    def _build(self):
        self.trainable_variables = [
            np.random.normal(scale=np.sqrt(2 / (self.n_in + self.n_out)), size=(self.n_in, self.n_out)),  # Xavier init
            np.zeros((1, self.n_out))
        ]

    def calc_gradients(self, g_high):  # input : 상위 layer 의 gradient
        # g_high = m * 10
        # g_activation = m * 10 * 10
        g_activation = self.activation.calc_gradients(self.a)  # activation 에 들어가기 전의 gradients 를 넣어야 함

        if isinstance(self.activation, Softmax):
            g_high = np.expand_dims(g_high, 1)  # m * 1 * 10
            delta = np.multiply(g_high, g_activation)  # delta = m * 10 * 10
            delta = np.sum(delta, axis=2)  # delta = m * 10
        else:
            delta = np.multiply(g_high, g_activation)

        self.g_W = np.matmul(self.x.T, delta)
        self.g_b = np.sum(delta, 0, keepdims=True)
        self.gradients = [self.g_W, self.g_b]

        # 여기서부턴 밑으로 gradient 를 전달
        # (현재 layer 의 delta 값에 해당 layer 의 weight 를 곱해서 전달하게 됨)

        # 여기서 일괄적으로 weight 를 업데이트 하게 됨
        return np.matmul(delta, self.trainable_variables[0].T)


class MultiLayerPerceptron(object):

    def __init__(self, loss_function, learning_rate):
        self.loss_function = loss_function()
        self.optimizer = GradientDescentOptimizer(learning_rate)

        self._build()

    def __call__(self, x):
        z = x

        for layer in self.layers:
            z = layer(z)

        return z

    # Forward
    def _build(self):
        self.layers = [
            DenseLayer(n_in=64, n_out=100, activation=Sigmoid, name="Layer1"),
            DenseLayer(n_in=100, n_out=100, activation=Sigmoid, name="Layer2"),
            DenseLayer(n_in=100, n_out=10, activation=Softmax, name="Layer3")
        ]

    # Backprop
    def fit(self, x, t):
        y = self(x)
        loss = self.loss_function(y, t)

        g = self.loss_function.calc_gradients(y, t)
        
        for layer in self.layers[::-1]:  # 역순 순회
            g = layer.calc_gradients(g)

        for layer in self.layers:
            self.optimizer.apply_gradients(layer)

        return np.mean(loss)

    def accuracy(self, x, t):
        y = self(x)
        accuracy = np.mean(np.equal(np.argmax(y, 1), np.argmax(t, 1)).astype(np.float32))

        return accuracy


def split(data):
    X = data.X / 16  # 값의 범위를 0-1 로 normalize
    y = data.target.reshape((-1, 1))

    # 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    encoder = OneHotEncoder()
    y = encoder.fit_transform(y).toarray()

    size_fold = int(len(X) / 5)
    train_X, val_X, test_X = X[:size_fold * 3], X[size_fold * 3:size_fold * 4], X[size_fold * 4:]
    train_y, val_y, test_y = y[:size_fold * 3], y[size_fold * 3:size_fold * 4], y[size_fold * 4:]

    return train_X, train_y, val_X, val_y, test_X, test_y


def main():
    data = load_digits()
    train_X, train_y, val_X, val_y, test_X, test_y = split(data)

    mlp = MultiLayerPerceptron(loss_function=CrossEntropy, learning_rate=1e-2)

    # Plotting
    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(7)
    fig.set_figwidth(15)

    # Train
    loss_list, train_acc_list, val_acc_list = list(), list(), list()
    n_batch = 100
    n_epoch = 20

    for e in range(n_epoch):
        for i in range(int(np.ceil(len(train_X) / n_batch))):
            X_batch, y_batch = train_X[i * n_batch:(i + 1) * n_batch], train_y[i * n_batch:(i + 1)* n_batch]

            loss = mlp.fit(X_batch, y_batch)  # Fit
            train_acc = mlp.accuracy(X_batch, y_batch)
            val_acc = mlp.accuracy(val_X, val_y)

            loss_list.append(loss)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)

            print("- epoch : %d, step: %d, loss: %f, tr ac: %f, val acc: %f" % (e, i, loss, train_acc, val_acc))

            # Plotting
            ax[0].cla()
            ax[1].cla()

            ax[0].plot(loss_list, label="Loss", c='r')
            ax[1].plot(train_acc_list, label="Train Acc", c='b')
            ax[1].plot(val_acc_list, label="Val Acc", c='r')

            fig.canvas.draw()
            plt.pause(0.001)

        test_acc = mlp.accuracy(test_X, test_y)
        print("* end of epoch(%d), te acc: %f" % (e, test_acc))
        print()

    # Plotting
    ax[0].set_title("Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")

    plt.legend()
    plt.savefig("./loss.png")

    fig.show()
    plt.show()


if __name__ == "__main__":
    main()
