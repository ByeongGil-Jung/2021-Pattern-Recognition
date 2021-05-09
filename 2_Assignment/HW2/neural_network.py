"""
210509
HW2 - Neural Network
"""
from pathlib import Path
import pickle
import random
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


time.time()
RANDOM_SEED = 777

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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


class ReLU(Activation):  # ReLu
    def __init__(self):
        super(ReLU, self).__init__()

    def _function(self, x):
        f = np.maximum(0, x)
        return f

    def calc_gradients(self, x):
        f = self(x)

        grad = np.where(f > 0, 1, 0)

        return grad


class Tanh(Activation):  # Tanh
    def __init__(self):
        super(Tanh, self).__init__()

    def _function(self, x):
        f = 2. * (1. / (1. + np.exp(-x))) - 1
        return f

    def calc_gradients(self, x):
        f = self(x)
        grad = np.multiply(1. - f, 1. + f)

        return grad


class Identity(Activation):
    def __init__(self):
        super(Identity, self).__init__()

    def _function(self, x):
        f = x

        return f

    def calc_gradients(self, x):
        f = self(x)

        # print(f)

        grad = np.ones(shape=np.array(f).shape)
        # print(grad)
        # assert False, "break"

        return grad


# Loss functions
class CrossEntropy(LossFunction):

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def _function(self, y, t):
        y = y + 1e-24
        f = - np.sum(np.multiply(t, np.log(y)), axis=1)

        return f

    def calc_gradients(self, y, t):
        grad = - np.multiply(t, 1 / y)
        return grad


class MSELoss(LossFunction):
    def __init__(self):
        super(MSELoss, self).__init__()

    def _function(self, y, t):
        f = np.mean((t - y) ** 2, 1)

        return f

    def calc_gradients(self, y, t):
        t = t.reshape(y.shape[0], -1)

        grad = - 2 * (t - y) / y.shape[0]

        return grad


class MAELoss(LossFunction):
    def __init__(self):
        super(MAELoss, self).__init__()

    def _function(self, y, t):
        f = np.mean(np.abs(t - y))

        return f

    def calc_gradients(self, y, t):
        t = t.reshape(y.shape[0], -1)

        grad = - np.abs(t - y) / y.shape[0]

        return grad


class RMSELoss(LossFunction):

    def __init__(self):
        super(RMSELoss, self).__init__()

    def _function(self, y, t):
        f = np.sqrt(np.mean((t - y) ** 2))

        return f

    def calc_gradients(self, y, t):
        t = t.reshape(y.shape[0], -1)

        grad = - np.abs(t - y) * (1 / np.sqrt(y.shape[0]))

        return grad


class MAPELoss(LossFunction):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def _function(self, y, t):
        f = np.mean(np.abs((t - y) / t)) * 100
        return f

    def calc_gradients(self, y, t):
        t = t.reshape(y.shape[0], -1)

        grad = - np.abs(1 - y / t) * 100 / (t * y.shape[0])

        return grad


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

    def __init__(self, loss_function, learning_rate, layers=None):
        self.loss_function = loss_function()
        self.optimizer = GradientDescentOptimizer(learning_rate)

        self._build(layers=layers)

    def __call__(self, x):
        z = x

        for layer in self.layers:
            z = layer(z)

        return z

    # Forward
    def _build(self, layers):
        if not layers:
            layers = [
                DenseLayer(9, 16, Tanh, name="Layer_0"),
                DenseLayer(16, 8, Tanh, name="Layer_1"),
                DenseLayer(8, 1, Tanh, name="Layer_2"),
                DenseLayer(1, 1, Identity, name="Identity")
            ]

        self.layers = layers

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

    def calculate_loss(self, x, t):
        y = self(x)
        loss = self.loss_function(y, t)
        return np.mean(loss)

    def predict(self, x):
        y = self(x)
        return y


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


def train(model, train_X, train_y, val_X, val_y, batch_size=32, n_epoch=100):
    train_loss_list = []
    val_loss_list = []

    # Start
    for epoch in range(n_epoch):
        total_train_loss = 0
        total_val_loss = 0

        # Train
        train_step = int(np.ceil(len(train_X) / batch_size))

        for i in range(train_step):
            mb_X, mb_t = train_X[i * batch_size:(i + 1) * batch_size], train_y[i * batch_size:(i + 1) * batch_size]
            current_train_loss = model.fit(mb_X, mb_t)

            total_train_loss = total_train_loss + current_train_loss

        train_loss_list.append(total_train_loss / train_step)

        # Validation
        val_step = int(np.ceil(len(val_X) / batch_size))

        for j in range(val_step):
            val_mb_X, val_mb_t = val_X[j * batch_size:(j + 1) * batch_size], val_y[j * batch_size:(j + 1) * batch_size]
            current_val_loss = model.calculate_loss(val_mb_X, val_mb_t)

            total_val_loss = total_val_loss + current_val_loss

        val_loss_list.append(total_val_loss / val_step)

        # print(f"[Epoch {epoch}]")
        # print(f"Training - Loss : {total_train_loss / train_step}")
        # print(f"Val - Loss : {total_val_loss / val_step}")

    print("- Last Epoch")
    print(f"Training - Loss : {train_loss_list[-1]}")
    print(f"Validation - Loss : {val_loss_list[-1]}")

    return dict(
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list
    )


def test(model, test_X, test_y, batch_size=32):
    test_loss = 0

    # Test
    total_test_loss = 0
    test_step = int(np.ceil(len(test_X) / batch_size))

    for i in range(test_step):
        mb_X, mb_t = test_X[i * batch_size:(i + 1) * batch_size], test_y[i * batch_size:(i + 1) * batch_size]
        current_train_loss = model.fit(mb_X, mb_t)

        total_test_loss = total_test_loss + current_train_loss

    test_loss = total_test_loss / test_step

    print(f"Testing - Loss : {test_loss}")

    return dict(
        test_loss=test_loss
    )


def plot_result(train_result_dict: dict, model_name, is_saved=True):
    train_loss_list = train_result_dict["train_loss_list"]
    val_loss_list = train_result_dict["val_loss_list"]

    plt.figure(figsize=(15, 7))
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} :: Train-Validation Loss")
    plt.legend()

    if is_saved:
        plt.savefig(f"./neural_network/{model_name}_loss_plot.png")

    plt.show()


def dataset_statistic(df, is_saved=True):
    statistic_dict = dict()

    for column_name in df.columns:
        data = df.loc[:, column_name]

        count = len(data)
        mean = np.mean(data)
        std = np.std(data)
        min = np.min(data)
        max = np.max(data)
        median = np.median(data)
        per_25, per_50, per_75 = np.percentile(data, [25, 50, 75])

        statistic_dict[column_name] = dict(
            count=count,
            mean=mean,
            std=std,
            min=min,
            max=max,
            median=median,
            per_25=per_25,
            per_50=per_50,
            per_75=per_75
        )

    if is_saved:
        with open("./neural_network/dataset_statistic.pkl", "wb") as f:
            pickle.dump(statistic_dict, f, pickle.HIGHEST_PROTOCOL)

    return statistic_dict


def main():
    Path("./neural_network").mkdir(parents=True, exist_ok=True)
    dataset = fetch_california_housing()

    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    data = df.iloc[:, :13].values
    label = df['target'].values
    features = df.columns
    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.20, shuffle=True)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.25, shuffle=True)

    """
    Dataset Analysis
    """
    statistic_dict = dataset_statistic(df=df, is_saved=True)
    print(statistic_dict)

    """
    Activation Functions & Loss Functions Experiments
    """
    print(" === [ Activation Functions & Loss Functions Experiments ] === ")

    print(" === MAE Loss === ")
    print("> Sigmoid")
    model = MultiLayerPerceptron(loss_function=MAELoss, learning_rate=1e-5, layers=[
        DenseLayer(9, 16, Sigmoid, name="Layer_0"),
        DenseLayer(16, 8, Sigmoid, name="Layer_1"),
        DenseLayer(8, 1, Sigmoid, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=200)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MAE_Sigmoid", is_saved=True)

    print("> ReLU")
    model = MultiLayerPerceptron(loss_function=MAELoss, learning_rate=1e-5, layers=[
        DenseLayer(9, 16, ReLU, name="Layer_0"),
        DenseLayer(16, 8, ReLU, name="Layer_1"),
        DenseLayer(8, 1, ReLU, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=400)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MAE_ReLU", is_saved=True)

    print("> Tanh")
    model = MultiLayerPerceptron(loss_function=MAELoss, learning_rate=1e-5, layers=[
        DenseLayer(9, 16, Tanh, name="Layer_0"),
        DenseLayer(16, 8, Tanh, name="Layer_1"),
        DenseLayer(8, 1, Tanh, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=200)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MAE_Tanh", is_saved=True)

    print("> Identity")
    model = MultiLayerPerceptron(loss_function=MAELoss, learning_rate=1e-11, layers=[
        DenseLayer(9, 16, Identity, name="Layer_0"),
        DenseLayer(16, 8, Identity, name="Layer_1"),
        DenseLayer(8, 1, Identity, name="Layer_2")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=50)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MAE_Identity", is_saved=True)

    print(" === MSE Loss === ")
    print("> Sigmoid")
    model = MultiLayerPerceptron(loss_function=MSELoss, learning_rate=1e-4, layers=[
        DenseLayer(9, 16, Sigmoid, name="Layer_0"),
        DenseLayer(16, 8, Sigmoid, name="Layer_1"),
        DenseLayer(8, 1, Sigmoid, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=150)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MSE_Sigmoid", is_saved=True)

    print("> ReLU")
    model = MultiLayerPerceptron(loss_function=MSELoss, learning_rate=1e-5, layers=[
        DenseLayer(9, 16, ReLU, name="Layer_0"),
        DenseLayer(16, 8, ReLU, name="Layer_1"),
        DenseLayer(8, 1, ReLU, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=200)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MSE_ReLU", is_saved=True)

    print("> Tanh")
    model = MultiLayerPerceptron(loss_function=MSELoss, learning_rate=1e-4, layers=[
        DenseLayer(9, 16, Tanh, name="Layer_0"),
        DenseLayer(16, 8, Tanh, name="Layer_1"),
        DenseLayer(8, 1, Tanh, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=100)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MSE_Tanh", is_saved=True)

    print("> Identity")
    model = MultiLayerPerceptron(loss_function=MSELoss, learning_rate=1e-11, layers=[
        DenseLayer(9, 16, Identity, name="Layer_0"),
        DenseLayer(16, 8, Identity, name="Layer_1"),
        DenseLayer(8, 1, Identity, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=100)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MSE_Identity", is_saved=True)

    print(" === RMSE Loss === ")
    print("> Sigmoid")
    model = MultiLayerPerceptron(loss_function=RMSELoss, learning_rate=1e-6, layers=[
        DenseLayer(9, 16, Sigmoid, name="Layer_0"),
        DenseLayer(16, 8, Sigmoid, name="Layer_1"),
        DenseLayer(8, 1, Sigmoid, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=16,
                              n_epoch=400)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="RMSE_Sigmoid", is_saved=True)

    print("> ReLU")
    model = MultiLayerPerceptron(loss_function=RMSELoss, learning_rate=1e-5, layers=[
        DenseLayer(9, 16, ReLU, name="Layer_0"),
        DenseLayer(16, 8, ReLU, name="Layer_1"),
        DenseLayer(8, 1, ReLU, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=80)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="RMSE_ReLU", is_saved=True)

    print("> Tanh")
    model = MultiLayerPerceptron(loss_function=RMSELoss, learning_rate=1e-6, layers=[
        DenseLayer(9, 16, Tanh, name="Layer_0"),
        DenseLayer(16, 8, Tanh, name="Layer_1"),
        DenseLayer(8, 1, Tanh, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=400)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="RMSE_Tanh", is_saved=True)

    print("> Identity")
    model = MultiLayerPerceptron(loss_function=RMSELoss, learning_rate=1e-12, layers=[
        DenseLayer(9, 16, Identity, name="Layer_0"),
        DenseLayer(16, 8, Identity, name="Layer_1"),
        DenseLayer(8, 1, Identity, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=16,
                              n_epoch=150)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="RMSE_Identity", is_saved=True)

    print(" === MAPE Loss === ")
    print("> Sigmoid")
    model = MultiLayerPerceptron(loss_function=MAPELoss, learning_rate=1e-7, layers=[
        DenseLayer(9, 16, Sigmoid, name="Layer_0"),
        DenseLayer(16, 8, Sigmoid, name="Layer_1"),
        DenseLayer(8, 1, Sigmoid, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=16,
                              n_epoch=275)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MAPE_Sigmoid", is_saved=True)

    print("> ReLU")
    model = MultiLayerPerceptron(loss_function=MAPELoss, learning_rate=1e-9, layers=[
        DenseLayer(9, 16, ReLU, name="Layer_0"),
        DenseLayer(16, 8, ReLU, name="Layer_1"),
        DenseLayer(8, 1, ReLU, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=200)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MAPE_ReLU", is_saved=True)

    print("> Tanh")
    model = MultiLayerPerceptron(loss_function=MAPELoss, learning_rate=1e-8, layers=[
        DenseLayer(9, 16, Tanh, name="Layer_0"),
        DenseLayer(16, 8, Tanh, name="Layer_1"),
        DenseLayer(8, 1, Tanh, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=32,
                              n_epoch=500)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MAPE_Tanh", is_saved=True)

    print("> Identity")
    model = MultiLayerPerceptron(loss_function=MAPELoss, learning_rate=1e-11, layers=[
        DenseLayer(9, 16, Identity, name="Layer_0"),
        DenseLayer(16, 8, Identity, name="Layer_1"),
        DenseLayer(8, 1, Identity, name="Layer_2"),
        DenseLayer(1, 1, Identity, name="Identity")
    ])
    train_result_dict = train(model=model, train_X=train_X, train_y=train_y, val_X=val_X, val_y=val_y, batch_size=16,
                              n_epoch=10)
    test_result_dict = test(model=model, test_X=test_X, test_y=test_y, batch_size=32)
    plot_result(train_result_dict=train_result_dict, model_name="MAPE_Identity", is_saved=True)


if __name__ == "__main__":
    main()
