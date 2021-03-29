"""
210329
02_Linear_Regression
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


def main():
    num_data = 500

    X = np.random.uniform(low=-5, high=5, size=num_data)
    X.sort()
    print(X)
    print(f"Shape : {X.shape}")
    X = np.expand_dims(X, 1)
    print(f"Shape (after expand_dims) : {X.shape}")

    # y = a*x + b + e
    t = 0.5 * X - 1 + np.random.normal(size=[num_data, 1])  # Noise 고려
    print(f"X shape : {X.shape} | t shape : {t.shape}")

    # Figure
    fig = plt.figure()
    plt.scatter(X, t, color='b')
    plt.show()

    # Split train data and val data
    train_size = int(len(X) * 3 / 4)
    train_X, val_X = X[:train_size], X[train_size:]
    train_t, val_t = t[:train_size], t[train_size:]

    model = LinearRegression()

    print("Start to fit ...")
    model.fit(X=train_X, y=train_t)
    print("Finished fitting ...")

    print(f"Coef (기울기) : {model.coef_} | Intercept (절편) : {model.intercept_}")

    fig = plt.figure()
    plt.scatter(train_X[:, 0], train_t[:, 0], color='b', label="train")
    plt.scatter(val_X[:, 0], val_t[:, 0], color='r', label="val")

    x_space = np.expand_dims(np.linspace(-5, 5, num_data), 1)
    y_p = model.predict(x_space)
    y = 0.5 * x_space - 1
    plt.plot(x_space[:, 0], y[:, 0], color='b', label="True")
    plt.plot(x_space[:, 0], y_p[:, 0], color='r', label="Predict")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
