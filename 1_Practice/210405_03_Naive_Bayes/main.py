"""
210329
03_Naive_Bayes
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB


def plot(data):
    X = pd.DataFrame(data.X, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=["species"])

    print(f"pd_iloc, \n"
          f"X : {X.iloc[:3]} \n"
          f"y: {y.iloc[:3]}")

    pd.plotting.scatter_matrix(X, c=y["species"], diagonal='kde')
    iris_data = pd.concat([X, y], axis=1)  # (150, 4), (150,) -> (150, 5)
    iris_data["species"] = iris_data["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    sns.pairplot(iris_data, hue="species")  # 여기서 3, 4번째 feature 를 이용하면 분류가 잘 될 것이라 추측

    plt.savefig("./data_plot.png")
    plt.show()


def naive_bayes_partial_features(data):
    print(f"Type of data : {type(data)}")

    # Set data
    X = data.X[:, [2, 3]]  # 3, 4 번 feature 사용
    y = data.target  # 0, 1, 2

    print(f"Shape X : {X.shape} y: {y.shape}")
    print(f"X : \n{X[:3]} (상위 3개)")
    print(f"y : \n{y[:3]} (상위 3개)")

    index_list = np.arange(len(X))
    np.random.shuffle(index_list)

    X = X[index_list]
    y = y[index_list]

    # Training
    train_size = int(len(X) * 3 / 4)
    train_X, val_X = X[:train_size], X[train_size:]
    train_y, val_y = y[:train_size], y[train_size:]

    clf = GaussianNB()  # Naive Bayes 에 사용할 확률분포는 gaussian
    clf.fit(X=train_X, y=train_y)

    pred_val_y = clf.predict(X=val_X)
    score = clf.score(X=val_X, y=val_y)
    print(f"Score (val): {score}")

    # Plotting
    fig, ax = plt.subplots(1, 2)
    min_f1, max_f1 = np.min(X[:, 0]), np.max(X[:, 0])
    min_f2, max_f2 = np.min(X[:, 1]), np.max(X[:, 1])

    f1_space = np.linspace(min_f1, max_f1, 100)
    f2_space = np.linspace(min_f2, max_f2, 100)

    # subplot 를 scatter 형식으로 뿌리기 위함 (경계 사이를 부드럽게 표현하기 위함)
    f1f1, f2f2 = np.meshgrid(f1_space, f2_space)  # (100, 100), (100, 100)

    feature_space = np.stack([f1f1, f2f2], axis=2)  # (100, 100, 2)
    feature_space = feature_space.reshape((-1, 2))  # (10000, 2)

    prob_feature_space = clf.predict_proba(feature_space)
    prob_feature_space = prob_feature_space.reshape((100, 100, 3))

    ax[0].imshow(
        prob_feature_space,
        extent=(f1f1.min(), f1f1.max(), f2f2.min(), f2f2.max()),
        origin="lower",
        aspect="auto",
        alpha=0.3
    )
    ax[1].imshow(
        prob_feature_space,
        extent=(f1f1.min(), f1f1.max(), f2f2.min(), f2f2.max()),
        origin="lower",
        aspect="auto",
        alpha=0.3
    )

    ax[0].scatter(
        val_X[:, 0],
        val_X[:, 1],
        c=val_y,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("cmap", colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    )
    ax[1].scatter(
        val_X[:, 0],
        val_X[:, 1],
        c=pred_val_y,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("cmap", colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    )

    ax[0].set_title("Ground Truth")
    ax[1].set_title("Prediction")

    plt.savefig("./naive_bayes_plot.png")
    fig.show()
    plt.show()


def main():
    data = load_iris()

    plot(data=data)
    naive_bayes_partial_features(data=data)


if __name__ == "__main__":
    main()
