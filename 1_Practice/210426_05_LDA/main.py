"""
210426
05_LDA
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def load_data():
    X, y = load_digits(return_X_y=True)

    idx = np.arange(len(X))
    np.random.shuffle(idx)  # Shuffle index

    X, y = X[idx], y[idx]

    TR_SIZE = int(len(X) * 3 / 4)
    train_X, test_X = X[:TR_SIZE], X[TR_SIZE:]
    train_y, test_y = y[:TR_SIZE], y[TR_SIZE:]

    return train_X, train_y, test_X, test_y


def transform(train_X, train_y, test_X, test_y, transfrom_method):
    transformer = None

    if transfrom_method.lower() == "lda":
        transformer = LinearDiscriminantAnalysis(n_components=2)  # 몇 차원까지 축소할 것인지(n_components)
    if transfrom_method.lower() == "pca":
        transformer = PCA(n_components=2)

    transformer.fit(train_X, train_y)

    transformed_train_X = transformer.transform(X=train_X)
    transformed_test_X = transformer.transform(X=test_X)

    # Visualization
    fig, ax = plt.subplots(1, 5)
    for i in range(5):
        ax[i].imshow(train_X[i].reshape((8, 8)), cmap='gray')

    plt.savefig("./data_visualization.png")
    fig.show()

    fig = plt.figure()
    color_map = {
        0: 'r',
        1: 'g',
        2: 'b',
        3: 'c',
        4: 'm',
        5: 'y',
        6: 'k',
        7: 'grey',
        8: 'lightblue',
        9: 'violet'
    }

    for i in range(10):
        idx = np.where(train_y == i)[0]
        plt.scatter(transformed_train_X[idx, 0], transformed_train_X[idx, 1], c=color_map[i], label=str(i))

    plt.legend()
    plt.savefig(f"./{transfrom_method}_transformed_2d_plot.png")
    fig.show()

    return transformed_train_X, transformed_test_X


def classify(train_X, train_y, test_X, test_y, title):
    print(title)

    clf = KNeighborsClassifier(n_neighbors=5)  # 참고로 n_neighbors 는 홀수 (클래스간 동수가 나오지 않게 하기 위해)
    clf.fit(X=train_X, y=train_y)

    print("[-] Accuracy:", clf.score(X=test_X, y=test_y))
    print()


def main():
    train_X, train_y, test_X, test_y = load_data()  # (?, 64), (?, )

    print("== LDA ==")
    transformed_train_X, transformed_test_X = transform(train_X, train_y, test_X, test_y, transfrom_method="lda")

    classify(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, title="Original")
    classify(train_X=transformed_train_X, train_y=train_y, test_X=transformed_test_X, test_y=test_y, title="LDA")

    print("== PCA ==")
    transformed_train_X, transformed_test_X = transform(train_X, train_y, test_X, test_y, transfrom_method="pca")

    classify(train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, title="Original")
    classify(train_X=transformed_train_X, train_y=train_y, test_X=transformed_test_X, test_y=test_y, title="PCA")

    plt.show()


if __name__ == "__main__":
    main()
