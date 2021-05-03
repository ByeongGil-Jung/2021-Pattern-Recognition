"""
210503
07_manifold
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_digits
from sklearn.manifold import *
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

cmap = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'violet', 'lightblue']


# def mlp_embedding(X, y):
#     X = X / 16
#     t =  y.reshape((-1, 1))
#     encoder = OneHotEncoder()
#     t = encoder.fit_transform(t).toarray()
#
#     # ... (NN model)
#     mlp = None
#
#     h = mlp.layers[0](X)
#     h = mlp.layers[1](h, do_act=False)
#
#     fig = plt.figure()
#     scatter = plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
#     handles, labels = scatter.legend_elements()
#     plt.legend(handles, np.unique(y))
#     plt.title("MLP")
#     plt.savefig(f"./mlp.png")
#
#     fig.show()


def isomap(X, y):
    isomap = Isomap(n_neighbors=30)
    transformed_X = isomap.fit_transform(X=X)

    fig = plt.figure()
    scatter = plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title("Isomap")
    plt.savefig(f"./isomap.png")

    fig.show()


def lle(X, y):
    lle = LinearDiscriminantAnalysis()
    transformed_X = lle.fit_transform(X=X, y=y)

    fig = plt.figure()
    scatter = plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title("LLE")
    plt.savefig(f"./lle.png")

    fig.show()


def tsne(X, y):
    pca = TSNE(n_components=2)
    transformed_X = pca.fit_transform(X=X)

    fig = plt.figure()
    scatter = plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title("TSNE")
    plt.savefig(f"./tsne.png")

    fig.show()


def pca(X, y):
    pca = PCA(n_components=2)
    transformed_X = pca.fit_transform(X=X)

    fig = plt.figure()
    scatter = plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title("PCA")
    plt.savefig(f"./pca.png")

    fig.show()


def kpca(X, y):
    kpca = KernelPCA(n_components=2, kernel='cosine')
    transformed_X = kpca.fit_transform(X=X)
    fig = plt.figure()
    scatter = plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title("KernelPCA")
    plt.savefig(f"./kernel_pca.png")

    fig.show()


def lda(X, y):
    lda = LinearDiscriminantAnalysis(n_components=2)
    transformed_X = lda.fit_transform(X=X, y=y)

    fig = plt.figure()
    scatter = plt.scatter(transformed_X[:, 0], transformed_X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title("LDA")
    plt.savefig(f"./lda.png")

    fig.show()


def main():
    X, y = load_digits(return_X_y=True)
    ind = np.arange(len(X))
    np.random.shuffle(ind)
    X, y = X[ind], y[ind]

    isomap(X, y)
    lle(X, y)
    tsne(X, y)
    pca(X, y)
    kpca(X, y)
    lda(X, y)


if __name__ == "__main__":
    main()
