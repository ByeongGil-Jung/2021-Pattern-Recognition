"""
210426
06_PCA (Generate random value)
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA


def do_pca(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # X -> covariance -> eigen decomposition (O(n^3)) -> lambda (eigenvalue), principal component (eigenvector)
    pc, mean_orig, cov = pca.components_, pca.mean_, np.diag(pca.explained_variance_)  # cov 는 eigenvalue 임 (크면 클 수록 데이터를 설명하는 중요한 feature 임)

    transformed_X = pca.transform(X)  # (?, n_components)
    transformed_mean = np.mean(transformed_X, axis=0)  # (n_components, ) // 모든 데이터에 대한 평균

    return transformed_mean, pc, mean_orig, cov


def generate_random_data(transformed_mean, pc, mean_orig, cov):
    # mean 과 cov 를 알고 있으로 gaussian dist 를 만들수 있음 (참고로 cov 는 서로 수직임)
    f_rand = np.random.multivariate_normal(mean=transformed_mean, cov=cov, size=10)

    # 원본 데이터의 도메인으로 돌아감, 뒤에 mean_orig 를 더하는 이유는
    # 원본 데이터의 평균을 더해줌으로써 복원을 가능하게 하기 위함 (역으로)

    # 원본 PCA transformation 은 X_tf = (X - m) * PC 이므로
    # X = X-tf * PC^T + m 이기 때문임

    # PCA 를 역으로 계산하는 과정으로
    generated_X = np.matmul(f_rand, pc) + mean_orig.reshape((1, -1))

    return generated_X


def main():
    # (70000, 784)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    idx = np.arange(len(X))
    np.random.shuffle(idx)  # Shuffle index

    X = X[idx]
    X = X[:30000]

    print(X.shape)  # (30000, 784)

    # 784 -> 4
    transformed_mean, pc, mean_orig, cov = do_pca(X=X, n_components=4)
    generated_X = generate_random_data(transformed_mean, pc, mean_orig, cov)

    # Visualization
    fig, ax = plt.subplots(1, 10)

    for i in range(10):
        ax[i].imshow(generated_X[i].reshape((28, 28)), cmap='gray')

    plt.savefig(f"./generated_data_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()
