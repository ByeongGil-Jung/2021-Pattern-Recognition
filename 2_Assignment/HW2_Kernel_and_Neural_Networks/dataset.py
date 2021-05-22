import random
import time

import numpy as np
import matplotlib.pyplot as plt


time.time()
RANDOM_SEED = 777

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class MGDDataset(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __repr__(self):
        return f"{self.X}"

    def __len__(self):
        return len(self.y)

    @classmethod
    def generate_class_1_dataset(cls, n=1000, is_shuffle=True):
        def __generate_data(n):
            data_1, data_2 = 0, 0

            for i in range(n):
                if np.random.rand() >= 0.5:
                    data_1 += 1
                else:
                    data_2 += 1

            return data_1, data_2

        data_1, data_2 = __generate_data(n=n)

        mean_vector_1 = [2, 2]
        mean_vector_2 = [-2, -2]

        cov_mtx_1 = [[0.5, 0],
                     [0, 0.5]]
        cov_mtx_2 = [[0.5, 0],
                     [0, 0.5]]

        mgd1 = np.random.multivariate_normal(mean=mean_vector_1, cov=cov_mtx_1, size=data_1)
        mgd2 = np.random.multivariate_normal(mean=mean_vector_2, cov=cov_mtx_2, size=data_2)

        mgd = np.concatenate([mgd1, mgd2])

        if is_shuffle:
            np.random.shuffle(mgd)

        return MGDDataset(X=mgd, y=np.zeros(shape=(n,)))

    @classmethod
    def generate_class_2_dataset(cls, n=1000, is_shuffle=True):
        mean_vector = [0, 0]
        cov_mtx = [[1, -0.5],
                   [-0.5, 1]]

        mgd = np.random.multivariate_normal(mean=mean_vector, cov=cov_mtx, size=n)

        if is_shuffle:
            np.random.shuffle(mgd)

        return MGDDataset(X=mgd, y=np.ones(shape=(n,)))

    @classmethod
    def generate_entire_dataset(cls, cls1_dataset_n=1000, cls2_dataset_n=1000, is_shuffle=True):
        mgd_cls1 = MGDDataset.generate_class_1_dataset(n=cls1_dataset_n, is_shuffle=is_shuffle)
        mgd_cls2 = MGDDataset.generate_class_2_dataset(n=cls2_dataset_n, is_shuffle=is_shuffle)

        X = np.concatenate([mgd_cls1.X, mgd_cls2.X])
        y = np.concatenate([mgd_cls1.y, mgd_cls2.y])

        if is_shuffle:
            shuffle_index_list = np.arange(X.shape[0])
            np.random.shuffle(shuffle_index_list)

            X = X[shuffle_index_list]
            y = y[shuffle_index_list]

        return MGDDataset(X=X, y=y)

    def train_test_split(self, test_ratio):
        data_size = len(self.y)
        test_size = int(test_ratio * data_size)

        return dict(
            train_dataset=MGDDataset(X=self.X[test_size:], y=self.y[test_size:]),
            val_dataset=MGDDataset(X=self.X[:test_size], y=self.y[:test_size])
        )

    def convert_label_zero_to_neg(self):
        self.y = np.where(self.y == 0, -1, 1)

    def filter_by_class(self, cls):
        X_filtered = list()

        for x, y in zip(self.X, self.y):
            if y == cls:
                X_filtered.append(x)

        return np.array(X_filtered)

    def filter_by_pred(self, pred_y_list, cls):
        X_filtered = list()

        for x, y in zip(self.X, pred_y_list):
            if y == cls:
                X_filtered.append(x)

        return np.array(X_filtered)

