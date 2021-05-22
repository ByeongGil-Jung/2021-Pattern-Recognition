"""
210509
HW2 - RBF_Network
"""
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from dataset import MGDDataset


time.time()
RANDOM_SEED = 777

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


"""
Author: Octavio Arriaga
Github: https://github.com/oarriaga
"""
class RBFN(object):

    def __init__(self, hidden_shape, sigma=1.0):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma*np.linalg.norm(center-data_point)**2)

    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                        center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions


def train(train_dataset, val_dataset, hidden_shape=75, sigma=1.0):
    clf = RBFN(hidden_shape=hidden_shape, sigma=sigma)
    clf.fit(X=train_dataset.X, Y=train_dataset.y)

    pred_list = clf.predict(X=val_dataset.X)

    transformed_pred_list = [1 if pred >= 0 else -1 for pred in pred_list]

    metrics_dict = classification_report(
        y_true=val_dataset.y,
        y_pred=transformed_pred_list,
        target_names=["class_1", "class_2"],
        output_dict=True
    )
    print(classification_report(
        y_true=val_dataset.y,
        y_pred=transformed_pred_list,
        target_names=["class_1", "class_2"],
        output_dict=False
    ))

    return dict(
        classifier=clf,
        pred_list=pred_list,
        metrics=metrics_dict
    )


def plot_decision_boundary(clf, dataset, ax):
    min_f1, max_f1 = np.min(dataset.X[:, 0]), np.max(dataset.X[:, 0])
    min_f2, max_f2 = np.min(dataset.X[:, 1]), np.max(dataset.X[:, 1])

    f1_space = np.linspace(min_f1, max_f1, 100)
    f2_space = np.linspace(min_f2, max_f2, 100)

    f1f1, f2f2 = np.meshgrid(f1_space, f2_space)

    feature_space = np.vstack([f1f1.ravel(), f2f2.ravel()]).T
    prob_feature_space = clf.predict(feature_space).reshape(f1f1.shape)

    ax.contour(f1f1, f2f2, prob_feature_space * 2, colors='k', levels=1, alpha=0.5, linestypes=['--','-','--'])
    ax.pcolormesh(f1f1, f2f2, - prob_feature_space * 2, cmap=plt.cm.RdBu)


def plot(clf_result_dict, entire_dataset, val_dataset, is_saved=True):
    clf = clf_result_dict["classifier"]
    pred_y_list = clf_result_dict["pred_list"]
    val_dataset.filter_by_pred(pred_y_list=pred_y_list, cls=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

    ax[0].scatter(entire_dataset.filter_by_class(cls=-1).T[0], entire_dataset.filter_by_class(cls=-1).T[1], label="Class 1")
    ax[0].scatter(entire_dataset.filter_by_class(cls=1).T[0], entire_dataset.filter_by_class(cls=1).T[1], label="Class 2")
    ax[0].set_title("Entire Dataset")
    ax[0].legend()

    plot_decision_boundary(clf=clf, dataset=val_dataset, ax=ax[1])
    ax[1].scatter(val_dataset.filter_by_class(cls=-1).T[0], val_dataset.filter_by_class(cls=-1).T[1], label="Class 1")
    ax[1].scatter(val_dataset.filter_by_class(cls=1).T[0], val_dataset.filter_by_class(cls=1).T[1], label="Class 2")
    ax[1].set_title(f"RBF Network :: Validation Dataset")
    ax[1].legend()

    if is_saved:
        plt.savefig(f"./rbf_network/rbf_network_plot.png")

    plt.show()


def main():
    Path("./rbf_network").mkdir(parents=True, exist_ok=True)

    # Generate Dataset
    dataset = MGDDataset.generate_entire_dataset(cls1_dataset_n=1000, cls2_dataset_n=1000, is_shuffle=True)
    dataset.convert_label_zero_to_neg()

    train_test_dataset_dict = dataset.train_test_split(test_ratio=0.25)
    train_dataset = train_test_dataset_dict["train_dataset"]
    val_dataset = train_test_dataset_dict["val_dataset"]

    print(f"Size of train dataset : {len(train_dataset)}, Size of val dataset : {len(val_dataset)}")

    print("Start to train")
    # Train
    clf_result_dict = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        hidden_shape=75,
        sigma=1.0
    )
    print("Complete to train")

    # Plotting
    print("Start to plot")
    plot(clf_result_dict=clf_result_dict, entire_dataset=dataset, val_dataset=val_dataset, is_saved=True)
    print("Complete to plot")


if __name__ == "__main__":
    main()
