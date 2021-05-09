"""
210509
HW2 - K_SVM
"""
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from dataset import MGDDataset


time.time()
RANDOM_SEED = 777

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def train(train_dataset, val_dataset, kernel_name_list=["linear", "poly", "rbf", "sigmoid"]):
    clf_result_dict = {kernel_name: dict() for kernel_name in kernel_name_list}

    for kernel_name in kernel_name_list:
        print(f" === [ {kernel_name} ] === ")
        clf = SVC(kernel=kernel_name)
        clf.fit(X=train_dataset.X, y=train_dataset.y)

        y_pred_list = clf.predict(X=val_dataset.X)
        # y_pred_list = np.expand_dims(y_pred_list, axis=1)
        # print(y_pred_list)

        metrics_dict = classification_report(
                y_true=val_dataset.y,
                y_pred=y_pred_list,
                target_names=["class_1", "class_2"],
                output_dict=True
            )
        print(classification_report(
                y_true=val_dataset.y,
                y_pred=y_pred_list,
                target_names=["class_1", "class_2"],
                output_dict=False
            ))

        clf_result_dict[kernel_name]["classifier"] = clf
        clf_result_dict[kernel_name]["pred_y_list"] = y_pred_list
        clf_result_dict[kernel_name]["metrics"] = metrics_dict

    return clf_result_dict


def plot_decision_boundary(clf, dataset, ax):
    min_f1, max_f1 = np.min(dataset.X[:, 0]), np.max(dataset.X[:, 0])
    min_f2, max_f2 = np.min(dataset.X[:, 1]), np.max(dataset.X[:, 1])

    f1_space = np.linspace(min_f1, max_f1, 100)
    f2_space = np.linspace(min_f2, max_f2, 100)

    f1f1, f2f2 = np.meshgrid(f1_space, f2_space)

    feature_space = np.vstack([f1f1.ravel(), f2f2.ravel()]).T

    Z1 = clf.decision_function(feature_space).reshape(f1f1.shape) # 100,100

    ax.contour(f1f1, f2f2, Z1, colors='k', levels=1, alpha=0.5, linestypes=['--','-','--'])
    ax.pcolormesh(f1f1, f2f2, -Z1, cmap=plt.cm.RdBu)


def plot(clf_result_dict, entire_dataset, val_dataset, is_saved=True):

    for kernel_name in clf_result_dict.keys():
        print(f" === [ {kernel_name} ] ===")

        current_clf_result_dict = clf_result_dict[kernel_name]
        clf = current_clf_result_dict["classifier"]
        pred_y_list = current_clf_result_dict["pred_y_list"]
        val_dataset.filter_by_pred(pred_y_list=pred_y_list, cls=0)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

        ax[0].scatter(entire_dataset.filter_by_class(cls=0).T[0], entire_dataset.filter_by_class(cls=0).T[1], label="Class 1")
        ax[0].scatter(entire_dataset.filter_by_class(cls=1).T[0], entire_dataset.filter_by_class(cls=1).T[1], label="Class 2")
        ax[0].set_title("Entire Dataset")
        ax[0].legend()

        plot_decision_boundary(clf=clf, dataset=val_dataset, ax=ax[1])
        ax[1].scatter(val_dataset.filter_by_class(cls=0).T[0], val_dataset.filter_by_class(cls=0).T[1], label="Class 1")
        ax[1].scatter(val_dataset.filter_by_class(cls=1).T[0], val_dataset.filter_by_class(cls=1).T[1], label="Class 2")
        ax[1].set_title(f"{kernel_name} :: Validation Dataset")
        ax[1].legend()

        if is_saved:
            plt.savefig(f"./k_svm/{kernel_name}_plot.png")

        plt.show()


def main():
    Path("./k_svm").mkdir(parents=True, exist_ok=True)

    # Generate Dataset
    dataset = MGDDataset.generate_entire_dataset(cls1_dataset_n=1000, cls2_dataset_n=1000, is_shuffle=True)

    train_test_dataset_dict = dataset.train_test_split(test_ratio=0.25)
    train_dataset = train_test_dataset_dict["train_dataset"]
    val_dataset = train_test_dataset_dict["val_dataset"]

    print(f"Size of train dataset : {len(train_dataset)}, Size of val dataset : {len(val_dataset)}")

    print("Start to train")
    # Train
    clf_result_dict = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        kernel_name_list=["linear", "poly", "rbf", "sigmoid"]
    )
    print("Complete to train")

    # Plotting
    print("Start to plot")
    plot(clf_result_dict=clf_result_dict, entire_dataset=dataset, val_dataset=val_dataset, is_saved=True)
    print("Complete to plot")


if __name__ == "__main__":
    main()
