"""
2020572045 정 병길
Pattern Recognition
HW2 - Main Code
"""
import random
import time

import numpy as np

from k_svm import main as k_svm_main
from rbf_network import main as rbf_network_main
from neural_network import main as nn_main

time.time()
RANDOM_SEED = 777

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def main():
    print("[ 1. RBF Network ]")
    rbf_network_main()

    print("[ 2. K-SVM ]")
    k_svm_main()

    print("[ 3. Neural Network ]")
    nn_main()


if __name__ == "__main__":
    main()
