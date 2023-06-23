import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def forward(output, A, B, PI) -> np.array:
    """
    predict models by forward algorithm

    Args:
        output (np.array): output sequence
        A (np.array): transition probability matrix
        B (np.array): output probability
        PI (np.array): initial probability

    Returns:
        predicted_models(np.array): predicted models sequence
    """
    number_of_outputs = output.shape[0]
    length_of_output = output.shape[1]
    predict_models = np.zeros(number_of_outputs)
    for i in range(number_of_outputs):
        # initialization
        output_o1 = output[i, 0]
        alpha = PI[:, :, 0] * B[:, :, output_o1]
        # recursive
        for j in range(1, length_of_output):
            output_oi = output[i, j]
            alpha = np.sum(alpha[:, :, np.newaxis] * A, axis=1)\
                * B[:, :, output_oi]
        # calculate probabilities for each model
        P = np.sum(alpha, axis=1)
        predict_models[i] = np.argmax(P)

    return predict_models


def viterbi(output, A, B, PI) -> np.array:
    """
    predict models by viterbi algorithm

    Args:
        output (np.array): output sequence
        A (np.array): transition probability matrix
        B (np.array): output probability
        PI (np.array): initial probability

    Returns:
        predicted_models(np.array): predicted models sequence
    """
    number_of_outputs = output.shape[0]
    length_of_output = output.shape[1]
    predict_models = np.zeros(number_of_outputs)
    for i in range(number_of_outputs):
        # initialization
        output_o1 = output[i, 0]
        alpha = PI[:, :, 0] * B[:, :, output_o1]
        # recursive
        for j in range(1, length_of_output):
            output_oi = output[i, j]
            alpha = np.max(alpha[:, :, np.newaxis] * A, axis=1)\
                * B[:, :, output_oi]
        # calculate probabilities for each model
        P = np.max(alpha, axis=1)
        predict_models[i] = np.argmax(P)

    return predict_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path to input data")
    args = parser.parse_args()

    path = args.path
    filename = os.path.splitext(os.path.basename(path))[0]

    # load pickle file
    data = pickle.load(open(path, "rb"))
    output = np.array(data["output"])
    A = np.array(data["models"]["A"])
    B = np.array(data["models"]["B"])
    PI = np.array(data["models"]["PI"])
    answer_models = np.array(data["answer_models"])

    # execute HMM
    predicted_models_forward = forward(output, A, B, PI)
    predicted_models_viterbi = viterbi(output, A, B, PI)

    # display results
    plt.rcParams["figure.figsize"] = (13, 6)
    fig = plt.figure()
    # result by forward algorithm
    plt.subplot(121)
    acc_forward = 100 * accuracy_score(answer_models, predicted_models_forward)
    cm = confusion_matrix(answer_models, predicted_models_forward)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")
    plt.title(f"Forward algorithm\n(Acc. {acc_forward:.0f}%)")
    # result by viterbi algorithm
    plt.subplot(122)
    acc_viterbi = 100 * accuracy_score(answer_models, predicted_models_viterbi)
    cm = confusion_matrix(answer_models, predicted_models_viterbi)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel("Predicted model")
    plt.ylabel("Actual model")
    plt.title(f"Viterbi algorithm\n(Acc. {acc_viterbi:.0f}%)")

    fig.savefig(f"result_{filename}.png")


if __name__ == "__main__":
    main()