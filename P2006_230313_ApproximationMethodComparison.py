"""
Filename:   P2006_230313_ApproximationMethodComparison.py
Author:     Turibius Rozario
Advisor:    Dr. Ankit Goel
Created:    March 13, 2023
Compares different approximation methods using TensorFlow's default settings.
The four optimization method arrangements tested are:
1. SGD with Adam
2. SGD with no momentum
3. Batch with Adam
4. Batch with no momentum
"""

# Imports --------------------------------------------------------------------
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from matplotlib import pyplot as plt


# Functions ------------------------------------------------------------------
def generate_truth(num_samples, input_length, neurons_in_layer):
    Xin = np.random.normal(size=(num_samples, input_length)) * MULTIPLIER
    neurons_in_layer = np.array(neurons_in_layer)
    weights = list()
    model = generate_NN(input_length, neurons_in_layer)
    for w in model.get_weights():
        weights.append(w)
    Yout = model.predict(Xin)
    return Xin, Yout, weights


def generate_NN(input_length, neurons_in_layer):
    nl = len(neurons_in_layer)
    model = keras.models.Sequential((
        Dense(neurons_in_layer[0], input_dim=input_length,
              activation="relu")
    ))
    if nl > 2:
        for i in range(1, nl-1):
            model.add(Dense(neurons_in_layer[i], activation="relu"))
        model.add(Dense(neurons_in_layer[-1], activation=None))
    elif nl == 2:
        model.add(Dense(neurons_in_layer[-1], activation=None))
    return model


def train_model(Xin, Yout, model_epochs, neurons_in_layer, case):
    input_length = Xin.shape[1]
    num_samples = Xin.shape[0]
    model = generate_NN(input_length, neurons_in_layer)
    if case == 1:
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(Xin, Yout, epochs=model_epochs)
    elif case == 2:
        model.compile(optimizer="sgd", loss="mse")
        history = model.fit(Xin, Yout, epochs=model_epochs)
    elif case == 3:
        model.compile(optimizer="adam", loss="mse")
        history = model.fit(Xin, Yout, epochs=model_epochs,
                            batch_size=num_samples)
    elif case == 4:
        model.compile(optimizer="sgd", loss="mse")
        history = model.fit(Xin, Yout, epochs=model_epochs,
                            batch_size=num_samples)
    else:
        print("error in train_model")
        history = "error"
    Yhat = model.predict(Xin)
    weights = list()
    for w in model.get_weights():
        weights.append(w)
    return Yhat, weights, history.history['loss']


def compute_norms(weights_true, weights_predicted, aggregation=1):
    norm_residual = list()
    if len(weights_true) == len(weights_predicted):
        for i in range(len(weights_true)):
            norm_residual.append(np.linalg.norm(weights_true[i]
                                                - weights_predicted[i]))
        if aggregation:
            return np.linalg.norm(norm_residual)
        else:
            return norm_residual
    else:
        return 0


# Constants ------------------------------------------------------------------

MULTIPLIER = 5


if __name__ == "__main__":
    # Constant initializations
    seed = 1
    ns = 100  # number of samples
    lx1 = 2  # elements in each sample
    layers = [4, 2, 1]  # size of each layer
    epochs = 20
    epochs_vs_updates = "epochs"

    print("Some optimizers update every epoch, "
          "while others update every sample,"
          "therefore allowing the latter to converge faster; "
          "to allow this, set "
          "option to 'epochs'. To see the effect of UPDATES, select 'updates'")
    print("Current setting is:", epochs_vs_updates)

    # Variable initialization
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    policy = keras.mixed_precision.Policy("float64")
    keras.mixed_precision.set_global_policy(policy)
    X, Y, THETA = generate_truth(ns, lx1, layers)

    # Case 1: Adam, update at each sample
    Yhat_1, THETA_hat_1, J1 = train_model(X, Y, epochs, layers, 1)
    # Case 2: Update at each sample
    Yhat_2, THETA_hat_2, J2 = train_model(X, Y, epochs, layers, 2)
    # Case 3: Adam, update at each epoch
    if epochs_vs_updates == "epochs":
        Yhat_3, THETA_hat_3, J3 = train_model(X, Y, epochs, layers, 3)
    elif epochs_vs_updates == "updates":
        Yhat_3, THETA_hat_3, J3 = train_model(X, Y, epochs * ns, layers, 3)
        J3 = J3[0::ns]
    # Case 4: Update at each epoch
    if epochs_vs_updates == "epochs":
        Yhat_4, THETA_hat_4, J4 = train_model(X, Y, epochs, layers, 4)
    elif epochs_vs_updates == "updates":
        Yhat_4, THETA_hat_4, J4 = train_model(X, Y, epochs * ns, layers, 4)
        J4 = J4[0::ns]

    # Printing the norms:
    print("Norm 1:", compute_norms(THETA, THETA_hat_1))
    print("Norm 2:", compute_norms(THETA, THETA_hat_2))
    print("Norm 3:", compute_norms(THETA, THETA_hat_3))
    print("Norm 4:", compute_norms(THETA, THETA_hat_4))

    for i in range(len(THETA_hat_1)):
        print(THETA_hat_1[i] - THETA[i])

    plt.title("Cost using various methods")
    plt.xlabel("Epoch")
    plt.ylabel("J")
    plt.semilogy(J1, label="stochastic + momentum")
    plt.semilogy(J2, label="stochastic")
    plt.semilogy(J3, label="batch + momentum")
    plt.semilogy(J4, label="batch")
    plt.legend()
    plt.show()
