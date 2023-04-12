"""
Filename:   P2007_230319_EffectOfBatchSize.py
Author:     Turibius Rozario
Advisor:    Dr. Ankit Goel
Created:    March 19, 2023
Illustrates effect of different batch sizes on training results. The following
parameters are tested:
1. Fixed alpha (learning rate) vs. alpha = default alpha * batch size.
2. Batch sizes (sizes are 2^n, from n=0 to n=6).
3. Optimizer momentum (none vs adam).
Each training will be done 10 times.
Plots will be made externally in MATLAB, with project name of P2007.
"""


# Imports
import numpy as np
from tensorflow import keras
from keras.layers import Dense


# Functions
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


def train_model(Xin, Yout, model_epochs, neurons_in_layer,
                optimizer='sgd', lr=0.01, num_samples=1):
    # Initializations
    input_length = Xin.shape[1]
    model = generate_NN(input_length, neurons_in_layer)
    # Customizing the training method and the learning rate
    if optimizer == 'sgd':
        optim = keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'adam':
        optim = keras.optimizers.Adam(learning_rate=lr)
    else:
        print("Error in train_model! 'optimizer' error. Default used")
        optim = "sgd"
    model.compile(optimizer=optim, loss="mse")
    history = model.fit(Xin, Yout, epochs=model_epochs,
                        batch_size=num_samples, verbose=VERBOSE)
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
# 0 = silent, 1 = progress bar, 2 = one line per epoch
VERBOSE = 0
RUNS = 20  # number of times to do a particular test
POW = 6
SAVE_LOC_PATH = "P2007_Files/TEST_"


if __name__ == "__main__":
    # Constant initializations
    seed = 1
    ns = 2 ** POW  # number of samples
    lx1 = 2  # elements in each sample
    layers = [4, 2, 1]  # size of each layer
    epochs = 100
    epochs_vs_updates = "epochs"
    alpha = 0.01

    # Variable initialization
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    policy = keras.mixed_precision.Policy("float64")
    keras.mixed_precision.set_global_policy(policy)
    X, Y, THETA = generate_truth(ns, lx1, layers)

    # Testing
    for k in range(4):
        test = k + 1
        costs_all = list()
        costs = list()
        norms = np.zeros((RUNS, POW + 1))
        for i in range(POW + 1):
            nb = 2 ** i
            for j in range(RUNS):
                # alpha = c; no momentum
                if test == 1:
                    THETA_hat, J = train_model(X, Y, epochs, layers,
                                               optimizer='sgd', lr=alpha,
                                               num_samples=nb)[1:3]
                    costs.append(J)
                    norms[j][i] = compute_norms(THETA, THETA_hat)
                # alpha = c; momentum
                elif test == 2:
                    THETA_hat, J = train_model(X, Y, epochs, layers,
                                               optimizer='adam', lr=alpha,
                                               num_samples=nb)[1:3]
                    costs.append(J)
                    norms[j][i] = compute_norms(THETA, THETA_hat)
                # alpha = c * nb; no momentum
                elif test == 3:
                    THETA_hat, J = train_model(X, Y, epochs, layers,
                                               optimizer='sgd', lr=alpha * nb,
                                               num_samples=nb)[1:3]
                    costs.append(J)
                    norms[j][i] = compute_norms(THETA, THETA_hat)
                # alpha = c * nb; momentum
                elif test == 4:
                    THETA_hat, J = train_model(X, Y, epochs, layers,
                                               optimizer='adam', lr=alpha * nb,
                                               num_samples=nb)[1:3]
                    costs.append(J)
                    norms[j][i] = compute_norms(THETA, THETA_hat)
                print("nb =", nb, "run", j + 1, "end cost: ", costs[j][-1])
            costs_all.append(costs)
            costs = []

        # Saving the result
        filename = SAVE_LOC_PATH + str(test) + "_"
        # Saving the norms:
        np.savetxt(filename + "norms.csv", norms, delimiter=",")
        # Saving the costs:
        for i in range(POW + 1):
            file = filename + "J" + "_" + str(i) + ".csv"
            costs = costs_all[i]
            np.savetxt(file, np.asarray(costs), delimiter=",")

        print("Test #\t\t\t No momentum \t Momentum\n" +
              "alpha = c \t\t 1 \t\t\t\t 2\n" +
              "alpha = c * nb \t 3 \t\t\t\t 4\n")
        print("Current test # was:", test, "\n")

    # Information
    print("Some optimizers update every epoch, "
          "while others update every sample,"
          "therefore allowing the latter to converge faster;"
          "to allow this, set"
          "option to 'epochs'. To see the effect of UPDATES, select 'updates'",
          "\n")
    print("Current setting is:", epochs_vs_updates, "\n")

    print("The norms will have vary in\n",
          "runs going top to bottom\n",
          "number of batches in 2**POW format left to right\n")

    print("In J_#, # is the power. Within it, it will vary in\n",
          "runs going top to bottom\n",
          "epoch going left to right")
