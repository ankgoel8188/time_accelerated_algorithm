"""
Filename:   P2008_230320_GradientTapeAndFiniteTime.py
Author:     Turibius Rozario
Advisor:    Dr. Ankit Goel
Created:    March 20, 2023
Creates and tests a few simple hard coded discrete time finite time methods.
Several small steps are first done to get familiar with TensorFlow's
GradientTape function.
"""


# Imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense


# Functions
def tape_on_scalar(x):
    """
    Uses tape on a scalar variable. Function is y=x^2
    :param x: Any number
    :return: Its derivative.
    """
    x = tf.Variable(x)
    with tf.GradientTape() as tape:
        y = x ** 3
    dydx = tape.gradient(y, x)
    return dydx


def tape_on_vector(x):
    w = tf.Variable(np.random.normal(size=(3, 2)),
                    trainable=True)
    b = tf.Variable(tf.zeros(2, dtype=tf.float64), trainable=True)
    with tf.GradientTape() as tape:
        y = x @ w + b
        loss = tf.reduce_mean(y ** 2)
    [dl_dw, dl_db] = tape.gradient(loss, [w, b])
    del tape
    return y, w, b, dl_dw, dl_db


def generate_truth(num_samples, input_length, neurons_in_layer,
                   activation="mixed"):
    """
    Generates true values based on random weight and bias.
    :param num_samples: Number of input samples
    :param input_length: Length of a single input
    :param neurons_in_layer: List containing number of neurons in each layer
    :param activation: "mixed" for Relu Relu ... None, "linear" for single
    layered linear NN, "relu" for single layered ReLU. latter two requires that
    neurons_in_layer have a length of 1
    :return: (X values, Y values, random weights used to generate Y from X)
    """
    Xin = np.random.normal(size=(num_samples, input_length)) * MULTIPLIER
    neurons_in_layer = np.array(neurons_in_layer)
    weights = list()
    model = generate_NN(input_length, neurons_in_layer, activation=activation)
    for w in model.get_weights():
        weights.append(w)
    Yout = model.predict(Xin)
    return Xin, Yout, weights


def generate_NN(input_length, neurons_in_layer, activation="mixed"):
    """
    Generates a model that is: one layer with either ReLU or no activation, or
    multi layered ReLU with last layer being no activation function.
    :param input_length: The dimension of a single input sample
    :param neurons_in_layer: A list where each element denotes number of neurons
    in a given layer
    :param activation: Relu = "relu", no activation = "linear", multi layered
    ReLU with last layer being linear = "mixed" (default)
    :return: Keras model
    """
    nl = len(neurons_in_layer)
    if activation == "linear":
        model = keras.models.Sequential((
            Dense(neurons_in_layer[0], input_dim=input_length,
                  activation=None)
        ))
    elif activation == "relu":
        model = keras.models.Sequential((
            Dense(neurons_in_layer[0], input_dim=input_length,
                  activation="relu")
        ))
    else:
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


def linear_minimization(num_samples, input_length, neurons_in_layer):
    activation = "linear"
    if len(neurons_in_layer) == 1:
        X, Y, THETA = generate_truth(num_samples, input_length,
                                     neurons_in_layer, activation)
        NN = generate_NN(input_length, neurons_in_layer, activation)
    else:
        print("neurons_in_layer must have one element ONLY!")


MULTIPLIER = 5
SEED = 1
EPOCHS = 5


if __name__ == "__main__":
    keras.utils.set_random_seed(SEED)
    keras.mixed_precision.set_global_policy(
        keras.mixed_precision.Policy("float64"))
    # The following should print out 12 if value is 2. Must be a float:
    print(tape_on_scalar(2.0))
    # Should return a vector o 3, 12, 27 if entered 1, 2, 3:
    print(tape_on_scalar([1.0, 2.0, 3.0]))
    out, weight, bias, weight_der, bias_der = tape_on_vector([[1.0, 2.0, 3.0]])
    print(f"{out}\n\n{weight}\n\n{bias}\n\n{weight_der}\n\n{bias_der}")
    del out, weight, bias, weight_der, bias_der


