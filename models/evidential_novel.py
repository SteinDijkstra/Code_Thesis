from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    Layer,
    Conv1D,
    Flatten,
    RepeatVector,
    LSTM,
    TimeDistributed,
)
import tensorflow as tf
from utility.config import Options
import pandas as pd
import numpy as np


def create_model_regression(options: Options):
    inputs = Input(shape=(options.inputlayer.number_input,))
    first = True
    for layer in options.layers:
        if first:
            x = Dense(layer.nodes, activation=layer.activation)(inputs)
            first = False
        else:
            x = Dense(layer.nodes, activation=layer.activation)(x)
    outputs = LastLayer(options.outputlayer.number_output)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=EvidentialRegression, optimizer=options.compilation.optimizer)
    return model


def create_model_pedestrian(options: Options):
    inputs = Input(shape=(options.pedestrian.lookback, 2))
    first = True
    for layer in options.layers:
        if first:
            x = Conv1D(
                filters=layer.nodes,
                kernel_size=layer.kernel_size,
                activation=layer.activation,
                padding="same",
            )(inputs)
            first = False
        else:
            x = Conv1D(
                filters=layer.nodes,
                kernel_size=layer.kernel_size,
                activation=layer.activation,
                padding="same",
            )(x)
    encoder_flattened = Flatten()(x)
    r_vec = RepeatVector(options.pedestrian.forward)(encoder_flattened)
    decoder = LSTM(
        options.pedestrian.nodes,
        return_sequences=True,
        activation=options.pedestrian.activation,
    )(r_vec)
    output = TimeDistributed(LastLayer(2))(decoder)

    model = Model(inputs, output)
    model.compile(loss=EvidentialRegression, optimizer=options.compilation.optimizer)
    return model


def NIG_NLL(y, gamma, v, alpha, beta, reduce=True):
    omega = 2 * beta * (1 + v)

    nll = (
        0.5 * tf.math.log(np.pi / v)
        - alpha * tf.math.log(omega)
        + (alpha + 0.5) * tf.math.log(v * (y - gamma) ** 2 + omega)
        + tf.math.lgamma(alpha)
        - tf.math.lgamma(alpha + 0.5)
    )

    return tf.reduce_mean(nll) if reduce else nll


def NIG_Reg(y, gamma, v, alpha, beta, reduce=True):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    wst = tf.math.sqrt((beta * (1 + v)) / (alpha * v))
    error = tf.abs((y - gamma))
    reg = error / wst

    return tf.reduce_mean(reg) if reduce else reg


def EvidentialRegression(y_true, evidential_output, coeff=1):
    gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)
    return loss_nll + coeff * (loss_reg)


class LastLayer(Layer):
    def __init__(self, units):
        super(LastLayer, self).__init__()
        self.units = int(units)
        self.dense = Dense(4 * self.units, activation=None)

    def call(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = tf.split(output, 4, axis=-1)
        v = tf.nn.softplus(logv)
        alpha = tf.nn.softplus(logalpha) + 1
        beta = tf.nn.softplus(logbeta)
        return tf.concat([mu, v, alpha, beta], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4 * self.units)

    def get_config(self):
        base_config = super(DenseNormalGamma, self).get_config()
        base_config["units"] = self.units
        return base_config


def train_model(model: Model, data: pd.DataFrame, options: Options):
    model.fit(
        data["x"],
        data["y"],
        epochs=options.training.epochs,
        batch_size=options.training.batch_size,
    )
    return


def predict_model(model: Model, x: pd.DataFrame, options: Options):
    result = model.predict(x)
    if len(x.shape) == 3:
        gamma = result[:, :, 0:2]
        v = result[:, :, 2:4]
        alpha = result[:, :, 4:6]
        beta = result[:, :, 6:8]
    elif len(x.shape) == 1:
        gamma = result[:, 0:1]
        v = result[:, 1:2]
        alpha = result[:, 2:3]
        beta = result[:, 3:4]

    return gamma, ((beta * (1 + v)) / (v * (alpha - 1))), 2 * alpha
