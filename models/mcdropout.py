from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Input,
    Dropout,
    Conv1D,
    Flatten,
    RepeatVector,
    LSTM,
    TimeDistributed,
)
from utility.config import Options
import pandas as pd
import numpy as np
from keras.regularizers import l2


def create_model_regression(data, options: Options):
    N = data.shape[0]
    batch_size = 128
    lengthscale = 1e-2
    reg = (
        lengthscale**2
        * (1 - options.variant.dropoutrate)
        / (2.0 * N * options.variant.tau)
    )

    inputs = Input(shape=(options.inputlayer.number_input,))
    first = True
    for layer in options.layers:
        if first:
            x = Dense(
                layer.nodes, activation=layer.activation, kernel_regularizer=l2(reg)
            )(inputs)
            x = Dropout(options.variant.dropoutrate)(x, training=True)
            first = False
        else:
            x = Dense(
                layer.nodes, activation=layer.activation, kernel_regularizer=l2(reg)
            )(x)
            x = Dropout(options.variant.dropoutrate)(x, training=True)
    outputs = Dense(options.outputlayer.number_output)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=options.compilation.loss_function, optimizer=options.compilation.optimizer
    )
    return model


def create_model_pedestrian(data, options: Options):
    N = data["x"].shape[0]
    batch_size = options.training.batch_size
    lengthscale = 1e-2
    reg = (
        lengthscale**2
        * (1 - options.variant.dropoutrate)
        / (2.0 * N * options.variant.tau)
    )

    inputs = Input(shape=(options.pedestrian.lookback, data["x"].shape[2]))
    first = True
    for layer in options.layers:
        if first:
            x = Conv1D(
                filters=layer.nodes,
                kernel_size=layer.kernel_size,
                activation=layer.activation,
                padding="same",
                kernel_regularizer=l2(reg),
            )(inputs)
            x = Dropout(options.variant.dropoutrate)(x, training=True)
            first = False
        else:
            x = Conv1D(
                filters=layer.nodes,
                kernel_size=layer.kernel_size,
                activation=layer.activation,
                padding="same",
                kernel_regularizer=l2(reg),
            )(x)
            x = Dropout(options.variant.dropoutrate)(x, training=True)

    encoder_flattened = Flatten()(x)
    r_vec = RepeatVector(options.pedestrian.forward)(encoder_flattened)
    decoder = LSTM(
        options.pedestrian.nodes,
        return_sequences=True,
        activation=options.pedestrian.activation,
        kernel_regularizer=l2(reg),
    )(r_vec)
    decoder = Dropout(options.variant.dropoutrate)(decoder, training=True)
    output = TimeDistributed(Dense(2, activation="linear"))(decoder)
    model = Model(inputs, output)

    model.compile(
        loss=options.compilation.loss_function, optimizer=options.compilation.optimizer
    )
    return model


def train_model(model: Model, data: pd.DataFrame, options: Options):
    model.fit(
        data["x"],
        data["y"],
        epochs=options.training.epochs,
        batch_size=options.training.batch_size,
    )
    return


def predict_model(model: Model, x: np.array, options: Options):
    n_draws = options.variant.n_draws
    mean_dropout = []
    std_dropout = []

    for i in range(x.shape[0]):
        predictions = model.predict(np.repeat(x[i : i + 1], n_draws, axis=0), verbose=0)
        mean_dropout.append(predictions.mean(axis=0))
        std_dropout.append(predictions.std(axis=0))
    mean_dropout = np.array(mean_dropout)
    std_dropout = np.array(std_dropout)
    return mean_dropout, std_dropout
