from tensorflow.keras.layers import (
    Dense,
    Input,
    Conv1D,
    Flatten,
    RepeatVector,
    LSTM,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from utility.config import Options
import pandas as pd
import numpy as np


def create_model_regression(options: Options) -> list[Model]:
    ensemble = []
    for i in range(options.variant.n_ensemble):
        inputs = Input(shape=(options.inputlayer.number_input,))
        first = True
        for layer in range(np.random.randint(2, options.variant.max_layers)):
            if first:
                x = Dense(
                    np.random.randint(
                        options.variant.min_nodes, options.variant.max_nodes
                    ),
                    activation=np.random.choice(
                        options.variant.activation_functions, 1
                    )[0],
                )(inputs)
                first = False
            else:
                x = Dense(
                    np.random.randint(
                        options.variant.min_nodes, options.variant.max_nodes
                    ),
                    activation=np.random.choice(
                        options.variant.activation_functions, 1
                    )[0],
                )(x)
        outputs = Dense(options.outputlayer.number_output)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=options.compilation.loss_function,
            optimizer=np.random.choice(options.variant.optimizers, 1)[0],
        )
        ensemble.append(model)
    return ensemble


def create_model_pedestrian(options) -> list[Model]:
    ensemble = []
    for i in range(options.variant.n_ensemble):
        first = True
        inputs = Input(shape=(options.pedestrian.lookback, 2))
        for layer in range(np.random.randint(2, options.variant.max_layers)):
            if first:
                x = Conv1D(
                    filters=np.random.randint(
                        options.variant.min_nodes, options.variant.max_nodes
                    ),
                    kernel_size=3,
                    activation=np.random.choice(
                        options.variant.activation_functions, 1
                    )[0],
                    padding="same",
                )(inputs)
                first = False
            else:
                x = Conv1D(
                    filters=np.random.randint(
                        options.variant.min_nodes, options.variant.max_nodes
                    ),
                    kernel_size=3,
                    activation=np.random.choice(
                        options.variant.activation_functions, 1
                    )[0],
                    padding="same",
                )(x)

        encoder_flattened = Flatten()(x)
        r_vec = RepeatVector(options.pedestrian.forward)(encoder_flattened)
        decoder = LSTM(
            np.random.randint(options.variant.min_nodes, options.variant.max_nodes),
            return_sequences=True,
            activation=np.random.choice(options.variant.activation_functions, 1)[0],
        )(r_vec)

        output = TimeDistributed(Dense(2, activation="linear"))(decoder)
        model = Model(inputs, output)

        model.compile(
            loss=options.compilation.loss_function,
            optimizer=np.random.choice(options.variant.optimizers, 1)[0],
        )
        ensemble.append(model)
    return ensemble


def train_model(ensemble, data: pd.DataFrame, options: Options):
    for model in ensemble:
        ind_list = np.arange(data["x"].shape[0])
        np.random.shuffle(ind_list)
        print(model.summary())
        print()
        model.fit(
            data["x"][ind_list],
            data["y"][ind_list],
            epochs=options.training.epochs,
            batch_size=32,
        )
    return


def predict_model(ensemble: list[Model], x: np.array, options: Options):
    predictions = []

    for model in ensemble:
        prediction = model.predict(x, verbose=0)
        predictions.append(prediction)

    result = np.array(predictions)
    mean = np.mean(result, axis=0)
    std = np.std(result, axis=0)
    return mean, std


if __name__ == "__main__":
    model = create_model()
