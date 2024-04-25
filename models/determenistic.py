from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np


def create_model() -> Model:
    """Create determinstic model that is not able to quantify uncertainty

    Returns:
        Model: tensorflow model
    """
    inputs = Input(shape=(1,))
    x = Dense(31, activation="relu")(inputs)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mse")
    return model


def train(model: Model, data: pd.DataFrame) -> None:
    """Train determenistic model

    Args:
        model (Model): model to be trained
        data (pd.DataFrame): data to train the model on
    """
    model.fit(data["x"], data["y"], epochs=20, batch_size=20)
    return


def predict(model: Model, x: pd.DataFrame) -> np.ndarray:
    """Create predictions

    Args:
        model (Model): trained model
        x (pd.DataFrame): data to predict

    Returns:
        np.ndarray: predictions
    """
    y = model.predict(x)
    return y


if __name__ == "__main__":
    model = create_model()
