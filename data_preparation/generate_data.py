import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import pickle


def generate_dataset1(
    n: int = 1000,
    min_x: float = -3,
    max_x: float = 4,
    var_eps: float = 3,
    seed: int = 14122023,
):
    """Generate data according to x^3 with a heteroskedastic variance

    Args:
        n (int, optional): _description_. Defaults to 200.
        min_x (float, optional): _description_. Defaults to -3.
        max_x (float, optional): _description_. Defaults to 4.
        var_eps (float, optional): _description_. Defaults to 3.
        seed (int, optional): _description_. Defaults to 14122023.

    Returns:
        float,float: _description_
    """
    np.random.seed(seed)
    x = np.random.uniform(min_x, max_x, n)
    variance = np.maximum(np.ones(n) * 3, 36 - 36 * (x**2))
    y = x**3 + np.random.normal(np.zeros(n), np.sqrt(variance), n)

    data = pd.DataFrame({"x": x, "y": y})
    return data


def save_data(train: pd.DataFrame, test: pd.DataFrame, path):
    """save data as pickle

    Args:
        train (pd.DataFrame): traindata
        test (pd.DataFrame): testdata
        path (_type_): path where to save data
    """
    with open(path, "wb") as f:
        pickle.dump(train, f)
        pickle.dump(test, f)
    return


def load_data(path):
    """Load data from path

    Args:
        path (Path): Where to load data from

    Returns:
        Tuple(pd.DatFrame,pd.DataFrame): (train,test)
    """
    with open(path, "rb") as f:
        train = pickle.load(f)
        test = pickle.load(f)
    return train, test
