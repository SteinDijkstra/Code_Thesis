import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import stats


# load pickle
def load_data(path):
    """Load data of experiment

    Args:
        path (Path): Path to folder with experiments

    Returns:
        List(Pd.dataframe): data, testdata, evidential, adpt_evidential, dropout, ensemble
    """
    with open(path / "data.pkl", "rb") as f:
        data = pickle.load(f)

    with open(path / "testdata.pkl", "rb") as f:
        testdata = pickle.load(f)

    with open(path / "evidential/results.pkl", "rb") as f:
        result_evidential = pickle.load(f)

    with open(path / "evidential_new/results.pkl", "rb") as f:
        result_evidential_new = pickle.load(f)

    with open(path / "mcdropout/results.pkl", "rb") as f:
        result_dropout = pickle.load(f)

    with open(path / "ensemble/results.pkl", "rb") as f:
        result_ensemble = pickle.load(f)

    return (
        data,
        testdata,
        result_evidential,
        result_evidential_new,
        result_dropout,
        result_ensemble,
    )


# calculate metrics
def create_tables(
    actual, result_evidential, result_evidential_new, result_dropout, result_ensemble
):
    """create small table for ETH, HOTEL, ZARA01 and ZARA02

    Args:
        actual
        result_evidential
        result_evidential_new
        result_dropout
        result_ensemble

    Returns:
        list (table): _description_
    """
    row_names = [
        "Evidential Network",
        "Adapted Evidential Network",
        "MC Dropout",
        "Deep Ensemble",
    ]
    col_names = ["ADE", "min", "max", "FDE", "min", "max"]
    table_accuracy = pd.DataFrame(
        [
            create_row_acc(actual, result_evidential),
            create_row_acc(actual, result_evidential_new),
            create_row_acc(actual, result_dropout),
            create_row_acc(actual, result_ensemble),
        ],
        columns=col_names,
        index=row_names,
    )

    col_names = [
        "CSI(p=0.5)",
        "CSI(p=0.5,t=1)",
        "CSI(p=0.5,t=12)",
        "CSI(p=0.95)",
        "CSI(p=0.95,t=1)",
        "CSI(p=0.95,t=12)",
        "CSI(p>0.95)",
        "CSI(p>0.95,t=1)",
        "CSI(p>0.95,t=12)",
    ]
    table_uncertainty = pd.DataFrame(
        [
            create_row_unc(actual, result_evidential),
            create_row_unc(actual, result_evidential_new, tdist=True),
            create_row_unc(actual, result_dropout),
            create_row_unc(actual, result_ensemble),
        ],
        columns=col_names,
        index=row_names,
    )

    return table_accuracy, table_uncertainty


def create_table_eth(
    actual, result_evidential, result_evidential_new, result_dropout, result_ensemble
):
    """_summary_

    Args:
        actual (_type_): _description_
        result_evidential (_type_): _description_
        result_evidential_new (_type_): _description_
        result_dropout (_type_): _description_
        result_ensemble (_type_): _description_

    Returns:
        _type_: _description_
    """
    row_names = [
        "Evidential Network",
        "Adapted Evidential Network",
        "MC Dropout",
        "Deep Ensemble",
    ]
    col_names = [
        "CSI(p=0.5)",
        "CSI(p=0.5,t=1)",
        "CSI(p=0.5,t=2)",
        "CSI(p=0.5,t=4)",
        "CSI(p=0.5,t=6)",
        "CSI(p=0.5,t=8)",
        "CSI(p=0.5,t=10)",
        "CSI(p=0.5,t=12)",
        "CSI(p=0.95)",
        "CSI(p=0.95,t=1)",
        "CSI(p=0.95,t=2)",
        "CSI(p=0.95,t=4)",
        "CSI(p=0.95,t=6)",
        "CSI(p=0.95,t=8)",
        "CSI(p=0.95,t=10)",
        "CSI(p=0.95,t=12)",
        "CSI(p>0.95)",
        "CSI(p>0.95,t=1)",
        "CSI(p>0.95,t=2)",
        "CSI(p>0.95,t=4)",
        "CSI(p>0.95,t=6)",
        "CSI(p>0.95,t=8)",
        "CSI(p>0.95,t=10)",
        "CSI(p>0.95,t=12)",
    ]
    table_uncertainty = pd.DataFrame(
        [
            create_row_unc_eth(actual, result_evidential),
            create_row_unc_eth(actual, result_evidential_new, tdist=True),
            create_row_unc_eth(actual, result_dropout),
            create_row_unc_eth(actual, result_ensemble),
        ],
        columns=col_names,
        index=row_names,
    )
    return table_uncertainty


def create_row_acc(actual, pred):
    row = [*ade(actual, pred), *fde(actual, pred)]
    return row


def create_row_unc_eth(actual, pred, tdist=False):
    row = [
        csi_k(actual, pred, 0.5, tdist),
        csi_tk(actual, pred, 0, 0.5, tdist),
        csi_tk(actual, pred, 1, 0.5, tdist),
        csi_tk(actual, pred, 3, 0.5, tdist),
        csi_tk(actual, pred, 5, 0.5, tdist),
        csi_tk(actual, pred, 7, 0.5, tdist),
        csi_tk(actual, pred, 9, 0.5, tdist),
        csi_tk(actual, pred, 11, 0.5, tdist),
        csi_k(actual, pred, 0.95, tdist),
        csi_tk(actual, pred, 0, 0.95, tdist),
        csi_tk(actual, pred, 1, 0.95, tdist),
        csi_tk(actual, pred, 3, 0.95, tdist),
        csi_tk(actual, pred, 5, 0.95, tdist),
        csi_tk(actual, pred, 7, 0.95, tdist),
        csi_tk(actual, pred, 9, 0.95, tdist),
        csi_tk(actual, pred, 11, 0.95, tdist),
        csi_k(actual, pred, 0.95, tdist, chebychev=True),
        csi_tk(actual, pred, 0, 0.95, tdist, chebychev=True),
        csi_tk(actual, pred, 1, 0.95, tdist, chebychev=True),
        csi_tk(actual, pred, 3, 0.95, tdist, chebychev=True),
        csi_tk(actual, pred, 5, 0.95, tdist, chebychev=True),
        csi_tk(actual, pred, 7, 0.95, tdist, chebychev=True),
        csi_tk(actual, pred, 9, 0.95, tdist, chebychev=True),
        csi_tk(actual, pred, 11, 0.95, tdist, chebychev=True),
    ]
    return row


def create_table_regresssion(
    actual, result_evidential, result_evidential_new, result_dropout, result_ensemble
):
    row_names = [
        "Evidential Network",
        "Adapted Evidential Network",
        "MC Dropout",
        "Deep Ensemble",
    ]
    col_names = [
        "MSE",
        "Training time (s)",
        "Prediction time (s)",
        "k=0.5",
        "k=0.95",
        "k>0.95",
    ]
    table = pd.DataFrame(
        [
            create_row_regression(actual, result_evidential),
            create_row_regression(actual, result_evidential_new, tdist=True),
            create_row_regression(actual, result_dropout),
            create_row_regression(actual, result_ensemble),
        ],
        columns=col_names,
        index=row_names,
    )

    return table


def create_row_unc(actual, pred, tdist=False):
    row = [
        csi_k(actual, pred, 0.5, tdist),
        csi_tk(actual, pred, 0, 0.5, tdist),
        csi_tk(actual, pred, 11, 0.5, tdist),
        csi_k(actual, pred, 0.95, tdist),
        csi_tk(actual, pred, 0, 0.95, tdist),
        csi_tk(actual, pred, 11, 0.95, tdist),
        csi_k(actual, pred, 0.95, tdist, chebychev=True),
        csi_tk(actual, pred, 0, 0.95, tdist, chebychev=True),
        csi_tk(actual, pred, 11, 0.95, tdist, chebychev=True),
    ]
    return row


def create_row_regression(actual, pred, tdist=False):
    row = [
        mse(actual, pred),
        "",
        "",
        csi_regression(actual, pred, 0.5, tdist),
        csi_regression(actual, pred, 0.95, tdist),
        csi_regression(actual, pred, 0.95, tdist, chebychev=True),
    ]
    return row


def mse(actual, pred):
    errors = np.sqrt((pred[0].transpose()[0] - actual["y"]) ** 2)
    return np.mean(errors)


def ade(actual, pred):
    errors = np.sqrt(np.sum((pred[0] - actual["y"]) ** 2, axis=2))
    errors = np.mean(errors, axis=1)
    return (np.mean(errors), np.min(errors), np.max(errors))


def fde(actual, pred):
    errors = np.sqrt(np.sum((pred[0][:, -1, :] - actual["y"][:, -1, :]) ** 2, axis=1))
    return (np.mean(errors), np.min(errors), np.max(errors))


def csi_tk(actual, pred, t, p, tdist=False, chebychev=False):
    if chebychev:
        k = np.sqrt(1 / (1 - p))

    elif tdist:
        k = stats.t.ppf([1 - ((1 - p) / 2)], pred[2][:, t, :], 0, 1)
    else:
        k = stats.norm.ppf([1 - ((1 - p) / 2)], 0, 1)
        # k = k[:, np.newaxis, np.newaxis]
        # print(k)
    squared_errors = (actual["y"][:, t, :] - pred[0][:, t, :]) ** 2
    # print(k.shape)
    # print(pred[1][:, t, :].shape)
    radius_squared = k**2 * pred[1][:, t, :]
    isinside = np.sum(squared_errors / radius_squared, axis=1) < 1
    return np.mean(isinside, axis=0)


def csi_k(actual, pred, p, tdist=False, chebychev=False):
    if chebychev:
        k = np.sqrt(1 / (1 - p))
    elif tdist:
        k = stats.t.ppf(1 - ((1 - p) / 2), pred[2], 0, 1)
    # print(f"tdist {tdist} k {k}")
    # k = k[:, np.newaxis, :]
    else:
        k = stats.norm.ppf([1 - ((1 - p) / 2)], 0, 1)
        # print(k)
        # print(f"tdist {tdist} k {k}")
        # k = k[:, np.newaxis, np.newaxis]
    squared_errors = (actual["y"] - pred[0]) ** 2
    radius_squared = k**2 * pred[1]
    isinside = np.sum(squared_errors / radius_squared, axis=2) < 1
    return np.mean(isinside)


def csi_regression(actual, pred, p, tdist=False, chebychev=False):
    if chebychev:
        k = np.sqrt(1 / (1 - p))
    elif tdist:
        k = stats.t.ppf([1 - ((1 - p) / 2)], pred[2], 0, 1)

    else:
        k = stats.norm.ppf([1 - ((1 - p) / 2)], 0, 1)
    abs_errors = np.abs(actual["y"] - pred[0].transpose()[0])

    isinside = abs_errors < (k * np.sqrt(pred[1].transpose()))[0]
    return np.mean(isinside)


def plot_results(testdata, predictions, p, index=None, tdist=False):
    if index is None:
        index = np.random.randint(testdata["y"].shape[0])
    print(f"index is {index}")
    if tdist:
        k = stats.t.ppf([1 - ((1 - p) / 2)], predictions[2][index, :, :], 0, 1)
    else:
        k = stats.norm.ppf([1 - ((1 - p) / 2)], 0, 1)

    fig, ax = plt.subplots()
    # create input series
    data_input = testdata["x"][index, :, :]
    plt.scatter(data_input[:, 0], data_input[:, 1], color="blue", marker="o")
    # create true output
    data_output = testdata["y"][index, :, :]
    plt.scatter(
        data_output[:, 0], data_output[:, 1], color="mediumaquamarine", marker="^"
    )
    # create predictions
    pred_output = predictions[0][index, :, :]
    plt.scatter(pred_output[:, 0], pred_output[:, 1], color="red", marker="+")
    # Create ellipses
    pred_std = predictions[1][index, :, :]
    for i in range(testdata["y"].shape[1]):
        if tdist:
            print(k[i, 0])
            ellipse = Ellipse(
                xy=(pred_output[i, 0], pred_output[i, 1]),
                width=2 * k[i, 0] * np.sqrt(pred_std[i, 0]),
                height=2 * k[i, 1] * np.sqrt(pred_std[i, 1]),
                edgecolor="r",
                fc="None",
                lw=1.5,
            )
        else:
            ellipse = Ellipse(
                xy=(pred_output[i, 0], pred_output[i, 1]),
                width=2 * k * np.sqrt(pred_std[i, 0]),
                height=2 * k * np.sqrt(pred_std[i, 1]),
                edgecolor="r",
                fc="None",
                lw=1.5,
            )
        ax.add_patch(ellipse)


def plot_data(data, index=None):
    if index is None:
        index = np.random.randint(data["y"].shape[0])
    print(f"index is {index}")
    fig, ax = plt.subplots()
    # create input series
    data_input = data["x"][index, :, :]
    plt.scatter(data_input[:, 0], data_input[:, 1], color="blue", marker="o")
    # create true output
    data_output = data["y"][index, :, :]
    plt.scatter(
        data_output[:, 0], data_output[:, 1], color="mediumaquamarine", marker="^"
    )


# Create some graphs
