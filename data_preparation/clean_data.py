import numpy as np
import pandas as pd
import pickle


def get_cleaned_data(path):
    """Create processed data from a file

    Args:
        path (Path): Location of the original data

    Returns:
        (pd.DataFrame, pd.DataFrame): a tuple with (Train, Test)
    """
    data = load_raw_data(path)
    data = filter_data(data)
    train, test = train_test_split(data)
    # data augmentation
    train = augment_data(train)
    test = augment_data(test)
    return train, test


def load_raw_data(path) -> pd.DataFrame:
    """Load data from file, based on obsmath.txt structure

    Args:
        path (filepath): Location of original data

    Returns:
        pd.DataFrame: unprocessed dataframe
    """
    n_col = 8
    col_names = [
        "frame_number",
        "pedestrian_ID",
        "pos_x",
        "pos_z",
        "pos_y",
        "v_x",
        "v_z",
        "v_y",
    ]
    with open(path, "r") as f:
        data_txt = f.read()
        data_np = np.array(data_txt.split(), dtype=float)
        data_np = data_np.reshape((int(len(data_np) / n_col), n_col))
        data = pd.DataFrame(data_np, columns=col_names)
    return data


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filter data based on a minimum of 20 datapoints

    Args:
        data (pd.DataFrame): data to be filterd

    Returns:
        pd.DataFrame: filtered data
    """
    data = data.drop(["pos_z", "v_z", "v_x", "v_y"], axis=1)
    data = data[data.groupby(["pedestrian_ID"]).transform("size") >= 20]
    return data


def train_test_split(data: pd.DataFrame):
    """Create train test split

    Args:
        data (pd.DataFrame): dataframe with original data

    Returns:
        tuple(pd.Dataframe,pd.DataFrame): (traindata, testdata)
    """

    ids = data["pedestrian_ID"].unique()
    choice = np.random.choice(
        range(ids.shape[0]), size=(int(ids.shape[0] * 0.7),), replace=False
    )
    ind = np.zeros(ids.shape[0], dtype=bool)
    ind[choice] = True
    rest = ~ind

    data_train = data[data["pedestrian_ID"].isin(ids[ind])]
    data_test = data[data["pedestrian_ID"].isin(ids[rest])]

    return data_train, data_test


def augment_data(data: pd.DataFrame) -> pd.DataFrame:
    """Augment the data using a rolling window with stepsize 1

    Args:
        data (pd.Dataframe): data to be augmented

    Returns:
        pd.DataFrame: data where every series is lenght 20 and the number of series should have increased
    """
    x = []
    y = []
    x_size = 8
    y_size = 12
    window_size = x_size + y_size
    step = 1
    for pedestrian in data.groupby(["pedestrian_ID"]):
        df = pedestrian[1].sort_values("frame_number")[["pos_x", "pos_y"]]
        for i in range(0, len(df) - window_size + 1, step):
            x.append(df.iloc[i : i + x_size])
            y.append(df.iloc[i + x_size : i + x_size + y_size])

    x = np.array(x)
    y = np.array(y)
    data = {"x": x, "y": y}
    return data


def save_data(train: pd.DataFrame, test: pd.DataFrame, path):
    """Save data a picklye

    Args:
        train (pd.DataFrame): traindata
        test (pd.DataFrame): testdata
        path (Path): path where to save data
    """
    with open(path, "wb") as f:
        pickle.dump(train, f)
        pickle.dump(test, f)


def load_data(path):
    """Load data from picle

    Args:
        path (Path): where to load data from

    Returns:
        tuple(pd.Dataframe,pd.Dataframe): (traindata,testdata)
    """
    with open(path, "rb") as f:
        train = pickle.load(f)
        test = pickle.load(f)
    return train, test
