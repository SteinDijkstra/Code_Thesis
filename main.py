from utility import base, config
from data_preparation import generate_data, clean_data
from experiment import regression_problem, pedestrian_problem, analyze

import numpy as np
import pandas as pd


def main():
    path = base.get_absolute_path()
    options = config.read_yaml(path / "config/config.yaml")

    # train, test, _, _, _, _ = analyze.load_data(
    #     path / "results/Experiment_3_pedestrian_eth_seq/"
    # )
    # pedestrian_problem.run_experiment(train, test, options, "eth_seq")

    # train, test, _, _, _, _ = analyze.load_data(
    #     path / "results/Experiment_4_pedestrian_hotel_seq/"
    # )
    # pedestrian_problem.run_experiment(train, test, options, "hotel_seq")
    # pedestrian_problem.run_experiment(train, test, options, "zara01")
    # pedestrian_problem.run_experiment(train, test, options, "zara02")

    # data = pd.read_parquet("./data/dataset1.parquet")
    train = generate_data.generate_dataset1(2000, -3, 4)
    test = generate_data.generate_dataset1(2000, -5, 7)

    # generate_data.save_data(train, test, path / "data/preprocessed/regression.pkl")
    # train, test = generate_data.load_data(path / "data/preprocessed/regression.pkl")
    regression_problem.run_experiment(train, test, options)


if __name__ == "__main__":
    main()
