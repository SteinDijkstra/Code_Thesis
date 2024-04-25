from models import mcdropout, evidential, evidential_novel, ensemble
from utility.config import Options
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import sys, os
import time


def run_experiment(data, testdata, options, datasetname):
    # Create folder with date+time+pedestrian & 4 subfolders
    outputfolder = init_folder(data, testdata, options, datasetname)

    # pedestrian
    model_evidential, result_evidential, time_evidential = pedestrian_evidential(
        data, testdata, options
    )
    saveresult(
        options,
        model_evidential,
        result_evidential,
        time_evidential,
        outputfolder,
        "evidential",
    )

    # pedestrian
    model_evidential_new, result_evidential_new, time_evidential_new = (
        pedestrian_evidential_new(data, testdata, options)
    )
    saveresult(
        options,
        model_evidential_new,
        result_evidential_new,
        time_evidential_new,
        outputfolder,
        "evidential_new",
    )

    # pedestrian
    model_dropout, result_dropout, time_dropout = pedestrian_mcdropout(
        data, testdata, options
    )
    saveresult(
        options, model_dropout, result_dropout, time_dropout, outputfolder, "mcdropout"
    )

    # pedestrian
    model_ensemble, result_ensemble, time_ensemble = pedestrian_ensemble(
        data, testdata, options
    )
    saveresult(
        options,
        model_ensemble,
        result_ensemble,
        time_ensemble,
        outputfolder,
        "ensemble",
    )

    return result_evidential, result_evidential_new, result_dropout, result_ensemble


def pedestrian_evidential(data, testdata, options):
    model_evidential = evidential.create_model_pedestrian(options)

    start_train = time.time()
    evidential.train_model(model_evidential, data, options)
    end_train = time.time()

    start_pred = time.time()
    mean, var, dof = evidential.predict_model(model_evidential, testdata["x"], options)
    end_pred = time.time()

    return (
        model_evidential,
        (mean, var, dof),
        (end_train - start_train, end_pred - start_pred),
    )


def pedestrian_evidential_new(data, testdata, options):
    model_evidential_new = evidential_novel.create_model_pedestrian(options)

    start_train = time.time()
    evidential_novel.train_model(model_evidential_new, data, options)
    end_train = time.time()

    start_pred = time.time()
    mean, var, dof = evidential_novel.predict_model(
        model_evidential_new, testdata["x"], options
    )
    end_pred = time.time()

    return (
        model_evidential_new,
        (mean, var, dof),
        (end_train - start_train, end_pred - start_pred),
    )


def pedestrian_mcdropout(data, testdata, options):
    model_mcdropout = mcdropout.create_model_pedestrian(data, options)

    start_train = time.time()
    mcdropout.train_model(model_mcdropout, data, options)
    end_train = time.time()

    start_pred = time.time()
    mean, var = mcdropout.predict_model(
        model_mcdropout, np.array(testdata["x"]), options
    )
    end_pred = time.time()

    return (
        model_mcdropout,
        (mean, var),
        (end_train - start_train, end_pred - start_pred),
    )


def pedestrian_ensemble(data, testdata, options):
    model_ensemble = ensemble.create_model_pedestrian(options)

    start_train = time.time()
    ensemble.train_model(model_ensemble, data, options)
    end_train = time.time()

    start_pred = time.time()
    mean, var = ensemble.predict_model(model_ensemble, testdata["x"], options)
    end_pred = time.time()

    return ensemble, (mean, var), (end_train - start_train, end_pred - start_pred)


def init_folder(data, testdata, options, datasetname):
    if options.outputoptions.save_output:
        outputdir = options.outputoptions.output_location
        experiment = (
            f"Experiment_{len(os.listdir('./results'))+1}_pedestrian_{datasetname}"
        )
        os.makedirs(outputdir / experiment)
        with open(outputdir / experiment / "data.pkl", "wb") as f:
            pickle.dump(data, f)
        with open(outputdir / experiment / "testdata.pkl", "wb") as f:
            pickle.dump(testdata, f)
        return outputdir / experiment
    return


def saveresult(options: Options, model, results, time, outputfolder, name):
    if options.outputoptions.save_output:
        path = outputfolder / name
        os.makedirs(path)
        with open(path / "options.pkl", "wb") as f:
            pickle.dump(options, f)
        # with open(path / "model", "wb") as f:
        #    pickle.dump(model, f)
        with open(path / "results.pkl", "wb") as f:
            pickle.dump(results, f)
        with open(path / "time.txt", "w") as f:
            f.write(f"{time}")
    return
