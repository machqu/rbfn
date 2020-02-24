#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import math
import pickle
import gzip
import copy
import numpy as np
import xgboost as xgb
import torch
import torch.utils.data
import os

import dnn
import util
import util.dataset as dataset
from rbfn.train import train_rbfn, compute_mse


def load_baseline_params(path, surface, prefix):
    """
    Loads the best baseline model (XGBoost or DNN) parameters for a given surface.

    :param path: The path that contains the optimized hyperparameters.
    :param surface: Surface type.

    :return: A dictionary that contains the hyperparameters.
    """

    target_file = os.path.join(path, "best-{}-{}.json".format(prefix, surface))
    return json.load(open(target_file))


def main():
    """
    Trains an RBFN, XGBoost, or DNN model, saves the model, and computes
    test set statistics.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True,
                        choices=dataset.valid_surfaces)
    parser.add_argument("--model", required=True,
                        choices=['rbfn', 'xgb', 'dnn'])
    parser.add_argument("--seed", required=True, type=int, default=42)
    parser.add_argument("--output-file-model", required=True, type=str,
                        help="Output model file name")
    parser.add_argument("--output-file-json", required=True, type=str,
                        help="Output results file name")
    parser.add_argument("--num-training-points", type=int, default=100000,
                        help="Number of points in the training set")
    parser.add_argument("--num-validation-points", type=int, default=10000,
                        help="Number of points in the validation set")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--num-rbfn-centroids", type=int,
                        help="Number of RBFN centroids")
    args = parser.parse_args()

    # Initialize the results dictionary with the command line arguments
    results = vars(args)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load the data set
    X, y = dataset.load_cu_migration_barriers(args.surface)
    data = util.split_data(X, y,
                           args.num_training_points,
                           args.num_validation_points)
    num_features = data['train']['X'].shape[1]

    if args.model == 'xgb':
        # Load the optimized xgb hyperparameters
        xgb_params = load_baseline_params(
            os.path.join("output", "xgb_tuning_combined"),
            args.surface,
            prefix="xgb")

        # Initialize the model (note: the seed is used for subsampling)
        bst = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=xgb_params['args']['num_estimators'],
            n_jobs=args.num_threads,
            random_state=args.seed,
            **xgb_params['params'])

        # Fit the model to the training data
        bst.fit(
            data['train']['X'], data['train']['y'],
            early_stopping_rounds=xgb_params['args']['early_stopping_rounds'],
            eval_set=[(data['validation']['X'], data['validation']['y'])])

        # Save the model
        pickle.dump(bst, gzip.open(args.output_file_model, "wb"))

        # Compute and store the test set RMSE
        y_pred = bst.predict(data['test']['X'],
                             ntree_limit=bst.best_ntree_limit,
                             validate_features=False)
        results['rmse'] = np.sqrt(
            np.mean((data['test']['y'] - y_pred)**2)
        ).item()
    elif args.model == 'dnn':
        # Load the optimized DNN hyperparameters
        dnn_params = load_baseline_params(
            os.path.join("output", "dnn_tuning_combined"),
            args.surface,
            prefix="dnn")

        # Compute response normalization constants
        mu_y = np.mean(data['train']['y']).item()
        sigma_y = np.std(data['train']['y']).item()

        # Set up the data sets
        dataset_train = util.NumPyDataset(
            data['train']['X'],
            (data['train']['y'] - mu_y) / sigma_y)
        dataset_validation = util.NumPyDataset(
            data['validation']['X'],
            (data['validation']['y'] - mu_y) / sigma_y)
        dataset_test = util.NumPyDataset(
            data['test']['X'],
            (data['test']['y'] - mu_y) / sigma_y)

        # Train the model
        train_res = dnn.train(
            dataset_train=dataset_train,
            dataset_validation=dataset_validation,
            dataset_test=dataset_test,
            params=dnn_params['params'],
            num_features=num_features,
            early_stopping_rounds=dnn_params['args']['early_stopping_rounds'],
            rmse_scaling_factor=sigma_y
        )

        # Save the model
        torch.save(
            train_res['model'].state_dict(),
            gzip.open(args.output_file_model, "wb"))

        # Compute the test set RMSE
        results['rmse'] = train_res['test_rmse']
    elif args.model == 'rbfn':
        if args.num_rbfn_centroids is None:
            raise RuntimeError("must provide a value for --num-rbfn-centroids")

        # Compute response normalization constants
        mu_y = np.mean(data['train']['y'])
        sigma_y = np.std(data['train']['y'])
        # And store them in the results dictionary for later use
        results["mu_y"] = mu_y.item()
        results["sigma_y"] = sigma_y.item()

        # Prepare the data sets objects
        dataset_train = util.NumPyDataset(
            2 * data['train']['X'] - 1,
            (data['train']['y'] - mu_y) / sigma_y)
        dataset_validation = util.NumPyDataset(
            2 * data['validation']['X'] - 1,
            (data['validation']['y'] - mu_y) / sigma_y)
        dataset_test = util.NumPyDataset(
            2 * data['test']['X'] - 1,
            (data['test']['y'] - mu_y) / sigma_y)

        # Train an RBFN
        res_train = train_rbfn(
            dataset_train=dataset_train,
            dataset_validation=dataset_validation,
            num_centroids=args.num_rbfn_centroids,
            batch_size=64,
            lr_initial=1e-2,
            lr_factor=0.1,
            lr_steps=2,
            lr_patience=10,
            lr_cooldown=10,
            weight_decay=1e-5,
            verbose=True,
            refresh=1
        )

        # Save the model
        torch.save(
            res_train['rbfn'].state_dict(),
            gzip.open(args.output_file_model, "wb"))

        # Compute the test set RMSE
        results['rmse'] = \
            sigma_y * math.sqrt(compute_mse(dataset=dataset_test,
                                            model=res_train['rbfn'],
                                            batch_size=64))

    print(json.dumps(results, indent=2))

    # Save the result
    fp = open(args.output_file_json, "w")
    json.dump(results, fp, indent=2)
    fp.close()


if __name__ == "__main__":
    main()
