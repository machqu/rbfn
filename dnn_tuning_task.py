#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# This script executes a given number of random hyperparameter search
# steps for the DNN regressor with the Cu migration barrier data set.
#

import json
import argparse
import numpy as np
import scipy.stats
import sklearn.model_selection
import torch
import torch.utils.data
import os

import dnn
import util
import util.dataset as dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True,
                        choices=dataset.valid_surfaces)
    parser.add_argument("--seed", required=True, type=int, default=42)
    parser.add_argument("--num-training-points", type=int, default=100000,
                        help="Number of points in the training set")
    parser.add_argument("--num-validation-points", type=int, default=10000,
                        help="Number of points in the validation set")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-experiments", type=int, default=10,
                        help="Number of trials")
    parser.add_argument("--early-stopping-rounds", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X, y = dataset.load_cu_migration_barriers(args.surface)
    num_features = X.shape[1]

    # Sample the hyperparameters
    params = sklearn.model_selection.ParameterSampler(
        {
            'learning_rate': [1e-2, 1e-3, 1e-4],
            'batch_size': [32, 64, 128, 256],
            'weight_decay': [0] + [10**(-x) for x in range(1, 6)],
            'num_layers': scipy.stats.randint(3, 10),
            'num_nodes_per_layer': [32, 64, 128, 256],
            'activation': ["relu", "elu"],
        }, n_iter=1)
    params = list(params)[0]
    print(params)

    scores = []
    for _ in range(args.num_experiments):
        data = util.split_data(
            X, y, args.num_training_points, args.num_validation_points)

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

        train_res = dnn.train(
            dataset_train=dataset_train,
            dataset_validation=dataset_validation,
            dataset_test=dataset_test,
            params=params,
            num_features=num_features,
            early_stopping_rounds=args.early_stopping_rounds,
            rmse_scaling_factor=sigma_y
        )

        scores.append(train_res['test_rmse'])

    results = {'avg_score': np.mean(scores),
               'scores': scores}
    results.update({'params': params})
    results.update({'args': vars(args)})

    print(json.dumps(results, indent=2))

    # Save the result
    fp = open(args.output_file, "w")
    json.dump(results, fp, indent=2)
    fp.close()


if __name__ == "__main__":
    main()
