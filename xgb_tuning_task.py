#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# This script executes a given number of random hyperparameter search
# steps for the XGBoost regressor with the Cu migration barrier data set.
#

import json
import argparse
import numpy as np
import scipy.stats
import sklearn.model_selection
import xgboost as xgb
import os

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
    parser.add_argument("--num-estimators", type=int, default=10000,
                        help="XGBoost: maximum number of estimators")
    parser.add_argument("--early-stopping-rounds", type=int, default=10,
                        help="XGBoost: early stopping parameter")
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    X, y = dataset.load_cu_migration_barriers(args.surface)

    # Sample the hyperparameters
    params = sklearn.model_selection.ParameterSampler(
        {
            'learning_rate': scipy.stats.uniform(0.01, 0.09),
            'subsample': [x/10 for x in range(5, 11)],
            'colsample_bytree': [x/10 for x in range(5, 11)],
            'max_depth': scipy.stats.randint(3, 16),
            'min_child_weight': scipy.stats.randint(1, 11),
            'gamma': [0, 0.01, 0.1, 1, 10],
            'reg_alpha': [0, 0.01, 0.1, 1, 10],
            'reg_lambda': [0, 0.01, 0.1, 1, 10],
        }, n_iter=1)
    params = list(params)[0]
    print(params)

    scores = []
    for _ in range(args.num_experiments):
        # Split the data set
        data = util.split_data(
            X, y, args.num_training_points, args.num_validation_points)

        # Train the model
        bst = xgb.XGBRegressor(objective="reg:squarederror",
                               n_estimators=args.num_estimators,
                               n_jobs=args.num_threads,
                               **params)
        bst.fit(data['train']['X'], data['train']['y'],
                early_stopping_rounds=args.early_stopping_rounds,
                eval_set=[(data['validation']['X'], data['validation']['y'])])

        # Predict for the test set
        y_pred = bst.predict(data['test']['X'],
                             ntree_limit=bst.best_ntree_limit,
                             validate_features=False)
        scores.append(np.sqrt(np.mean((data['test']['y'] - y_pred)**2)).item())

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
