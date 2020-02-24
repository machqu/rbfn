#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pathlib
import json
import pathlib
import gzip
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import os

from rbfn import rbfn
from rbfn.train import prune_rbfn


def prune_best_rbfn(traineval_raw_path,
                    traineval_combined_path,
                    pruning_results_path,
                    seed,
                    num_pruned_centroids,
                    num_pruning_attempts):
    """
    Prunes the best RBFN's for each surface.

    :param traineval_raw_path: Path to trained models and results.
    :param traineval_combined_path: Path to postprocessed traineval results.
    :param pruning_results_path: Path to created pruned models.
    :param seed: Random seed.
    :param num_pruned_centroids: Number of centroids in the pruned RBFN.
    :param num_pruning_attempts: Number of pruning trials.
    """

    num_inputs = 26

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Find the best RBFN predictors for each surface
    df = pd.read_csv(os.path.join(traineval_combined_path, "traineval.csv"))
    df = df[df.model == "rbfn"]
    df = df.loc[df.groupby("surface")["rmse"].idxmin()]

    for _, row in df.iterrows():
        model_fn = os.path.join(
            traineval_raw_path,
            pathlib.Path(row.output_file_model).name
        )

        print("Loading RBFN model for surface {} from file {}".format(
            row.surface, model_fn))

        model = rbfn.RBFN(num_inputs, row.num_rbfn_centroids)
        model.load_state_dict(torch.load(gzip.open(model_fn, "rb")))

        print("Pruning")
        pruned = prune_rbfn(
            old_rbfn=model,
            pruning_loss_class=rbfn.BinaryPruningLoss,
            num_centroids=num_pruned_centroids,
            num_attempts=num_pruning_attempts,
            lr_initial=1e-3,
            lr_factor=0.1,
            lr_steps=2,
            lr_patience=10,
            lr_cooldown=10,
            verbose=True,
            refresh=1000
        )

        # Compute and store the pruning loss
        loss = (row['sigma_y']**2) * pruned['loss']
        loss_fn = os.path.join(pruning_results_path,
                               "{}.loss.txt".format(row.surface))
        fp = open(loss_fn, "w")
        fp.write(str(loss) + "\n")
        fp.close()

        # Save the centroids
        centroid_fn = os.path.join(pruning_results_path,
                                   "{}.centroids.txt".format(row.surface))
        centroids = pruned['rbfn'].Z.detach().numpy().transpose()
        np.savetxt(centroid_fn, centroids)

        # Save the centroid weights
        weight_fn = os.path.join(pruning_results_path,
                                 "{}.weights.txt".format(row.surface))
        weights = pruned['rbfn'].beta.detach().numpy()
        weights *= row['sigma_y']
        np.savetxt(weight_fn, weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pruning", action='store_true')
    parser.add_argument("--pruning-seed", type=int, default=0)
    parser.add_argument("--num-pruned-centroids", type=int, default=16)
    parser.add_argument("--num-pruning-attempts", type=int, default=10)
    args = parser.parse_args()

    traineval_raw_path = os.path.join("output", "traineval_raw")
    traineval_combined_path = os.path.join("output", "traineval_combined")

    # Combine the results from different jobs
    res = []
    p = pathlib.Path(traineval_raw_path)
    for f in p.glob('traineval-*.json'):
        res.append(json.load(f.open()))
    df = pd.DataFrame(res)

    # Write the combined results into a CSV file
    df.to_csv(os.path.join(traineval_combined_path, "traineval.csv"),
              index=False)

    # Prune the best RBFN's if so requested
    if args.no_pruning is False:
        pruning_results_path = os.path.join("output", "pruning")
        prune_best_rbfn(
            traineval_raw_path=traineval_raw_path,
            traineval_combined_path=traineval_combined_path,
            pruning_results_path=pruning_results_path,
            seed=args.pruning_seed,
            num_pruned_centroids=args.num_pruned_centroids,
            num_pruning_attempts=args.num_pruning_attempts
        )


if __name__ == "__main__":
    main()
