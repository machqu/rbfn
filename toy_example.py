#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import sklearn.model_selection
import matplotlib.pyplot as plt
import os

from rbfn import rbfn
from rbfn.train import train_rbfn, prune_rbfn
from util import NumPyDataset


def target_function(x):
    return np.exp(-x ** 2) + 0.2 * np.cos(4 * x)


def sample_dataset(num_points, x_min, x_max):
    """
    Generates uniformly distributed samples between x_min and x_max, and
    computes the target function values.
    """
    X = (x_max - x_min) * np.random.rand(num_points, 1) + x_min
    y = np.squeeze(target_function(X))
    return X, y


def predict(rbfn, X):
    """
    Uses the provided RBFN to provide predictions for a given
    NumPy features array.
    """
    return rbfn(
        torch.from_numpy(
            X.astype(np.float32)
        )
    ).reshape((-1,)).detach().numpy()


def main():
    seed = 0
    num_points = 1000
    x_min = -4.0
    x_max = 4.0
    validation_fraction = 0.2
    num_centroids = 100
    num_pruned_centroids = 3
    num_plot_points = 1000
    num_pruning_attempts = 30
    plot_extra_range = 2.0

    np.random.seed(seed)
    torch.manual_seed(seed)

    X, y = sample_dataset(
        num_points, x_min, x_max)

    X_train, X_validation, y_train, y_validation = \
        sklearn.model_selection.train_test_split(
            X, y, test_size=validation_fraction
        )

    # Prepare the data sets objects
    dataset_train = NumPyDataset(
        X_train, y_train)
    dataset_validation = NumPyDataset(
        X_validation, y_validation)

    # Train an RBFN
    trained = train_rbfn(
        dataset_train=dataset_train,
        dataset_validation=dataset_validation,
        num_centroids=num_centroids,
        batch_size=64,
        lr_initial=1e-2,
        lr_factor=0.1,
        lr_steps=2,
        lr_patience=5,
        lr_cooldown=5,
        weight_decay=1e-5,
        verbose=True, refresh=1
    )

    # Prune the RBF assuming a normal data distribution
    pruned_normal = prune_rbfn(
        old_rbfn=trained['rbfn'],
        pruning_loss_class=rbfn.UnitNormalPruningLoss,
        num_centroids=num_pruned_centroids,
        num_attempts=num_pruning_attempts,
        lr_initial=1e-3,
        lr_factor=0.1,
        lr_steps=2,
        lr_patience=5,
        lr_cooldown=5,
        verbose=True, refresh=1000
    )

    # Prune the RBF assuming a uniform data distribution
    pruned_uniform = prune_rbfn(
        old_rbfn=trained['rbfn'],
        pruning_loss_class=rbfn.UniformPruningLoss,
        pruning_loss_kwargs={'a': x_min, 'b': x_max},
        num_centroids=num_pruned_centroids,
        num_attempts=num_pruning_attempts,
        lr_initial=1e-3,
        lr_factor=0.1,
        lr_steps=2,
        lr_patience=5,
        lr_cooldown=5,
        verbose=True, refresh=1000
    )

    # Compute and store the data points and predictions
    results = []
    results.append(pd.DataFrame({
        'type': 'pred',
        'x': np.squeeze(X),
        'y_true': y,
        'y_rbfn': predict(trained['rbfn'], X),
        'y_rbfn_pruned_normal': predict(pruned_normal['rbfn'], X),
        'y_rbfn_pruned_uniform': predict(pruned_uniform['rbfn'], X)
    }))

    # Compute and store data points for visualization
    x_min_plot = x_min - plot_extra_range
    x_max_plot = x_max + plot_extra_range
    x_visu = np.linspace(x_min_plot, x_max_plot, num_plot_points)
    y_visu_true = target_function(x_visu)
    y_visu_rbfn = predict(trained['rbfn'], np.expand_dims(x_visu, 1))
    y_visu_rbfn_pruned_normal = predict(
        pruned_normal['rbfn'], np.expand_dims(x_visu, 1))
    y_visu_rbfn_pruned_uniform = predict(
        pruned_uniform['rbfn'], np.expand_dims(x_visu, 1))
    results.append(pd.DataFrame({
        'type': 'visu',
        'x': x_visu,
        'y_true': y_visu_true,
        'y_rbfn': y_visu_rbfn,
        'y_rbfn_pruned_normal': y_visu_rbfn_pruned_normal,
        'y_rbfn_pruned_uniform': y_visu_rbfn_pruned_uniform
    }))
    results_df = pd.concat(results)

    # Save the results into a csv file
    path = os.path.join("output", "toy_example")
    results_df.to_csv(os.path.join(path, "toy_example.csv"), index=False)

    # Make a matplotlib plot with results visualisation
    plt.style.use('seaborn-muted')
    ax = results_df.loc[
        results_df.type == 'visu',
        ['x', 'y_true', 'y_rbfn', 'y_rbfn_pruned_normal',
         'y_rbfn_pruned_uniform']
    ].plot(x='x', grid=True, xlim=(x_min_plot, x_max_plot), figsize=(9, 4))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    for x in [x_min, x_max]:
        ax.axvline(x, color="black", linestyle="--")
    for pruned_rbfn, color in [
        (pruned_normal["rbfn"], ax.get_lines()[2].get_color()),
        (pruned_uniform["rbfn"], ax.get_lines()[3].get_color())
    ]:
        xs = pruned_rbfn.Z.detach().squeeze().numpy()
        for x in xs:
            ax.axvline(x, color=color, linestyle=":")

    ax.legend(["exp(-x²) + 0.2 cos(4x)",
               "RBFN, 100 centr.",
               "Pruned RBFN, 3 centr., N(0, 1)",
               "Pruned RBFN, 3 centr., U(-4, 4)"],
              loc='upper center',
              bbox_to_anchor=(0.5, 1.1),
              ncol=4,
              frameon=False,
              prop={'size': 8})
    ax.set_xlabel('')

    # Save the plot
    plt.savefig(os.path.join(path, "toy_example.pdf"),
                dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
