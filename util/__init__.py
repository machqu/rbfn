import numpy as np
import sklearn.datasets
import sklearn.model_selection
import torch


def split_data(X, y, num_training_points, num_validation_points):
    """
    Split a data set into train, test, and validation sets.
    """
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(
            X, y,
            train_size=num_training_points + num_validation_points
        )

    X_train, X_validation, y_train, y_validation = \
        sklearn.model_selection.train_test_split(
            X_train, y_train,
            train_size=num_training_points
        )

    return {
        'train': {'X': X_train, 'y': y_train},
        'validation': {'X': X_validation, 'y': y_validation},
        'test': {'X': X_test, 'y': y_test}
    }


class NumPyDataset(torch.utils.data.Dataset):
    """
    PyTorch compatible Dataset class for NumPy data.
    Converts everything to 32-bit floats.
    """

    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, ix):
        return self.X[ix], self.y[ix]
