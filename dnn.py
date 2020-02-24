import copy
import math
import torch
import torch.utils.data
import numpy as np


def _default_input_transform(X):
    return 2 * X - 1


class DNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_layers,
        num_nodes_per_layer,
        activation,
        input_transform=_default_input_transform
    ):
        super(DNN, self).__init__()

        self.num_features = num_features
        self.num_layers = num_layers
        self.num_nodes_per_layer = num_nodes_per_layer
        self.input_transform = input_transform

        if activation not in ["relu", "elu"]:
            raise RuntimeError(
                "invalid activation function {}".format(activation))
        else:
            self.activation = activation
            activation_cls = {
                'relu': torch.nn.ReLU,
                'elu': torch.nn.ELU
            }[activation]

        modules = []
        modules.append(torch.nn.Linear(num_features, num_nodes_per_layer))
        modules.append(activation_cls())
        for _ in range(num_layers - 2):
            modules.append(torch.nn.Linear(
                num_nodes_per_layer, num_nodes_per_layer))
            modules.append(activation_cls())
        modules.append(torch.nn.Linear(num_nodes_per_layer, 1))

        self.model = torch.nn.Sequential(*modules)

    def forward(self, X):
        X = self.input_transform(X)
        return self.model(X).reshape((-1,))


def train(
    dataset_train,
    dataset_validation,
    dataset_test,
    params,
    num_features,
    early_stopping_rounds,
    verbose=True,
    rmse_scaling_factor=1
):
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=params['batch_size'], shuffle=True)
    dataloader_validation = torch.utils.data.DataLoader(
        dataset_validation, batch_size=params['batch_size'], shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=params['batch_size'], shuffle=False)

    loss_func = torch.nn.MSELoss(reduction='sum')

    model = DNN(num_features=num_features,
                num_layers=params['num_layers'],
                num_nodes_per_layer=params['num_nodes_per_layer'],
                activation=params['activation'])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'])

    best_model = None
    best_loss = float("inf")
    epoch_ix = 1
    num_no_improvement = 0
    while num_no_improvement < early_stopping_rounds:
        train_loss = 0
        for X, y in dataloader_train:
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_func(y_pred, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(dataset_train)

        validation_loss = 0
        with torch.no_grad():
            for X, y in dataloader_validation:
                y_pred = model(X)
                validation_loss += loss_func(y_pred, y).item()
        validation_loss /= len(dataset_validation)
        if validation_loss < best_loss:
            num_no_improvement = 0
            best_model = copy.deepcopy(model)
            best_loss = validation_loss
        else:
            num_no_improvement += 1

        if verbose:
            print("epoch=%d train_rmse=%.4f validation_rmse=%.4f" % (
                epoch_ix,
                rmse_scaling_factor * np.sqrt(train_loss),
                rmse_scaling_factor * np.sqrt(validation_loss)
            ))
        epoch_ix += 1

    # Compute the test set RMSE
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader_test:
            y_pred = best_model(X)
            test_loss += loss_func(y_pred, y).item()
    test_loss /= len(dataset_test)
    test_rmse = math.sqrt(test_loss)
    test_rmse *= rmse_scaling_factor

    return {
        'model': best_model,
        'test_rmse': test_rmse
    }
