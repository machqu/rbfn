import copy
import torch
import torch.nn
import torch.utils.data

from rbfn import rbfn


class LRScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    A small tweak to PyTorch's ReduceLROnPlateau learning rate scheduler to
    determine when to stop training.

    NB: This might break if the implementation of the parent class changes.
    """

    def __init__(self, *args, **kwargs):
        super(LRScheduler, self).__init__(*args, **kwargs)
        self.num_reduction_steps = 0

    def _reduce_lr(self, epoch):
        super(LRScheduler, self)._reduce_lr(epoch)
        self.num_reduction_steps += 1


def compute_mse(dataset, model, batch_size=32):
    """
    Iterates over batches in a data set and computes the MSE for a given model.

    :param dataset: PyTorch data set
    :param model: The model to be evaluated
    :param batch_size: batch size (only affects the execution time)
    """

    loss_func = torch.nn.MSELoss(reduction='sum')
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False
    )
    loss = 0.0
    cnt = 0
    for X, y in dataloader:
        y_pred = model(X).reshape((-1,))
        loss += loss_func(y, y_pred).item()
        cnt += len(y)
    return loss / cnt


def train_rbfn(dataset_train,
               dataset_validation,
               num_centroids,
               batch_size,
               lr_initial,
               lr_factor,
               lr_steps,
               lr_patience,
               lr_cooldown,
               weight_decay,
               verbose=False,
               refresh=10
               ):
    num_inputs = len(dataset_train[0][0])
    model = rbfn.RBFN(num_inputs, num_centroids)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_initial,
        weight_decay=weight_decay
    )

    # Initialize a custom LR scheduler
    scheduler = LRScheduler(
        optimizer,
        factor=lr_factor,
        patience=lr_patience,
        cooldown=lr_cooldown,
        threshold=0,
        threshold_mode='abs',
        verbose=verbose
    )

    loss_func = torch.nn.MSELoss(reduction='sum')

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)

    # Main training loop. Continues training until the learning rate has been
    # reduced lr_steps times.
    best = {
        'rbfn': None,
        'validation_mse': float('inf'),
        'train_mse': float('inf')
    }
    epoch_ix = 0
    while scheduler.num_reduction_steps <= lr_steps:
        train_loss = 0.0
        train_cnt = 0
        for X, y in dataloader_train:
            optimizer.zero_grad()
            y_pred = model(X).reshape((-1,))
            loss = loss_func(y, y_pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_cnt += len(y)
        train_mse = train_loss / train_cnt

        validation_mse = compute_mse(dataset_validation, model, batch_size)
        scheduler.step(validation_mse)

        # Save the best RBFN
        if validation_mse < best['validation_mse']:
            best['rbfn'] = copy.deepcopy(model)
            best['validation_mse'] = validation_mse
            best['train_mse'] = train_mse
            new_best = True
        else:
            new_best = False

        if verbose and (epoch_ix % refresh == 0):
            print("train_rbfn: epoch={}, train_mse={:.2e}, validation_mse={:.2e} {}".format(
                epoch_ix, train_mse, validation_mse, "(*)" if new_best else ""))

        epoch_ix += 1

    return best


def prune_rbfn(old_rbfn,
               pruning_loss_class,
               num_centroids,
               num_attempts,
               lr_initial,
               lr_steps,
               lr_factor,
               lr_patience,
               lr_cooldown,
               pruning_loss_kwargs={},
               verbose=False,
               refresh=1000
               ):
    if not issubclass(pruning_loss_class, rbfn.RBFNPruningLoss):
        raise RuntimeError("invalid pruning_loss_class")

    loss_func = pruning_loss_class(old_rbfn, **pruning_loss_kwargs)
    new_rbfn = rbfn.RBFN(old_rbfn.num_inputs, num_centroids)

    best = {
        'rbfn': None,
        'loss': float('inf')
    }

    # Do pruning with multiple restarts in order to avoid local minima
    for attempt_ix in range(1, 1 + num_attempts):
        new_rbfn.initialize(old_rbfn)

        optimizer = torch.optim.Adam(
            new_rbfn.parameters(), lr=lr_initial)

        scheduler = LRScheduler(
            optimizer,
            factor=lr_factor,
            patience=lr_patience,
            cooldown=lr_cooldown,
            threshold=0,
            threshold_mode='abs',
            verbose=verbose
        )

        iteration = 0

        # Main pruning loop. Continues until the learning rate has been
        # reduced lr_steps times.
        while scheduler.num_reduction_steps <= lr_steps:
            optimizer.zero_grad()
            loss = loss_func(new_rbfn)
            loss.backward()
            optimizer.step()

            loss = loss.item()
            scheduler.step(loss)
            if loss < best['loss']:
                best['rbfn'] = copy.deepcopy(new_rbfn)
                best['loss'] = loss
                new_best = True
            else:
                new_best = False

            if verbose and (iteration % refresh == 0):
                print("prune_rbfn: attempt={} iteration={} loss={:.2e} {}".format(
                    attempt_ix, iteration, loss, "(*)" if new_best else ""
                ))

            iteration += 1

    return best
