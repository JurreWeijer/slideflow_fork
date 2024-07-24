import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Union, Tuple
from torch import nn
from torch import Tensor
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
from fastai.vision.all import (
    DataLoader, DataLoaders, Learner, RocAuc, SaveModelCallback, CSVLogger, FetchPredsCallback
)
from fastai.metrics import mae
from slideflow import log
import slideflow.mil.data as data_utils
from slideflow.model import torch_utils
from .._params import TrainerConfigFastAI, ModelConfigCLAM
import logging

from lifelines.utils import concordance_index

# -----------------------------------------------------------------------------

def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return -log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)

class CoxPHLoss(nn.Module):
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return cox_ph_loss(log_h, durations, events, self.eps)

# Define a custom metric for the concordance index
class ConcordanceIndex:
    def __init__(self):
        self.name = "concordance_index"

    def __call__(self, preds, targets):
        logging.info(f"preds: {preds}")
        logging.info(f"targets: {targets}")
        durations, events = targets[:, 0], targets[:, 1]
        logging.info(f"durations: {durations}")
        logging.info(f"events: {events}")
        ci = concordance_index(durations.cpu().numpy(), preds.cpu().numpy(), events.cpu().numpy())
        logging.info(f"Concordance Index: {ci}")
        return ci

def train(learner, config, callbacks=None):
    """Train an attention-based multi-instance learning model with FastAI.

    Args:
        learner (``fastai.learner.Learner``): FastAI learner.
        config (``TrainerConfigFastAI``): Trainer and model configuration.

    Keyword args:
        callbacks (list(fastai.Callback)): FastAI callbacks. Defaults to None.
    """
    cbs = [
        SaveModelCallback(fname=f"best_valid", monitor=config.save_monitor),
        CSVLogger(),
    ]
    if callbacks:
        cbs += callbacks
    if config.fit_one_cycle:
        if config.lr is None:
            #Try lr.find to get the learning rate
            try:
                lr = learner.lr_find().valley
            except:
                #If lr.find fails, try again until it works
                while True:
                    try:
                        lr = learner.lr_find().valley
                        break
                    except:
                        print("lr.find failed, trying again")
                        pass

            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit_one_cycle(n_epoch=config.epochs, lr_max=lr, cbs=cbs)
    else:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit(n_epoch=config.epochs, lr=lr, wd=config.wd, cbs=cbs)
    return learner

# -----------------------------------------------------------------------------

def build_learner(config, *args, **kwargs) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for training an MIL model.

    Args:
        config (``TrainerConfigFastAI``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:
        fastai.learner.Learner, (int, int): FastAI learner and a tuple of the
            number of input features and output classes.

    """
    if isinstance(config.model_config, ModelConfigCLAM):
        return _build_clam_learner(config, *args, **kwargs)
    else:
        return _build_fastai_learner(config, *args, **kwargs)


def _build_clam_learner(
    config: TrainerConfigFastAI,
    bags: List[str],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for a CLAM model.

    Args:
        config (``TrainerConfigFastAI``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:
        FastAI Learner, (number of input features, number of classes).
    """
    from ..clam.utils import loss_utils

    # Prepare device.
    device = torch.device(device if device else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Determine problem type and set the appropriate loss function
    problem_type = determine_problem_type(targets)
    logging.info(f"Problem type: {problem_type}")

    if problem_type == "classification":
        if version.parse(sklearn_version) < version.parse("1.2"):
            oh_kw = {"sparse": False}
        else:
            oh_kw = {"sparse_output": False}
        encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None  # No encoder needed for regression or survival

    # Build datasets and dataloaders.
    train_dataset = data_utils.build_clam_dataset(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        bag_size=config.bag_size
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False,
        **dl_kwargs
    )
    val_dataset = data_utils.build_clam_dataset(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        bag_size=None
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        **dl_kwargs
    )

    # Prepare model.
    batch = next(iter(train_dl))
    n_in, n_out = batch[0][0].shape[-1], batch[-1].shape[-1]
    logging.info(f"Training model {config.model_fn.__name__} (in={n_in}, out={n_out}, loss={config.loss_fn.__name__})")
    model = config.build_model(size=[n_in] + config.model_fn.sizes[config.model_config.model_size][1:], n_classes=n_out)

    if hasattr(model, 'relocate'):
        model.relocate()

    # Set the appropriate loss function and metrics
    if problem_type == "classification":
        counts = pd.value_counts(targets[train_idx])
        weight = counts.sum() / counts
        weight /= weight.sum()
        weight = torch.tensor(
            list(map(weight.get, encoder.categories_[0])), dtype=torch.float32
        ).to(device)
        loss_func = nn.CrossEntropyLoss(weight=weight)
        metrics = [RocAuc()]
    elif problem_type == "regression":
        loss_func = nn.MSELoss()
        metrics = [mae]
    elif problem_type == "survival":
        loss_func = CoxPHLoss()
        metrics = [ConcordanceIndex()]
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    logging.info(f"Based on {problem_type} problem, using loss function: {loss_func}")

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=metrics, path=outdir)

    return learner, (n_in, n_out)


def determine_problem_type(targets: np.ndarray) -> str:

    # Check if targets contain both time and event fields
    if isinstance(targets, np.ndarray):
        return "survival"

    # Determine if the target is regression or classification
    if targets.dtype.char in np.typecodes['AllFloat']:
        return "regression"

    unique_values = np.unique(targets)
    num_unique_values = len(unique_values)

    # Assuming binary classification if the target has two unique values
    if num_unique_values == 2:
        return "classification"

    # Assuming multiclass classification if the target has more than two unique values
    if num_unique_values < (0.1 * targets.size):
        return "classification"

    return "unknown"
    
def _build_fastai_learner(
    config,
    bags: List[str],
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    unique_categories: np.ndarray,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for an MIL model.

    Args:
        config (``TrainerConfigFastAI``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:
        FastAI Learner, (number of input features, number of classes).
    """
    # Prepare device.
    device = torch.device(device if device else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Determine problem type and set the appropriate loss function
    problem_type = determine_problem_type(targets)
    logging.info(f"Problem type: {problem_type}")

    if problem_type == "classification":
        if version.parse(sklearn_version) < version.parse("1.2"):
            oh_kw = {"sparse": False}
        else:
            oh_kw = {"sparse_output": False}
        encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None  # No encoder needed for regression or survival

    # Build datasets and dataloaders.
    train_dataset = data_utils.build_dataset(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        bag_size=config.bag_size,
        use_lens=config.model_config.use_lens
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=config.drop_last,
        **dl_kwargs
    )
    val_dataset = data_utils.build_dataset(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        bag_size=None,
        use_lens=config.model_config.use_lens
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        **dl_kwargs
    )

    # Prepare model.
    batch = next(iter(train_dl))
    n_in, n_out = batch[0].shape[-1], batch[-1].shape[-1]
    logging.info(f"Training model {config.model_fn.__name__} (in={n_in}, out={n_out}, loss={config.loss_fn.__name__})")
    model = config.build_model(n_in, n_out).to(device)
    if hasattr(model, 'relocate'):
        model.relocate()

    # Set the appropriate loss function and metrics
    if problem_type == "classification":
        counts = pd.value_counts(targets[train_idx])
        weight = counts.sum() / counts
        weight /= weight.sum()
        weight = torch.tensor(
            list(map(weight.get, encoder.categories_[0])), dtype=torch.float32
        ).to(device)
        loss_func = nn.CrossEntropyLoss(weight=weight)
        metrics = [RocAuc()]
    elif problem_type == "regression":
        loss_func = nn.MSELoss()
        metrics = [mae]
    elif problem_type == "survival":
        loss_func = CoxPHLoss()
        metrics = [ConcordanceIndex()]
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    logging.info(f"Based on {problem_type} problem, using loss function: {loss_func}")

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=metrics, path=outdir)

    return learner, (n_in, n_out)


def _build_multimodal_learner(
    config: TrainerConfigFastAI,
    bags: List[List[str]],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    n_magnifications: int,
    *,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for an MIL model.

    Args:
        config (``TrainerConfigFastAI``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:

        FastAI Learner, (number of input features, number of classes).
    """
    # Prepare device.
    device = torch_utils.get_device(device)

    # Prepare data.
    # Set oh_kw to a dictionary of keyword arguments for OneHotEncoder,
    # using the argument sparse=False if the sklearn version is <1.2
    # and sparse_output=False if the sklearn version is >=1.2.
    if version.parse(sklearn_version) < version.parse("1.2"):
        oh_kw = {"sparse": False}
    else:
        oh_kw = {"sparse_output": False}
    encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))

    # Build dataloaders.
    train_dataset = data_utils.build_multibag_dataset(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        bag_size=config.bag_size,
        n_bags=n_magnifications,
        use_lens=config.model_config.use_lens
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=config.drop_last,
        device=device,
        **dl_kwargs
    )
    val_dataset = data_utils.build_multibag_dataset(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        bag_size=None,
        n_bags=n_magnifications,
        use_lens=config.model_config.use_lens
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        device=device,
        **dl_kwargs
    )

    # Prepare model.
    batch = train_dl.one_batch()  # batch returns features, lens, and targets
    if config.model_config.use_lens:
        n_in = [b[0].shape[-1] for b in batch[:-1]]
    else:
        n_in = [b.shape[-1] for b in batch[:-1][0]]
    n_out = batch[-1].shape[-1]

    log.info(f"Training model [bold]{config.model_fn.__name__}[/] "
             f"(in={n_in}, out={n_out}, loss={config.loss_fn.__name__})")
    model = config.build_model(n_in, n_out).to(device)
    if hasattr(model, 'relocate'):
        model.relocate()

    # Loss should weigh inversely to class occurences.
    counts = pd.value_counts(targets[train_idx])
    weight = counts.sum() / counts
    weight /= weight.sum()
    weight = torch.tensor(
        list(map(weight.get, encoder.categories_[0])), dtype=torch.float32
    ).to(device)
    loss_func = nn.CrossEntropyLoss(weight=weight)

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=[RocAuc()], path=outdir)

    return learner, (n_in, n_out)
