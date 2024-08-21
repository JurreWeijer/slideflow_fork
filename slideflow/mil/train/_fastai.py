import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
import slideflow as sf
from typing import List, Optional, Union, Tuple
from torch import nn
from torch import Tensor
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
from fastai.vision.all import (
    DataLoader, DataLoaders, Learner, RocAuc, SaveModelCallback, CSVLogger, FetchPredsCallback, Callback
)
from fastai.learner import Metric
from fastai.torch_core import to_detach, flatten_check
from fastai.metrics import mae
from slideflow import log
import slideflow.mil.data as data_utils
from slideflow.model import torch_utils
from .._params import TrainerConfigFastAI, ModelConfigCLAM
import logging

from lifelines.utils import concordance_index

#import custom losses and metrics
from pathbench import losses
from pathbench import metrics
from pathbench import augmentations

# -----------------------------------------------------------------------------


class MILAugmentationCallback(Callback):
    def __init__(self, augmentations: list = None):
        self.augmentations = [retrieve_augmentation(x) for x in augmentations]

    def before_batch(self):
        # Apply each augmentation function to the features in the batch
        features = self.learn.xb[0]
        for aug in self.augmentations:
            features = aug(features)
        # Update the features in the batch
        self.learn.xb = (features,) + self.learn.xb[1:]

def retrieve_augmentation(augmentation_name):
    augmentation_class = getattr(augmentations, augmentation_name)
    return augmentation_class()  # Instantiate the augmentation class

def retrieve_custom_loss(loss_name):
    loss_class = getattr(losses, loss_name)
    return loss_class()  # Instantiate the loss class

def retrieve_custom_metric(metric_name):
    metric_class = getattr(metrics, metric_name)
    return metric_class()  # Instantiate the metric class

class PadToMinLength:
    """Pad all tensors in a batch to the minimum length of the longest tensor."""
    def __call__(self, batch):
        # Filter out non-tensor elements
        batch_tensors = [item for item in batch if isinstance(item, torch.Tensor)]

        # Find the minimum length among the tensors that have dimensions
        min_length = min([item.size(0) if item.dim() > 0 else 1 for item in batch_tensors])

        # Pad each tensor to the minimum length
        padded_batch = []
        for item in batch:
            if isinstance(item, torch.Tensor):
                if item.dim() == 0:
                    # If tensor has no dimensions, convert it to a 1-dimensional tensor
                    item = item.unsqueeze(0)
                if item.size(0) > min_length:
                    item = item[:min_length]
                elif item.size(0) < min_length:
                    padding = (0, 0, 0, min_length - item.size(0))  # Adjust this if your tensor has more dimensions
                    item = torch.nn.functional.pad(item, padding)
            padded_batch.append(item)

        return padded_batch

def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast."""
    
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    loss = -log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum().add(eps))
    return loss

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast."""
    
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)

class CoxPHLoss(nn.Module):
    """Loss function for CoxPH model."""
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Predictions from the model.
            targets (torch.Tensor): Target values.
        
        Returns:
            torch.Tensor: Loss value.
        """
        durations = targets[:, 0]
        events = targets[:, 1]
        
        # Check for zero events and handle accordingly
        if torch.sum(events) == 0:
            logging.warning("No events in batch, returning near zero loss")
            return torch.tensor(1e-6, dtype=preds.dtype, device=preds.device)
        
        loss = cox_ph_loss(preds, durations, events).float()
        return loss

class ConcordanceIndex(Metric):
    """Concordance index metric for survival analysis."""
    def __init__(self):
        self.name = "concordance_index"
        self.reset()

    def reset(self):
        """Reset the metric."""
        self.preds, self.durations, self.events = [], [], []

    def accumulate(self, learn):
        """Accumulate predictions and targets from a batch."""
        preds = learn.pred
        targets = learn.y
        self.accum_values(preds, targets)

    def accum_values(self, preds, targets):
        """Accumulate predictions and targets from a batch."""
        preds, targets = to_detach(preds), to_detach(targets)

        # Ensure preds are tensors, handle dict, tuple, and list cases
        if isinstance(preds, dict):
            preds = torch.cat([torch.tensor(v).view(-1) if not isinstance(v, torch.Tensor) else v.view(-1) for v in preds.values()])
        elif isinstance(preds, tuple):
            preds = torch.cat([torch.tensor(p).view(-1) if not isinstance(p, torch.Tensor) else p.view(-1) for p in preds])
        elif isinstance(preds, list):
            preds = torch.cat([torch.tensor(p).view(-1) if not isinstance(p, torch.Tensor) else p.view(-1) for p in preds])
        else:
            preds = preds.view(-1) if isinstance(preds, torch.Tensor) else torch.tensor(preds).view(-1)

        # Handle survival targets (durations and events)
        durations = targets[:, 0].view(-1)
        events = targets[:, 1].view(-1)
        
        self.preds.append(preds)
        self.durations.append(durations)
        self.events.append(events)

    @property
    def value(self):
        """Calculate the concordance index."""
        if len(self.preds) == 0: return None
        preds = torch.cat(self.preds).cpu().numpy()
        durations = torch.cat(self.durations).cpu().numpy()
        events = torch.cat(self.events).cpu().numpy()
        ci = concordance_index(durations, preds, events)
        return ci

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

def train(learner, config, pb_config=None, callbacks=None):
    """Train an attention-based multi-instance learning model with FastAI.

    Args:
        learner (``fastai.learner.Learner``): FastAI learner.
        config (``TrainerConfigFastAI``): Trainer and model configuration.
        pb_config (dict): PathBench configuration. Defaults to None.

    Keyword args:
        callbacks (list(fastai.Callback)): FastAI callbacks. Defaults to None.
    """
    if pb_config is not None:
        cbs = [
            SaveModelCallback(fname=f"best_valid", monitor=pb_config['experiment']['best_epoch_based_on']),
            CSVLogger(),
        ]
    else:
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
                log.info(f"Using auto-detected learning rate: {lr}")
            except:
                lr = 1e-3
                log.info(f"Failed to find learning rate, using default: {lr}")
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

    pb_config = dl_kwargs.get("pb_config", None)
    problem_type = pb_config['experiment']['task']

    # Prepare device.
    device = torch.device(device if device else 'cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Problem type: {problem_type}")

    if problem_type == "classification":
        if version.parse(sklearn_version) < version.parse("1.2"):
            oh_kw = {"sparse": False}
        else:
            oh_kw = {"sparse_output": False}
        encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None  # No encoder needed for regression or survival

    if problem_type == "survival":
        targets = np.array(targets, dtype=float)
        targets[:, 0] = targets[:, 0].astype(int)  # Convert durations to integers
        targets[:, 1] = targets[:, 1].astype(int)  # Convert events to integers

    if problem_type == 'regression' or problem_type == 'survival':
        #Ensure all targets are float32
        targets = targets.astype(np.float32)

    # Build datasets and dataloaders.
    train_dataset = data_utils.build_clam_dataset(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        bag_size=config.bag_size
    )

    if problem_type == "classification":
        train_dl = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            drop_last=False,
            device=device,
            **dl_kwargs
        )
    else:
        train_dl = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=1,
            drop_last=False,
            device=device,
            **dl_kwargs
        )
    val_dataset = data_utils.build_clam_dataset(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        bag_size=None
    )
    if problem_type == "classification":
        batch_size = 1
        val_dl = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
            device=device,
            after_item=PadToMinLength(),
            **dl_kwargs
        )
    else:
        batch_size = config.batch_size
        val_dl = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    persistent_workers=True,
                    device=device,
                    after_item=PadToMinLength(),
                    **dl_kwargs
                )

    logging.info(f"{problem_type} task, because of using CLAM, chosen batch size: {batch_size}")
    # Prepare model.
    batch = next(iter(train_dl))
    n_in, n_out = batch[0][0].shape[-1], batch[-1].shape[-1]
    
    #Check if survival or regression, if so, set n_out to 1
    if problem_type == "survival" or problem_type == "regression":
        n_out = 1

    #Check if multiclass classification
    if problem_type == 'classification' and unique_categories.shape[0] > 2:
        n_out = unique_categories.shape[0]
        print(f"Multiclass classification detected, setting n_out to {n_out}")

    logging.info(f"Training model {config.model_fn.__name__} (in={n_in}, out={n_out}, loss={config.loss_fn.__name__})")

    if 'encoder_activation' in pb_config['experiment']:
        model = config.build_model(size=[n_in] + config.model_fn.sizes[config.model_config.model_size][1:], n_classes=n_out,
                                   task = problem_type, encoder_activation=pb_config['experiment']['encoder_activation']).to(device)
    else:
        model = config.build_model(size=[n_in] + config.model_fn.sizes[config.model_config.model_size][1:], n_classes=n_out,
                                   task = problem_type)

    if hasattr(model, 'relocate'):
        model.relocate()

    if problem_type == "classification":
        counts = pd.value_counts(targets[train_idx])
        weight = counts.sum() / counts
        weight /= weight.sum()
        weight = torch.tensor(
            list(map(weight.get, encoder.categories_[0])), dtype=torch.float32
        ).to(device)
        if 'custom_loss' in pb_config['experiment']:
            loss_func = retrieve_custom_loss(pb_config['experiment']['custom_loss'])
        else:
            loss_func = config.loss_fn()
        if 'custom_metrics' in pb_config['experiment']:
            metrics = [retrieve_custom_metric(x) for x in pb_config['experiment']['custom_metrics']]
        else:
            metrics = [loss_utils.RocAuc()]
    elif problem_type == "regression":
        if 'custom_loss' in pb_config['experiment']:
            loss_func = retrieve_custom_loss(pb_config['experiment']['custom_loss'])
        else:
            loss_func = nn.MSELoss()
        if 'custom_metrics' in pb_config['experiment']:
            metrics = [retrieve_custom_metric(x) for x in pb_config['experiment']['custom_metrics']]
        else:
            metrics = [mae]
    elif problem_type == "survival":
        assert targets.shape[1] == 2 # Duration and event
        if 'custom_loss' in pb_config['experiment']:
            loss_func = retrieve_custom_loss(pb_config['experiment']['custom_loss'])
        else: 
            loss_func = CoxPHLoss()
        if 'custom_metrics' in pb_config['experiment']:
            metrics = [retrieve_custom_metric(x) for x in pb_config['experiment']['custom_metrics']]
        else:
            metrics = [ConcordanceIndex()]
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


    logging.info(f"{problem_type} problem, using loss function: {loss_func}")

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=metrics, path=outdir)

    #If users wants to add mil-friendly augmentations, do so
    if 'augmentations' in pb_config['benchmark_parameters']:
        augmentation_callback = MILAugmentationCallback(pb_config['benchmark_parameters']['augmentations'])
        learner.add_cb(augmentation_callback)

    return learner, (n_in, n_out)

    
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
    """Build a FastAI learner for an MIL model."""

    pb_config = dl_kwargs.get("pb_config", None)
    problem_type = pb_config['experiment']['task']
    num_workers = pb_config['experiment']['num_workers']

    # Select the appropriate loss function based on the problem type
    if problem_type == "classification":
        default_loss = nn.CrossEntropyLoss()
    elif problem_type == "regression":
        default_loss = nn.MSELoss()
    elif problem_type == "survival":
        default_loss = CoxPHLoss()
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
    
    loss_function = dl_kwargs.get("loss", default_loss)
    if loss_function is not None:
        loss_function = retrieve_custom_loss(loss_function)

    # Prepare device.
    device = torch.device(device if device else 'cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Problem type: {problem_type}")

    # Handle encoding for classification
    if problem_type == "classification":
        encoder = OneHotEncoder(sparse_output=False).fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None  # No encoder needed for regression or survival

    # Prepare targets for survival or regression
    if problem_type == "survival" or problem_type == 'regression':
        targets = np.array(targets, dtype=float)
        if problem_type == "survival":
            targets[:, 0] = targets[:, 0].astype(int)  # Convert durations to integers
            targets[:, 1] = targets[:, 1].astype(int)  # Convert events to integers
        targets = torch.tensor(targets, dtype=torch.float32)

    # Build datasets and dataloaders
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
        num_workers=num_workers,
        persistent_workers=True,
        drop_last=config.drop_last,
        device=device,
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
        num_workers=num_workers,
        persistent_workers=True,
        device=device,
        after_item=PadToMinLength(),
        **dl_kwargs
    )

    # Prepare model
    batch = next(iter(train_dl))
    n_in, n_out = batch[0].shape[-1], batch[-1].shape[-1]
    if problem_type == "survival" or problem_type == "regression":
        n_out = 1

    activation_function = dl_kwargs.get("activation_function", None)
    if activation_function is None:
        model = config.build_model(n_in, n_out).to(device)
    else:
        model = config.build_model(n_in, n_out, activation_function=activation_function).to(device)

    # Handle class weights for classification
    if problem_type == "classification":
        counts = pd.value_counts(targets[train_idx])
        weight = counts.sum() / counts
        weight /= weight.sum()
        weight = torch.tensor(
            list(map(weight.get, encoder.categories_[0])), dtype=torch.float32
        ).to(device)
        if loss_function is None:
            loss_function = nn.CrossEntropyLoss(weight=weight)

    # Determine if attention values are required by the loss function
    require_attention = getattr(loss_function, 'require_attention', False)

    # Check if the model supports returning attention
    model_supports_attention = 'return_attention' in model.forward.__code__.co_varnames

    # Define custom forward function if attention is required
    def custom_forward(*args, **kwargs):
        if model_supports_attention and require_attention:
            preds, attention = model(*args, return_attention=True)
            #Check if loss function __init__ has weight attribute
            if hasattr(loss_function, 'weight'):
                return loss_function(preds, kwargs['yb'], attention_weights=attention, weight=weight)
            else:
                return loss_function(preds, kwargs['yb'], attention_weights=attention)
        else:
            preds = model(*args)
            return loss_function(preds, kwargs['yb'])

    # Set the loss function based on whether attention is required
    if require_attention and not model_supports_attention:
        logging.warning(f"Model does not support attention. Falling back to default loss function.")
        loss_func = nn.CrossEntropyLoss(weight=weight) if problem_type == "classification" else default_loss
    else:
        loss_func = custom_forward if require_attention else loss_function

    # Select metrics
    if 'custom_metrics' in pb_config['experiment']:
        metrics = [retrieve_custom_metric(x) for x in pb_config['experiment']['custom_metrics']]
    else:
        if problem_type == "classification":
            metrics = [RocAuc()]
        elif problem_type == "regression":
            metrics = [mae]
        elif problem_type == "survival":
            metrics = [ConcordanceIndex()]
        else:
            metrics = []

    # Create DataLoaders and Learner
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=metrics, path=outdir)

    # Add MIL augmentations if specified
    if 'augmentations' in pb_config['benchmark_parameters']:
        augmentation = dl_kwargs.get("augmentation", None)
        augmentation_callback = MILAugmentationCallback(augmentation)
        learner.add_cb(augmentation_callback)
        logging.info(f"Training augmentations: {pb_config['benchmark_parameters']['augmentations']}")

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
