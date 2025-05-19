import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
import slideflow as sf
from typing import List, Optional, Union, Tuple
from torch import nn
from torch import Tensor
import fastai.optimizer as optim
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
from fastai.vision.all import (
    DataLoader, DataLoaders, Learner, RocAuc, SaveModelCallback, CSVLogger, FetchPredsCallback, Callback
)
from fastai.callback.schedule import ParamScheduler
from fastai.learner import Metric
from fastai.torch_core import to_detach, flatten_check
from fastai.metrics import mae
import fastai.metrics as fastai_metrics
from slideflow import log
import slideflow.mil.data as data_utils
from slideflow.model import torch_utils
from .._params import TrainerConfigFastAI
import logging
from functools import partial

from lifelines.utils import concordance_index

#import custom losses and metrics
from pathbench import losses
from pathbench import metrics

# -----------------------------------------------------------------------------


class ReduceLROnPlateau(Callback):
    "A simple ReduceLROnPlateau callback that adjusts LR when a monitored metric plateaus."
    order = 60  # Ensure this runs after most other callbacks
    def __init__(self, monitor='valid_loss', factor=0.1, patience=2, min_lr=1e-7, verbose=False):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

    def before_fit(self):
        self.best = float('inf')
        self.num_bad_epochs = 0

    def after_epoch(self):
        # Assume that valid_loss is the first metric in recorder.values.
        # If you have a different ordering, adjust the index accordingly.
        if not self.recorder.values: return
        current = self.recorder.values[-1][0]
        if current < self.best:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for i, pg in enumerate(self.opt.param_groups):
                new_lr = max(pg['lr'] * self.factor, self.min_lr)
                if self.verbose:
                    print(f"Reducing lr for group {i} from {pg['lr']:.2e} to {new_lr:.2e}")
                pg['lr'] = new_lr
            self.num_bad_epochs = 0  # Reset the counter after a reduction



"""
This function retrieves the optimizer class based on the optimizer name.
Optimizer can be any torch optimizer class.
"""
def retrieve_optimizer(optimizer_name):
    optimizer_class = getattr(optim, optimizer_name)
    return optimizer_class


"""
This function retrieves the custom loss class based on the loss name.
Loss can be any loss class as defined in pathbench/utils/losses.py
"""
def retrieve_custom_loss(loss_name):
    logging.info(f"Retrieving custom loss: {loss_name}")
    loss_class = getattr(losses, loss_name)
    return loss_class()  # Instantiate the loss class

"""
This function retrieves the custom metric class based on the metric name.
Metric can be any metric class as defined in pathbench/utils/metrics.py or fastai.metrics
"""
def retrieve_custom_metric(metric_name):
    #Check if metric is in fastai.metrics
    if hasattr(fastai_metrics, metric_name):
        metric_class = getattr(fastai_metrics, metric_name)
    else:
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

    #Check for override of learning parameters
    if pb_config is not None:
        #Overwrite learning rate if specified
        if 'lr' in pb_config['experiment']:
            logging.info(f"Overriding learning rate to {pb_config['experiment']['lr']}")
            lr = float(pb_config['experiment']['lr'])
        else:
            if config.lr is None:
                try:
                    lr = learner.lr_find().valley
                    log.info(f"Using auto-detected learning rate: {lr}")
                except:
                    lr = 1e-3
                    log.info(f"Failed to find learning rate, using default: {lr}")
            else:
                lr = config.lr

        #Overwrite weight decay if specified
        if 'wd' in pb_config['experiment']:
            logging.info(f"Overriding weight decay to {pb_config['experiment']['wd']}")
            wd = float(pb_config['experiment']['wd'])
        else:
            wd = config.wd
        
        #Overwrite epochs if specified
        if 'epochs' in pb_config['experiment']:
            logging.info(f"Overriding epochs to {pb_config['experiment']['epochs']}")
            epochs = pb_config['experiment']['epochs']
        else:
            epochs = config.epochs

        #Add learning rate scheduler if specified
        if 'scheduler' in pb_config['experiment']:
            scheduler = pb_config['experiment']['scheduler']
            if scheduler == 'ReduceLROnPlateau':
                cbs.append(ReduceLROnPlateau())

        learner.fit(n_epoch=epochs, lr=lr, wd=wd, cbs=cbs)
        return learner

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
    return _build_fastai_learner(config, *args, **kwargs)

    
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

    logging.debug(f"dl_kwargs: {dl_kwargs}")

    pb_config = dl_kwargs.get("pb_config", None)
    if pb_config is not None:
        num_workers = pb_config['experiment']['num_workers']

    config_dict = config.to_dict() # Convert to dictionary
    logging.info(f"Building FastAI learner with config: {config}")

    # Extract parameters from config
    encoder_layers = config_dict['encoder_layers']
    dropout_p = config_dict['dropout_p']
    z_dim = config_dict['z_dim']
    activation_function = config_dict['activation_function']
    problem_type = goal = config_dict['task']
    slide_level = config_dict['slide_level']

    #Determine whether slide-level or bag-level training is required
    if slide_level:
        logging.info("Building slide-level FastAI learner....")
    else:
        logging.info("Building bag-level FastAI learner....")

    # Select the appropriate loss function based on the problem type
    if problem_type == "classification":
        default_loss = nn.CrossEntropyLoss()
    elif problem_type == "regression":
        default_loss = nn.MSELoss()
    elif problem_type == "survival":
        default_loss = retrieve_custom_loss("CoxPHLoss")
    elif problem_type == 'survival_discrete':
        default_loss = retrieve_custom_loss("NLLLogisticHazardLoss")
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
    
    loss_function = dl_kwargs.get("loss", None)
    if loss_function is not None:
        loss_function = retrieve_custom_loss(loss_function)
    else:
        loss_function = default_loss

    if 'class_weighting' in pb_config['experiment']:
        class_weighting = pb_config['experiment']['class_weighting']
    else:
        class_weighting = False

    # Prepare device.
    device = torch.device(device if device else 'cuda' if torch.cuda.is_available() else 'cpu')

    logging.debug(f"Problem type: {problem_type}")
    # === TARGETS & ENCODER PREPARATION ===
    if slide_level:
        # Slide-level handling:
        if problem_type == "classification":
            encoder = OneHotEncoder(sparse_output=False).fit(unique_categories.reshape(-1, 1))
        elif problem_type in ["survival", "survival_discrete"]:
            # Convert targets to float and make sure durations/events are ints.
            targets = np.array(targets, dtype=float)
            targets[:, 0] = targets[:, 0].astype(int)
            targets[:, 1] = targets[:, 1].astype(int)
            if problem_type == "survival_discrete":
                # Use time bins to define the output dimension.
                encoder = OneHotEncoder(sparse_output=False).fit(targets[:, 0].reshape(-1, 1))
            else:
                encoder = None
        else:  # regression
            targets = np.array(targets, dtype=np.float32)
            encoder = None
    else:
        # Bag-level handling.
        if problem_type == "classification":
            encoder = OneHotEncoder(sparse_output=False).fit(unique_categories.reshape(-1, 1))
        else:
            encoder = None

        if problem_type == 'survival_discrete':
            time_bins = targets[:, 0].astype(int)
            loggin.debug(f"Time bins shape: {time_bins.shape}")
            time_bin_centers = np.unique(time_bins)
            logging.debug(f"Unique time bins: {time_bin_centers}")
            encoder = OneHotEncoder(sparse_output=False).fit(time_bins.reshape(-1, 1))
            targets[:, 0] = targets[:, 0].astype(int)
            targets[:, 1] = targets[:, 1].astype(int)
            logging.debug("Encoder fitted for time bins")

        if problem_type in ["survival", "regression"]:
            targets = np.array(targets, dtype=np.float32)
            if problem_type == "survival":
                durations = targets[:, 0].astype(np.float32)
                events = targets[:, 1].astype(np.float32)
                durations = torch.tensor(durations, dtype=torch.float32)
                events = torch.tensor(events, dtype=torch.float32)
                if class_weighting:
                    num_events = torch.sum(events)
                    num_censored = targets.shape[0] - num_events
                    event_weight = num_censored / (num_events + num_censored)
                    censored_weight = num_events / (num_events + num_censored)
                    logging.debug(f"Event weight: {event_weight}, Censored weight: {censored_weight}")
                    loss_function = partial(loss_function, event_weight=event_weight, censored_weight=censored_weight)
                else:
                    loss_function = partial(loss_function, event_weight=1.0, censored_weight=1.0)
            targets = torch.tensor(targets, dtype=torch.float32)

    # === DATASET & DATALOADER CREATION ===
    if slide_level:
        logging.info("Building slide-level datasets....")
        #Log encoder and targets
        logging.debug(f"Encoder categories: {encoder.categories_}")
        logging.debug(f"Targets shape: {targets.shape}")
        train_dataset = data_utils.build_slide_dataset(
            [bags[i] for i in train_idx],
            targets[train_idx],
            survival_discrete=(problem_type == "survival_discrete"),
            encoder=encoder,
            bag_size=1
        )
        val_dataset = data_utils.build_slide_dataset(
            [bags[i] for i in val_idx],
            targets[val_idx],
            survival_discrete=(problem_type == "survival_discrete"),
            encoder=encoder,
            bag_size=1
        )

        #Log one sample from the dataset
        logging.debug(f"Sample from slide-level train dataset: {train_dataset[0]}")
        # Dataloaders for slide-level (fixed-length feature vectors)
        train_dl = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=dl_kwargs.get("num_workers", num_workers),
            device=device,
            drop_last=True,
            persistent_workers=True,
            **dl_kwargs
        )
        val_dl = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=dl_kwargs.get("num_workers", num_workers),
            persistent_workers=True,
            device=device,
            **dl_kwargs
        )
        #Log one sample from the dataloader
        logging.debug(f"Sample from slide-level train dataloader: {next(iter(train_dl))}")
    else:
        # Bag-level datasets (each bag may have variable length and requires padding)
        train_dataset = data_utils.build_dataset(
            bags[train_idx],
            targets[train_idx],
            encoder=encoder,
            bag_size=config.bag_size,
            use_lens=config.model_config.use_lens,
            survival_discrete=(problem_type == "survival_discrete")
        )
        val_dataset = data_utils.build_dataset(
            bags[val_idx],
            targets[val_idx],
            encoder=encoder,
            bag_size=None,
            use_lens=config.model_config.use_lens,
            survival_discrete=(problem_type == "survival_discrete")
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
        val_dl = DataLoader(
            val_dataset,
            batch_size=1 if problem_type == "classification" else config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
            device=device,
            after_item=PadToMinLength(),
            **dl_kwargs
        )

    # === DETERMINE INPUT/OUTPUT DIMENSIONS ===
    sample = next(iter(train_dl))
    n_in = sample[0].shape[-1]
    if slide_level:
        if problem_type == "classification":
            n_out = unique_categories.shape[0]
        elif problem_type in ["regression", "survival"]:
            n_out = 1
        elif problem_type == "survival_discrete":
            n_out = len(np.unique(targets[:, 0]))
        else:
            n_out = 1
    else:
        n_out = sample[-1].shape[-1]
        if problem_type in ["survival", "regression"]:
            n_out = 1
        elif problem_type == 'survival_discrete':
            n_out = np.unique(targets[:, 0]).shape[0]

    
    logging.info(f"Training model {config.model_fn.__name__} (in={n_in}, out={n_out}, "
                    f"z_dim={z_dim}, encoder_layers={encoder_layers}, dropout_p={dropout_p})")
    model = config.build_model(n_in, n_out, z_dim=z_dim,
                                encoder_layers=encoder_layers,
                                dropout_p=dropout_p,
                                activation_function=activation_function,
                                goal=problem_type).to(device)

    # === CLASS WEIGHTING FOR CLASSIFICATION ===
    weight = None
    if problem_type == "classification" and class_weighting:
        counts = pd.value_counts(targets[train_idx])
        weight = counts.sum() / counts
        weight /= weight.sum()
        weight = torch.tensor(
            list(map(weight.get, encoder.categories_[0])), dtype=torch.float32
        ).to(device)
        if loss_function is None:
            loss_function = nn.CrossEntropyLoss(weight=weight)
    elif problem_type == "classification" and not class_weighting:
        loss_function = nn.CrossEntropyLoss()

    # === ATTENTION & CUSTOM FORWARD HANDLING ===
    require_attention = getattr(loss_function, 'require_attention', False)
    model_supports_attention = 'return_attention' in model.forward.__code__.co_varnames

    def custom_forward(*args, **kwargs):
        if model_supports_attention and require_attention:
            preds, attention = model(*args, return_attention=True)
            if hasattr(loss_function, 'weight'):
                return loss_function(preds, kwargs['yb'], attention_weights=attention, weight=weight)
            elif hasattr(loss_function, 'event_weight') and hasattr(loss_function, 'censored_weight'):
                return loss_function(preds, kwargs['yb'], attention_weights=attention,
                                     event_weight=loss_function.event_weight,
                                     censored_weight=loss_function.censored_weight)
            else:
                return loss_function(preds, kwargs['yb'], attention_weights=attention)
        else:
            preds = model(*args)
            return loss_function(preds, kwargs['yb'])

    if require_attention and not model_supports_attention:
        logging.warning("Model does not support attention. Falling back to default loss function.")
        loss_func = nn.CrossEntropyLoss(weight=weight) if (problem_type == "classification" and weight is not None) else default_loss
    else:
        if 'loss' in pb_config['experiment']:
            loss_func = custom_forward if require_attention else loss_function
        else:
            loss_func = nn.CrossEntropyLoss(weight=weight) if (problem_type == "classification" and weight is not None) else default_loss

    # === METRICS ===
    if 'custom_metrics' in pb_config['experiment']:
        metrics = [retrieve_custom_metric(x) for x in pb_config['experiment']['custom_metrics']]
    else:
        if problem_type == "classification":
            metrics = [RocAuc()]
        elif problem_type == "regression":
            metrics = [mae]
        elif problem_type in ["survival", "survival_discrete"]:
            metrics = [ConcordanceIndex()]
        else:
            metrics = []

    logging.debug(f"Targets shape: {targets.shape}")
    if targets.ndim > 1 and targets.shape[1] == 1:
        targets = targets.flatten()

    # === CREATE LEARNER ===
    dls = DataLoaders(train_dl, val_dl)
    optimizer = dl_kwargs.get("optimizer", None)
    if optimizer is not None:
        optimizer = retrieve_optimizer(optimizer)
        learner = Learner(dls, model, loss_func=loss_func, metrics=metrics, path=outdir, opt_func=optimizer)
    else:
        learner = Learner(dls, model, loss_func=loss_func, metrics=metrics, path=outdir)

    return learner, (n_in, n_out)