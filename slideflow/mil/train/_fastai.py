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
from .._params import TrainerConfigFastAI, ModelConfigCLAM
import logging
from functools import partial

from lifelines.utils import concordance_index

#import custom losses and metrics
from pathbench import losses
from pathbench import metrics
from pathbench import augmentations

# -----------------------------------------------------------------------------

"""
This callback is used to apply augmentations to bags in a batch during model training.

Arguments:
    augmentations (list): List of augmentation names to apply to bags in a batch.

Methods:
    before_batch: Apply augmentations to bags in the batch.
    
"""
class MILAugmentationCallback(Callback):
    def __init__(self, augmentations: list = None):
        self.augmentations = [retrieve_augmentation(x) for x in augmentations]

    def before_batch(self):
        # Retrieve the batch of bags
        batch = self.learn.xb[0]
        
        # Apply augmentations to each bag in the batch
        augmented_batch = []
        for bag in batch:
            for aug in self.augmentations:
                bag = aug(bag)
            augmented_batch.append(bag)
        
        # Convert the list of augmented bags back to a tensor
        augmented_batch = torch.stack(augmented_batch)
        
        # Update the features in the batch
        self.learn.xb = (augmented_batch,) + self.learn.xb[1:]


"""
This function retrieves the optimizer class based on the optimizer name.
Optimizer can be any torch optimizer class.
"""
def retrieve_optimizer(optimizer_name):
    optimizer_class = getattr(optim, optimizer_name)
    return optimizer_class


"""
This function retrieves the augmentation class based on the augmentation name.
Augmentation can be any augmentation class as defined in pathbench/utils/augmentations.py
"""
def retrieve_augmentation(augmentation_name):
    augmentation_class = getattr(augmentations, augmentation_name)
    return augmentation_class()  # Instantiate the augmentation class

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
        if 'lr' in pb_config['experiment'] or 'wd' in pb_config['experiment'] or 'epochs' in pb_config['experiment']:
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
            if 'wd' in pb_config['experiment']:
                logging.info(f"Overriding weight decay to {pb_config['experiment']['wd']}")
                wd = float(pb_config['experiment']['wd'])
            else:
                wd = config.wd
            if 'epochs' in pb_config['experiment']:
                logging.info(f"Overriding epochs to {pb_config['experiment']['epochs']}")
                epochs = pb_config['experiment']['epochs']
            else:
                epochs = config.epochs
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

    if problem_type == 'survival_discrete':
        #One hot encode with regard to the time bins
        time_bins = targets[:, 0].astype(int)
        encoder = OneHotEncoder(sparse_output=False).fit(time_bins.reshape(-1, 1))
        #One hot encode with regard to the time bins
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
            num_workers=8,
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

    if problem_type == 'survival_discrete':
        n_out = unique_categories.shape[0]


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

    if 'ReduceLRonPlateau' in pb_config['experiment']:
        if pb_config['experiment']['ReduceLRonPlateau']:
            logging.info("Using ReduceLROnPlateau callback")
            from fastai.callback.tracker import ReduceLROnPlateau
            cbs = [ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=True)]
            learner.add_cbs(cbs)

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

    # Prepare device.
    device = torch.device(device if device else 'cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f"Problem type: {problem_type}")

    # Handle encoding for classification
    if problem_type == "classification":
        encoder = OneHotEncoder(sparse_output=False).fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None  # No encoder needed for regression or continous survival


    if problem_type == 'survival_discrete':
        #One hot encode with regard to the time bins
        time_bins = targets[:, 0].astype(int)
        #Print the shape of the time bins
        logging.info(f"Time bins shape: {time_bins.shape}")
        #Print the unique time bins
        time_bin_centers = np.unique(time_bins)
        logging.info(f"Unique time bins: {time_bin_centers}")

        encoder = OneHotEncoder(sparse_output=False).fit(time_bins.reshape(-1, 1))
        #One hot encode with regard to the time bins
        targets[:, 0] = targets[:, 0].astype(int)  # Convert durations to integers
        targets[:, 1] = targets[:, 1].astype(int)  # Convert events to integers
        logging.info("Encoder fitted for time bins")

    if problem_type == "survival" or problem_type == "regression":
        # Ensure targets are in float for both survival and regression tasks
        targets = np.array(targets, dtype=np.float32)
        
        if problem_type == "survival":
            # Ensure durations are float and events are integers (for event indicators)
            durations = targets[:, 0].astype(np.float32)  # Ensure durations are float32
            events = targets[:, 1].astype(np.float32)  # Convert events to float32 (1.0 or 0.0)
            
            # Convert to tensors
            durations = torch.tensor(durations, dtype=torch.float32)
            events = torch.tensor(events, dtype=torch.float32)

            if pb_config['experiment']['class_weighting'] == True:
            
                # Calculate weights for survival tasks
                num_events = torch.sum(events)
                num_censored = targets.shape[0] - num_events
                event_weight = num_censored / (num_events + num_censored)
                censored_weight = num_events / (num_events + num_censored)
                
                logging.info(f"Event weight: {event_weight}, Censored weight: {censored_weight}")
                
                # Pass the weights to the loss function if it's a survival task
                loss_function = partial(loss_function, event_weight=event_weight, censored_weight=censored_weight)
            else:
                loss_function = partial(loss_function, event_weight=1.0, censored_weight=1.0)
        targets = torch.tensor(targets, dtype=torch.float32)

    if problem_type == "survival_discrete":
        survival_discrete = True
    else:
        survival_discrete = False
    # Build datasets and dataloaders
    train_dataset = data_utils.build_dataset(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        bag_size=config.bag_size,
        use_lens=config.model_config.use_lens,
        survival_discrete=survival_discrete
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
        use_lens=config.model_config.use_lens,
        survival_discrete=survival_discrete
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

    # Prepare model
    batch = next(iter(train_dl))
    n_in, n_out = batch[0].shape[-1], batch[-1].shape[-1]
    if problem_type == "survival" or problem_type == "regression":
        n_out = 1
    elif problem_type == 'survival_discrete':
        n_out = time_bin_centers.shape[0]

    activation_function = dl_kwargs.get("activation_function", None)

    if 'z_dim' in pb_config['experiment']:
        z_dim = pb_config['experiment']['z_dim']
    else:
        z_dim = 256 # default latent dimension

    if 'encoder_layers' in pb_config['experiment']:
        encoder_layers = pb_config['experiment']['encoder_layers']
    else:
        encoder_layers = 1 # default number of encoder layers

    if 'dropout_p' in pb_config['experiment']:
        dropout_p = pb_config['experiment']['dropout_p']
    else:
        dropout_p = 0.1 # default

    logging.info(f"Training model {config.model_fn.__name__} (in={n_in}, out={n_out}, z_dim={z_dim}, encoder_layers={encoder_layers}, dropout_p={dropout_p})")
 
    model = config.build_model(n_in, n_out, z_dim=z_dim, encoder_layers=encoder_layers, dropout_p=dropout_p, goal=problem_type).to(device)

    # Handle class weights for classification
    weight=None
    if problem_type == "classification" and pb_config['experiment']['class_weighting'] == True:
        counts = pd.value_counts(targets[train_idx])
        weight = counts.sum() / counts
        weight /= weight.sum()
        weight = torch.tensor(
            list(map(weight.get, encoder.categories_[0])), dtype=torch.float32
        ).to(device)
        if loss_function is None:
            loss_function = nn.CrossEntropyLoss(weight=weight)

    elif problem_type == "classification" and pb_config['experiment']['class_weighting'] == False:
        loss_function = nn.CrossEntropyLoss()

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
            elif hasattr(loss_function, 'event_weight') and hasattr(loss_function, 'censored_weight'):
                return loss_function(preds, kwargs['yb'], attention_weights=attention, event_weight=loss_function.event_weight, censored_weight=loss_function.censored_weight)
            else:
                return loss_function(preds, kwargs['yb'], attention_weights=attention)
        else:
            preds = model(*args)
            return loss_function(preds, kwargs['yb'])

    # Set the loss function based on whether attention is required
    if require_attention and not model_supports_attention:
        logging.warning(f"Model does not support attention. Falling back to default loss function.")
        if weight is not None:
            loss_func = nn.CrossEntropyLoss(weight=weight) if problem_type == "classification" else default_loss
        else:
            loss_func = default_loss
    else:
        if 'loss' in pb_config['experiment']:
            loss_func = custom_forward if require_attention else loss_function
        else:
            if weight is not None:
                loss_func = nn.CrossEntropyLoss(weight=weight) if problem_type == "classification" else default_loss
            else:
                loss_func = default_loss

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
        elif problem_type == "survival_discrete":
            metrics = [ConcordanceIndex()]
        else:
            metrics = []


    # Create DataLoaders
    dls = DataLoaders(train_dl, val_dl)
    #Set optimzer and construct learner
    optimizer = dl_kwargs.get("optimizer", None)
    if optimizer is not None:
        optimizer = retrieve_optimizer(optimizer)   
        learner = Learner(dls, model, loss_func=loss_func, metrics=metrics, path=outdir, opt_func=optimizer)
    else:
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
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    unique_categories: np.ndarray,
    n_magnifications: int,
    *,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for a multimodal MIL model."""
    
    # Prepare device.
    device = torch.device(device if device else 'cuda' if torch.cuda.is_available() else 'cpu')
    
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
    elif problem_type == 'survival_discrete':
        default_loss = retrieve_custom_loss("NLLLogisticHazardLoss")
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
    
    loss_function = dl_kwargs.get("loss", None)
    
    if loss_function is not None:
        loss_function = retrieve_custom_loss(loss_function)
    
    # Handle encoding for classification
    if problem_type == "classification":
        encoder = OneHotEncoder(sparse_output=False).fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None  # No encoder needed for regression or survival

    # Prepare targets for survival or regression
    if problem_type == "survival" or problem_type == 'regression' or problem_type == 'survival_discrete':
        targets = np.array(targets, dtype=float)
        if problem_type == "survival" or problem_type == 'survival_discrete':
            targets[:, 0] = targets[:, 0].astype(int)  # Convert durations to integers
            targets[:, 1] = targets[:, 1].astype(int)  # Convert events to integers
        targets = torch.tensor(targets, dtype=torch.float32)

    # Build datasets and dataloaders
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
        num_workers=num_workers,
        persistent_workers=True,
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
        num_workers=num_workers,
        persistent_workers=True,
        device=device,
        after_item=PadToMinLength(),
        **dl_kwargs
    )

    # Prepare model
    batch = next(iter(train_dl))
    if config.model_config.use_lens:
        n_in = [b[0].shape[-1] for b in batch[:-1]]
    else:
        n_in = [b.shape[-1] for b in batch[:-1][0]]
    n_out = batch[-1].shape[-1]
    if problem_type in ["survival", "regression"]:
        n_out = 1

    activation_function = dl_kwargs.get("activation_function", None)

    z_dim = pb_config['experiment'].get('z_dim', 256)  # default latent dimension
    encoder_layers = pb_config['experiment'].get('encoder_layers', 1)  # default number of encoder layers
    dropout_p = pb_config['experiment'].get('dropout_p', 0.1)  # default dropout

    logging.info(f"Training model {config.model_fn.__name__} (in={n_in}, out={n_out}, z_dim={z_dim}, encoder_layers={encoder_layers}, dropout_p={dropout_p})")
    model = config.build_model(n_in, n_out, z_dim=z_dim, encoder_layers=encoder_layers, dropout_p=dropout_p).to(device)

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
    else:
        weight = None

    # Determine if attention values are required by the loss function
    require_attention = getattr(loss_function, 'require_attention', False)

    # Check if the model supports returning attention
    model_supports_attention = 'return_attention' in model.forward.__code__.co_varnames

    # Define custom forward function if attention is required
    def custom_forward(*args, **kwargs):
        if model_supports_attention and require_attention:
            preds, attention = model(*args, return_attention=True)
            # Check if loss function __init__ has weight attribute
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
        if weight is not None:
            loss_func = nn.CrossEntropyLoss(weight=weight) if problem_type == "classification" else default_loss
        else:
            loss_func = default_loss
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
        elif problem_type == "survival" or problem_type == 'survival_discrete':
            metrics = [ConcordanceIndex()]
        else:
            metrics = []

    # Create DataLoaders
    dls = DataLoaders(train_dl, val_dl)

    # Set optimizer and construct learner
    optimizer = dl_kwargs.get("optimizer", None)
    if optimizer is not None:
        optimizer = retrieve_optimizer(optimizer)   
        learner = Learner(dls, model, loss_func=loss_func, metrics=metrics, path=outdir, opt_func=optimizer)
    else:
        learner = Learner(dls, model, loss_func=loss_func, metrics=metrics, path=outdir)

    # Add MIL augmentations if specified
    if 'augmentations' in pb_config['benchmark_parameters']:
        augmentation = dl_kwargs.get("augmentation", None)
        augmentation_callback = MILAugmentationCallback(augmentation)
        learner.add_cb(augmentation_callback)
        logging.info(f"Training augmentations: {pb_config['benchmark_parameters']['augmentations']}")

    return learner, (n_in, n_out)
