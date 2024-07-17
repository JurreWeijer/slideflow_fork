"""
Ruifrok normalization based on the method of:

A. C. Ruifrok, D. A. Johnston, "Quantification of histochemical staining by color deconvolution", Analytical and Quantitative Cytology and Histology, vol. 23, no. 4, pp. 291-299, Aug. 2001.
"""

from typing import Tuple, Dict, Optional, Union
import torch
import numpy as np
from contextlib import contextmanager
import slideflow.norm.utils as ut
from .utils import clip_size, standardize_brightness

# -----------------------------------------------------------------------------

class RuifrokExtractor:
    """Ruifrok stain extractor.

    Get the stain matrix as defined in:

    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.
    """

    def __init__(self) -> None:
        """Initialize RuifrokExtractor."""
        self.__stain_matrix = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]])

    def get_stain_matrix(self) -> np.ndarray:
        """Get the pre-defined stain matrix.

        Returns:
            numpy.ndarray: Pre-defined stain matrix.
        """
        return self.__stain_matrix.copy()

# -----------------------------------------------------------------------------

def rgb_to_od(I: torch.Tensor) -> torch.Tensor:
    """Convert from RGB uint8 to optical density (OD).

    Args:
        I (torch.Tensor): RGB uint8 image.

    Returns:
        torch.Tensor: Optical density image.
    """
    I = I.to(torch.float32)
    I[I == 0] = 1  # To avoid log(0)
    return -torch.log(I / 255 + 1e-6)

def od_to_rgb(OD: torch.Tensor) -> torch.Tensor:
    """Convert from optical density (OD) to RGB uint8.

    Args:
        OD (torch.Tensor): Optical density image.

    Returns:
        torch.Tensor: RGB uint8 image.
    """
    return (255 * torch.exp(-OD)).to(torch.uint8)

def color_deconvolution(I: torch.Tensor, stain_matrix: torch.Tensor) -> torch.Tensor:
    """Perform color deconvolution to get stain concentrations.

    Args:
        I (torch.Tensor): Optical density image.
        stain_matrix (torch.Tensor): Stain matrix.

    Returns:
        torch.Tensor: Stain concentrations.
    """
    I = I.reshape(-1, 3).T
    stains = torch.linalg.solve(stain_matrix, I).T
    return stains.reshape((-1, I.shape[1], 3))

def color_convolution(stains: torch.Tensor, stain_matrix: torch.Tensor) -> torch.Tensor:
    """Convert stain concentrations back to optical density.

    Args:
        stains (torch.Tensor): Stain concentrations.
        stain_matrix (torch.Tensor): Stain matrix.

    Returns:
        torch.Tensor: Optical density image.
    """
    stains = stains.reshape(-1, 3).T
    OD = torch.matmul(stain_matrix, stains).T
    return OD.reshape((-1, stains.shape[1], 3))

# -----------------------------------------------------------------------------

def get_masked_mean_std(I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get mean and standard deviation of each channel, with white pixels masked.

    Args:
        I (torch.Tensor): RGB uint8 image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Channel means and standard deviations.
    """
    mask = torch.all(I == 255, dim=3)
    I = I[~mask]

    I = rgb_to_od(I)
    mean = torch.mean(I, dim=0)
    std = torch.std(I, dim=0)
    return mean, std

def get_mean_std(I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get mean and standard deviation of each channel.

    Args:
        I (torch.Tensor): RGB uint8 image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Channel means and standard deviations.
    """
    I = rgb_to_od(I)
    mean = torch.mean(I, dim=0)
    std = torch.std(I, dim=0)
    return mean, std

# -----------------------------------------------------------------------------

def transform(
    I: torch.Tensor,
    tgt_mean: torch.Tensor,
    tgt_std: torch.Tensor,
    stain_matrix: torch.Tensor,
    *,
    ctx_mean: Optional[torch.Tensor] = None,
    ctx_std: Optional[torch.Tensor] = None,
    mask_threshold: Optional[float] = None
) -> torch.Tensor:
    """Normalize an H&E image using Ruifrok method.

    Args:
        I (torch.Tensor): RGB uint8 image.
        tgt_mean (torch.Tensor): Target channel means.
        tgt_std (torch.Tensor): Target channel standard deviations.
        stain_matrix (torch.Tensor): Stain matrix.
        ctx_mean (torch.Tensor, optional): Context channel means.
        ctx_std (torch.Tensor, optional): Context channel standard deviations.
        mask_threshold (float, optional): Mask threshold for white pixels.

    Returns:
        torch.Tensor: Normalized image.
    """
    if ctx_mean is None and ctx_std is not None:
        raise ValueError("If 'ctx_std' is provided, 'ctx_mean' must not be None")
    if ctx_std is None and ctx_mean is not None:
        raise ValueError("If 'ctx_mean' is provided, 'ctx_std' must not be None")
    tgt_mean = tgt_mean.to(I.device)
    tgt_std = tgt_std.to(I.device)

    if mask_threshold:
        mask = torch.unsqueeze(torch.all(I >= mask_threshold, dim=-1), -1)

    I = rgb_to_od(I)
    stains = color_deconvolution(I, stain_matrix)
    if ctx_mean is not None and ctx_std is not None:
        stains_mean, stains_std = ctx_mean, ctx_std
    else:
        stains_mean, stains_std = get_mean_std(stains)

    norm_stains = (stains - stains_mean) * (tgt_std / stains_std) + tgt_mean
    norm_OD = color_convolution(norm_stains, stain_matrix)
    norm_I = od_to_rgb(norm_OD)

    if mask_threshold:
        return torch.where(mask, I, norm_I)
    else:
        return norm_I

# -----------------------------------------------------------------------------

class RuifrokNormalizer:

    vectorized = True
    preferred_device = 'cuda'
    preset_tag = 'ruifrok'

    def __init__(self) -> None:
        """Ruifrok H&E stain normalizer (PyTorch implementation).

        Normalizes an image as defined by:

        A. C. Ruifrok, D. A. Johnston, "Quantification of histochemical staining by color deconvolution", Analytical and Quantitative Cytology and Histology, vol. 23, no. 4, pp. 291-299, Aug. 2001.
        """
        self.threshold = None  # type: Optional[float]
        self._ctx_means = None  # type: Optional[torch.Tensor]
        self._ctx_stds = None  # type: Optional[torch.Tensor]
        self._augment_params = dict()  # type: Dict[str, torch.Tensor]
        self.stain_matrix = torch.tensor(RuifrokExtractor().get_stain_matrix(), dtype=torch.float32)
        self.set_fit(**ut.fit_presets[self.preset_tag]['v3'])  # type: ignore
        self.set_augment(**ut.augment_presets[self.preset_tag]['v2'])  # type: ignore

    def fit(
        self,
        target: torch.Tensor,
        reduce: bool = False,
        mask: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit normalizer to a target image.

        Args:
            target (torch.Tensor): RGB uint8 image.
            reduce (bool, optional): Reduce fit parameters across a batch of images.
            mask (bool, optional): Ignore white pixels when fitting.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Channel means and standard deviations.
        """
        if len(target.shape) == 3:
            target = torch.unsqueeze(target, dim=0)
        target = clip_size(target, 2048)
        target = rgb_to_od(target)
        target = color_deconvolution(target, self.stain_matrix)
        means, stds = get_mean_std(target)
        self.target_means = means
        self.target_stds = stds
        return means, stds

    def augment_preset(self, preset: str) -> Dict[str, np.ndarray]:
        """Configure normalizer augmentation using a preset.

        Args:
            preset (str): Preset.

        Returns:
            Dict[str, np.ndarray]: Augmentation values.
        """
        _aug = ut.augment_presets[self.preset_tag][preset]
        self.set_augment(**_aug)
        return _aug

    def fit_preset(self, preset: str) -> Dict[str, np.ndarray]:
        """Fit normalizer to a preset.

        Args:
            preset (str): Preset.

        Returns:
            Dict[str, np.ndarray]: Fitted values.
        """
        _fit = ut.fit_presets[self.preset_tag][preset]
        self.set_fit(**_fit)
        return _fit

    def get_fit(self) -> Dict[str, Optional[np.ndarray]]:
        """Get the current normalizer fit.

        Returns:
            Dict[str, Optional[np.ndarray]]: Current fit values.
        """
        return {
            'target_means': None if self.target_means is None else self.target_means.numpy(),
            'target_stds': None if self.target_stds is None else self.target_stds.numpy()
        }

    def _get_context_means(
        self,
        ctx_means: Optional[torch.Tensor] = None,
        ctx_stds: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._ctx_means is not None and self._ctx_stds is not None:
            return self._ctx_means, self._ctx_stds
        else:
            return ctx_means, ctx_stds

    def set_fit(
        self,
        target_means: Union[np.ndarray, torch.Tensor],
        target_stds: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Set the normalizer fit to the given values.

        Args:
            target_means (Union[np.ndarray, torch.Tensor]): Channel means.
            target_stds (Union[np.ndarray, torch.Tensor]): Channel standard deviations.
        """
        if not isinstance(target_means, torch.Tensor):
            target_means = torch.from_numpy(ut._as_numpy(target_means))
        if not isinstance(target_stds, torch.Tensor):
            target_stds = torch.from_numpy(ut._as_numpy(target_stds))
        self.target_means = target_means
        self.target_stds = target_stds

    def set_augment(
        self,
        means_stdev: Optional[Union[np.ndarray, torch.Tensor]] = None,
        stds_stdev: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """Set the normalizer augmentation to the given values.

        Args:
            means_stdev (Optional[Union[np.ndarray, torch.Tensor]]): Standard deviation of the target means.
            stds_stdev (Optional[Union[np.ndarray, torch.Tensor]]): Standard deviation of the target stds.
        """
        if means_stdev is None and stds_stdev is None:
            raise ValueError("One or both arguments 'means_stdev' and 'stds_stdev' are required.")
        if means_stdev is not None:
            self._augment_params['means_stdev'] = torch.from_numpy(ut._as_numpy(means_stdev))
        if stds_stdev is not None:
            self._augment_params['stds_stdev'] = torch.from_numpy(ut._as_numpy(stds_stdev))

    def transform(
        self,
        I: torch.Tensor,
        ctx_means: Optional[torch.Tensor] = None,
        ctx_stds: Optional[torch.Tensor] = None,
        *,
        augment: bool = False
    ) -> torch.Tensor:
        """Normalize an H&E image.

        Args:
            I (torch.Tensor): RGB uint8 image.
            ctx_means (torch.Tensor, optional): Context channel means.
            ctx_stds (torch.Tensor, optional): Context channel standard deviations.
            augment (bool, optional): Transform using stain augmentation.

        Returns:
            torch.Tensor: Normalized image.
        """
        if augment and not any(m in self._augment_params for m in ('means_stdev', 'stds_stdev')):
            raise ValueError("Augmentation space not configured.")

        _I = torch.unsqueeze(I, dim=0) if len(I.shape) == 3 else I
        _ctx_means, _ctx_stds = self._get_context_means(ctx_means, ctx_stds)
        aug_kw = self._augment_params if augment else {}
        transformed = transform(
            _I,
            self.target_means,
            self.target_stds,
            self.stain_matrix,
            ctx_mean=_ctx_means,
            ctx_std=_ctx_stds,
            mask_threshold=self.threshold,
            **aug_kw
        )
        if len(I.shape) == 3:
            return transformed[0]
        else:
            return transformed

    def augment(self, I: torch.Tensor) -> torch.Tensor:
        """Augment an H&E image.

        Args:
            I (torch.Tensor): RGB uint8 image.

        Returns:
            torch.Tensor: Augmented image.
        """
        if not any(m in self._augment_params for m in ('means_stdev', 'stds_stdev')):
            raise ValueError("Augmentation space not configured.")

        _I = torch.unsqueeze(I, dim=0) if len(I.shape) == 3 else I
        transformed = augment(
            _I,
            self._augment_params['means_stdev'],
            self._augment_params['stds_stdev'],
            mask_threshold=self.threshold
        )
        if len(I.shape) == 3:
            return transformed[0]
        else:
            return transformed

    @contextmanager
    def image_context(self, I: Union[np.ndarray, torch.Tensor]):
        """Set the whole-slide context for the stain normalizer.

        Args:
            I (Union[np.ndarray, torch.Tensor]): Context to use for normalization.
        """
        self.set_context(I)
        yield
        self.clear_context()

    def set_context(self, I: Union[np.ndarray, torch.Tensor]):
        """Set the whole-slide context for the stain normalizer.

        Args:
            I (Union[np.ndarray, torch.Tensor]): Context to use for normalization.
        """
        if not isinstance(I, torch.Tensor):
            I = torch.from_numpy(ut._as_numpy(I))
        if len(I.shape) == 3:
            I = torch.unsqueeze(I, dim=0)
        I = clip_size(I, 2048)
        self._ctx_means, self._ctx_stds = get_masked_mean_std(I)

    def clear_context(self):
        """Remove any previously set stain normalizer context."""
        self._ctx_means, self._ctx_stds = None, None
