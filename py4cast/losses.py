"""
This module contains the loss functions used in the training of the models.
"""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import math
from torch.nn import MSELoss

from py4cast.datasets.base import DatasetInfo, NamedTensor

class AlmostFairCRPS(torch.nn.Module): # Batched Almost Fair CRPS
    """
    AIFS-CRPS: Ensemble forecasting using a model trained with a loss function based on the Continuous Ranked Probability Score
    https://arxiv.org/abs/2412.15832
    """
    def __init__(self, alpha=0.95):
        super().__init__()
        self.alpha = alpha

    def forward(self, preds, targets, noise_members=4):

        B = targets.shape[0] # effective batch size 
        M = noise_members  # Number of ensemble members
        epsilon = (1 - self.alpha) / M

        # Reshape predictions 
        preds = preds.reshape(B, M, *preds.shape[1:])        # (B, M, T, W, H, d_f)
        # Replicate targets along the members dimension
        targets = targets.unsqueeze(1).expand(-1, M, *targets.shape[1:])  # (B, M, T, W, H, d_f)

        abs_diff = torch.abs(preds - targets)             # (B, M, T, W, H, d_f)

        # Pairwise differences: |x_j - x_k|
        preds_j = preds.unsqueeze(2)                      # (B, M, 1, T, W, H, d_f)
        preds_k = preds.unsqueeze(1)                      # (B, 1, M, T, W, H, d_f)
        pairwise_diff = torch.abs(preds_j - preds_k)      # (B, M, M, T, W, H, d_f)

        # |x_j - y| + |x_k - y|
        abs_j = abs_diff.unsqueeze(2)                     # (B, M, 1, T, W, H, d_f)
        abs_k = abs_diff.unsqueeze(1)                     # (B, 1, M, T, W, H, d_f)
        pairwise_abs_sum = abs_j + abs_k                  # (B, M, M, T, W, H, d_f)

        # Create off-diagonal mask (M, M)
        mask = ~torch.eye(M, dtype=torch.bool, device=preds.device)  # (M, M)

        # Apply mask per batch element
        pairwise_diff = pairwise_diff[:, mask].view(B, M, M - 1, *preds.shape[2:])
        pairwise_abs_sum = pairwise_abs_sum[:, mask].view(B, M, M - 1, *preds.shape[2:])

        # Final computation
        afcrps = (pairwise_abs_sum - (1 - epsilon) * pairwise_diff).mean(dim=(1, 2)) / 2  # (B, T, W, H, d_f)
        return afcrps


class Py4CastLoss(ABC):
    """
    Abstract class to force the user to implement the prepare and forward method because
    They are expected by the rest of the system.
    See https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html
    """

    def __init__(self, loss: str, *args, **kwargs) -> None:
        if loss in ["MSELoss", "L1Loss"]:
            self.loss = getattr(torch.nn, loss)(*args, **kwargs)
        elif loss == "AFCRPS":
            self.loss = AlmostFairCRPS()
        else:
            raise ValueError("Unrecognized loss function")

    @abstractmethod
    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ) -> None:
        """
        Prepare the loss function using the dataset informations and the interior mask
        """

    @abstractmethod
    def forward(self, prediction: NamedTensor, target: NamedTensor, epoch=None) -> torch.Tensor:
        """
        Compute the loss function
        """

    def register_loss_state_buffers(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        loss_state_weight: dict,
        squeeze_mask: bool = False,
    ) -> None:
        """
        We register the state_weight buffer to the lightning module
        and keep references to other buffers of interest
        """

        self.loss_state_weight = loss_state_weight
        attr_name = "interior_mask_s" if squeeze_mask else "interior_mask"
        if not hasattr(lm, attr_name):
            lm.register_buffer(
                attr_name,
                interior_mask.squeeze(-1) if squeeze_mask else interior_mask,
                persistent=False,
            )
        self.num_interior = torch.sum(interior_mask).item()

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    @lru_cache(maxsize=8)
    def weights(self, feature_names: Tuple[str], device: torch.device) -> torch.Tensor:
        """
        We build and cache the weights tensor for the given loss, feature names and device.
        """
        return torch.stack([self.loss_state_weight[name] for name in feature_names]).to(
            device, non_blocking=True
        )


class WeightedLoss(Py4CastLoss):
    """
    Compute a weighted loss function with a weight for each feature.
    During the forward step, the loss is computed for each feature and then weighted
    and optionally averaged over the spatial dimensions.
    """

    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ) -> None:
        # build the dictionnary of weight
        loss_state_weight = {}

        for name in dataset_info.state_weights:
            loss_state_weight[name] = torch.tensor(dataset_info.state_weights[name])

        self.register_loss_state_buffers(
            lm, interior_mask, loss_state_weight, squeeze_mask=True
        )
        self.lm = lm

    def forward(
        self,
        prediction: NamedTensor,
        target: NamedTensor,
        epoch=None,
        reduce_spatial_dim=True,
    ) -> torch.Tensor:
        """
        Computed weighted loss function.
        prediction/target: (B, pred_steps, N_grid, d_f) or (B, pred_steps, W, H, d_f)
        returns (B, pred_steps)
        """

        # Compute Torch loss (defined in the parent class when this Mixin is used)
        if self.loss.__class__ == AlmostFairCRPS:
            torch_loss = self.loss(prediction.tensor, target.tensor, noise_members=self.lm.noise_members)
        else:
            torch_loss = self.loss(prediction.tensor, target.tensor)
        # Retrieve the weights for each feature
        weights = self.weights(tuple(prediction.feature_names), prediction.device)

        # Apply the weights and sum over the feature dimension
        weighted_loss = torch.sum(torch_loss * weights, dim=-1)

        # if no reduction on spatial dimension is required, return the weighted loss
        if not reduce_spatial_dim:
            return weighted_loss

        # Compute the mean loss over all spatial dimensions
        # Take (unweighted) mean over only non-border (interior) grid nodes/pixels
        # We use forward indexing for the spatial_dim_idx of the target tensor
        # so the code below works even if the feature dimension has been reduced
        # The final shape is (B, pred_steps)

        time_step_mean_loss = (
            torch.sum(
                weighted_loss * self.lm.interior_mask_s,
                dim=target.spatial_dim_idx,
            )
            / self.num_interior
        )
        return time_step_mean_loss


class ScaledLoss(Py4CastLoss):
    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
    ) -> None:
        # build the dictionnary of weight
        loss_state_weight = {}
        for name in dataset_info.state_weights:
            loss_state_weight[name] = dataset_info.stats[name]["std"]
        self.register_loss_state_buffers(lm, interior_mask, loss_state_weight)
        self.lm = lm

    def forward(self, prediction: NamedTensor, target: NamedTensor, epoch=None) -> torch.Tensor:
        """
        Computed weighted loss function averaged over all spatial dimensions.
        prediction/target: (B, pred_steps, N_grid, d_f) or (B, pred_steps, W, H, d_f)
        returns (B, pred_steps)
        """
        # Compute Torch loss (defined in the parent class when this Mixin is used)
        if self.loss.__class__ == AlmostFairCRPS:
            torch_loss = self.loss(prediction.tensor, target.tensor, noise_members=self.lm.noise_members)
        else:
            torch_loss = self.loss(prediction.tensor, target.tensor)

        # Compute the mean loss value over spatial dimensions
        mean_loss = (
            torch.sum(
                torch_loss * self.lm.interior_mask,
                dim=target.spatial_dim_idx,
            )
            / self.num_interior
        )

        if self.loss.__class__ == MSELoss:
            mean_loss = torch.sqrt(mean_loss)

        return mean_loss * self.weights(
            tuple(prediction.feature_names), prediction.device
        )




class MultiScaleLoss(Py4CastLoss):
    """
    Compute a multi-scale loss function with band decomposition.
    Each prediction and target is decomposed into frequency bands,
    normalized by precomputed multi-scale stds, then reduced.
    Requires multi_scale_stats.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernel_cache = {}  # cache kernels per (n, device, dtype)

    def prepare(
        self,
        lm: pl.LightningModule,
        interior_mask: torch.Tensor,
        dataset_info: DatasetInfo,
        max_scale_power: int = 5,
    ) -> None:
        self.max_scale_power = max_scale_power
        self.scales = [2 ** i for i in reversed(range(max_scale_power + 1))]
        self.n_scales = len(self.scales)

        # Build state weights normalized by scale-dependent stds
        loss_state_weight = {}
        for name in dataset_info.state_weights:
            # shape (n_scales,)
            stds = torch.tensor(dataset_info.multi_scale_stats[name]["std"])
            inv_stds = 1 / stds
            # Normalize to sum to 1
            weights = inv_stds / inv_stds.sum()
            loss_state_weight[name] = dataset_info.state_weights[name] * weights
            # print(f"MultiScaleLoss: feature {name}, stds {stds}")
            # print(f"DatasetInfo: feature {name}, weights {dataset_info.state_weights[name]}")
            # print(f"MultiScaleLoss: feature {name}, weights {loss_state_weight[name]}")

        self.register_loss_state_buffers(
            lm, interior_mask, loss_state_weight, squeeze_mask=True
        )
        self.lm = lm

    # def smooth(self, x: torch.Tensor, scale: int = 1) -> NamedTensor:
    #     # Unpack shape
    #     B, T, N_lat, N_lon, d_f = x.tensor.shape

    #     # Rearrange for interpolation: (B*T, d_f, N_lat, N_lon)
    #     aro_outputs = x.tensor.view(B * T, N_lat, N_lon, d_f).permute(0, 3, 1, 2)

    #     # Compute low-res dimensions
    #     N_lat_low = N_lat // scale
    #     N_lon_low = N_lon // scale

    #     # Downsample
    #     aro_outputs_lowres = torch.nn.functional.interpolate(
    #         aro_outputs, size=(N_lat_low, N_lon_low), mode='bilinear', align_corners=False
    #     )

    #     # Upsample back
    #     aro_outputs_upsampled = torch.nn.functional.interpolate(
    #         aro_outputs_lowres, size=(N_lat, N_lon), mode='bilinear', align_corners=False
    #     )

    #     # Rearrange back to (B, T, N_lat, N_lon, d_f)
    #     aro_outputs_upsampled_tensor = aro_outputs_upsampled.permute(0, 2, 3, 1).view(B, T, N_lat, N_lon, d_f)

    #     # Wrap into NamedTensor
    #     aro_outputs_upsampled_nt = NamedTensor.new_like(
    #         aro_outputs_upsampled_tensor.type_as(x.tensor), x
    #     )
    #     return aro_outputs_upsampled_nt

    # def smooth(self, x: torch.Tensor, n: int = 1) -> NamedTensor:
    #     # Unpack shape
    #     B, T, N_lat, N_lon, d_f = x.tensor.shape
    #     # Rearrange for interpolation: (B*T, d_f, N_lat, N_lon)
    #     aro_inputs = x.tensor.view(B * T, N_lat, N_lon, d_f).permute(0, 3, 1, 2)

    #     # Map scale factor n -> equivalent Gaussian sigma
    #     sigma = n / math.sqrt(3.0)

    #     # Build Gaussian kernel
    #     kernel_size = int(6 * sigma + 1)
    #     if kernel_size % 2 == 0:  # ensure odd
    #         kernel_size += 1

    #     coords = torch.arange(kernel_size, device=aro_inputs.device, dtype=aro_inputs.dtype) - kernel_size // 2
    #     g_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    #     g_1d /= g_1d.sum()


    #     # Separable 2D kernel
    #     g_2d = torch.outer(g_1d, g_1d)
    #     g_2d /= g_2d.sum()

    #     # Depthwise conv kernel
    #     g_kernel = g_2d.view(1, 1, kernel_size, kernel_size).to(aro_inputs.dtype).to(aro_inputs.device)
    #     g_kernel = g_kernel.repeat(d_f, 1, 1, 1)  # one kernel per channel

    #     # Padding for "same" output size
    #     pad = kernel_size // 2
    #     aro_inputs_padded = F.pad(aro_inputs, (pad, pad, pad, pad), mode="reflect")

    #     # Apply depthwise convolution
    #     aro_outputs = F.conv2d(aro_inputs_padded, g_kernel, groups=d_f)

    #     # Rearrange back to (B, T, N_lat, N_lon, d_f)
    #     aro_outputs_tensor = aro_outputs.permute(0, 2, 3, 1).view(B, T, N_lat, N_lon, d_f)

    #     # Wrap into NamedTensor
    #     aro_outputs_nt = NamedTensor.new_like(
    #         aro_outputs_tensor.type_as(x.tensor), x
    #     )
    #     return aro_outputs_nt

    def _get_gaussian_1d(self, n: int, device, dtype) -> torch.Tensor:
        """
        Return cached 1D Gaussian kernel for scale n.
        """
        key = (n, device, dtype)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        sigma = n / math.sqrt(3.0)
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
        g_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g_1d /= g_1d.sum()

        self._kernel_cache[key] = g_1d
        return g_1d

    def smooth(self, x: torch.Tensor, n: int = 1) -> NamedTensor:
        B, T, N_lat, N_lon, d_f = x.tensor.shape
        aro_inputs = x.tensor.view(B * T, N_lat, N_lon, d_f).permute(0, 3, 1, 2)

        g_1d = self._get_gaussian_1d(n, aro_inputs.device, aro_inputs.dtype)

        # Build separable kernels
        g_row = g_1d.view(1, 1, 1, -1).repeat(d_f, 1, 1, 1)
        g_col = g_1d.view(1, 1, -1, 1).repeat(d_f, 1, 1, 1)

        pad = g_1d.numel() // 2
        aro_inputs_padded = F.pad(aro_inputs, (pad, pad, pad, pad), mode="reflect")

        # Apply separable conv
        out = F.conv2d(aro_inputs_padded, g_row, groups=d_f)
        out = F.conv2d(out, g_col, groups=d_f)

        aro_outputs_tensor = out.permute(0, 2, 3, 1).view(B, T, N_lat, N_lon, d_f)
        return NamedTensor.new_like(aro_outputs_tensor.type_as(x.tensor), x)

    def decompose_bands(self, tensor: NamedTensor) -> list[torch.Tensor]:
        """
        Perform smoothing at multiple scales and compute frequency bands.
        Returns list of tensors (one per scale) with same shape as input.
        """
        smooths = [
            self.smooth(tensor, n=s) if s != 1 else tensor
            for s in self.scales
        ]
        bands = []
        for i in range(len(self.scales) - 1):
            band = NamedTensor.new_like(
                (smooths[i + 1].tensor - smooths[i].tensor).type_as(tensor.tensor),
                tensor,
            )
            bands.append(band)
        bands.insert(0, smooths[0])  # lowest frequency
        return bands

    def forward(
        self,
        prediction: NamedTensor,
        target: NamedTensor,
        epoch=None,
        reduce_spatial_dim=True,
    ) -> torch.Tensor:
        """
        prediction/target: (B, pred_steps, N_lat, N_lon, d_f)
        returns (B, pred_steps)
        """

        # Decompose into bands
        pred_bands = self.decompose_bands(prediction)
        tgt_bands = self.decompose_bands(target)

        # Stack along n_scale: (B, pred_steps, N_lat, N_lon, n_scale, d_f)
        pred_stack = torch.stack([pb.tensor for pb in pred_bands], dim=-2)
        tgt_stack = torch.stack([tb.tensor for tb in tgt_bands], dim=-2)

        # Compute per-scale loss
        if self.loss.__class__ == AlmostFairCRPS:
            torch_loss = self.loss(pred_stack, tgt_stack, noise_members=self.lm.noise_members)
        else:
            torch_loss = self.loss(pred_stack, tgt_stack)
        # shape: (B, pred_steps, N_lat, N_lon, n_scale, d_f)

        # Retrieve weights: shape (n_scale, d_f)
        weights = []
        for i, name in enumerate(prediction.feature_names):
            weights.append(self.loss_state_weight[name])  # (n_scale,)
        weights = torch.stack(weights, dim=-1).to(prediction.device)  # (n_scale, d_f)

        # Apply weights
        weighted_loss = torch_loss * weights  # broadcast

        # Reduce over feature + scale dims
        weighted_loss = weighted_loss.sum(dim=-1).sum(dim=-1)  # (B, pred_steps, N_lat, N_lon)

        if not reduce_spatial_dim:
            return weighted_loss

        # Apply spatial mask and reduce
        time_step_mean_loss = (
            torch.sum(
                weighted_loss * self.lm.interior_mask_s,
                dim=target.spatial_dim_idx,
            )
            / self.num_interior
        )
        # Final shape: (B, pred_steps)
        return time_step_mean_loss


