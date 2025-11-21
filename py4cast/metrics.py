import warnings
from pathlib import Path
import os
import numpy as np
import torch
from scipy.fftpack import dct
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from py4cast.datasets.base import DatasetInfo, NamedTensor
from py4cast.plots import plot_log_psd, plot_log_psd_downscaling


class MetricPSDK(Metric):
    """
    Compute the PSD as a function of the wave number for each channels/features
    """

    def __init__(self, save_path: Path, pred_step: int = 0):
        super().__init__()
        self.save_path = save_path
        self.pred_step = pred_step  # timestep of pred where we compute psd

        # Declaration of state, states are reset when self.reset() is called
        # Sum PSD prediction at each epoch
        self.add_state(
            "sum_psd_pred",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        # Sum PSD target at each epoch
        self.add_state(
            "sum_psd_target",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        # Step counter, needed to compute psd mean at each epoch.
        self.add_state("step_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: NamedTensor, targets: NamedTensor, shape: tuple):
        """
        Add PSD to their respective sum. Should be called at each validation step's end
        """
        if preds.tensor.shape != targets.tensor.shape:
            raise ValueError("preds and targets must have the same shape")

        if shape is not None:
            unflatten_dims_name = ["Lon", "Lat"]
            unflattened_size = shape[2:4]
            dim = 2
            # reshape tensor from (batch_size, num_pred_steps, grid, nb_channels)
            # to (batch_size, num_pred_steps, lon, lat, nb_channels)
            preds.unflatten_(dim, unflattened_size, unflatten_dims_name)
            targets.unflatten_(dim, unflattened_size, unflatten_dims_name)

        if self.step_count == 0:
            self.feature_names = preds.feature_names

        # Add PSD to the sums
        self.sum_psd_pred = self.add_psd(
            preds.tensor, self.sum_psd_pred, self.pred_step
        )
        self.sum_psd_target = self.add_psd(
            targets.tensor, self.sum_psd_target, self.pred_step
        )

        # Increment count_step
        self.step_count += 1

        if shape is not None:
            # reshape tensor from (batch_size, num_pred_steps, lon, lat, nb_channels)
            # to (batch_size, num_pred_steps, grid, nb_channels)
            preds.flatten_("ngrid", 2, 3)
            targets.flatten_("ngrid", 2, 3)

    def compute(self, prefix: str = "val") -> dict:
        """
        Compute PSD mean for each channels/features, plot the figure, return a dict.
        Should be called at each epoch's end
        """

        # Compute PSD mean over an epoch
        mean_psd_pred = self.sum_psd_pred / self.step_count
        mean_psd_target = self.sum_psd_target / self.step_count
        features_name = self.feature_names
        nb_channel = mean_psd_pred.shape[0]

        # Compute max wave number
        Rmax = mean_psd_pred.shape[1]
        # Compute all wave number
        k = np.linspace(2 * np.pi / 2.6, Rmax * 2 * np.pi / 2.6, Rmax)

        # dict {"feature_name" : fig}
        dict_psd_metric = dict()
        for c in range(nb_channel):
            fig = plot_log_psd(
                k,
                mean_psd_pred[c].cpu().numpy(),
                mean_psd_target[c].cpu().numpy(),
                f"PSD for {features_name[c]} at +{self.pred_step+1}h",
            )
            np.save(
                self.save_path / f"{prefix}_mean_psd_k_{features_name[c]}_{self.pred_step+1}.npy",
                np.array([k, mean_psd_pred[c].cpu().numpy(), mean_psd_target[c].cpu().numpy()])
            )
            dict_psd_metric[f"{prefix}_mean_psd_k/{features_name[c]}"] = fig
            dest_file = (
                self.save_path
                / f"{prefix}_mean_psd_k/{features_name[c]}_{self.pred_step+1}.png"
            )
            if self.save_path.exists():
                dest_file.parent.mkdir(exist_ok=True)
                fig.savefig(dest_file)

        # Reset metric's state
        self.reset()

        return dict_psd_metric

    def add_psd(self, x: torch.Tensor, sum_psd: torch.Tensor, pred_step: int):
        """
        Add the PSD of x to the variable sum_psd
        """
        # x should be (Batch, channels, Lon, Lat) to be an argument of power_spectral_density
        x = x.permute(0, -1, *range(2, x.dim() - 1), 1)
        channels = x.shape[1]

        # Compute the PSD for each channel of x.
        for c in range(channels):
            psd = power_spectral_density(
                x[:, c : c + 1, :, :, pred_step].cpu().numpy()
            ).squeeze()

            psd_torch = torch.from_numpy(psd).to(device=self.device)

            # Initialize sum_psd as a tensor of dim (nb_features, wave_number)
            if not sum_psd.ndim:
                sum_psd = torch.zeros([x.shape[1], psd.shape[0]], device=self.device)

            # Add PSD for this channel
            sum_psd[c, :] += psd_torch

        return sum_psd

class MetricPSDKdownscaling(MetricPSDK):
    """
    Compute the PSD as a function of the wave number for each channels/features
    """

    def __init__(self, save_path: Path, pred_step: int = 0):
        super().__init__(save_path, pred_step)

        # Declaration of state, states are reset when self.reset() is called
        # Sum PSD lowres at each epoch
        self.add_state(
            "sum_psd_lowres",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def update(self, preds: NamedTensor, targets: NamedTensor, lowres: NamedTensor, shape: tuple):
        """
        Add PSD to their respective sum. Should be called at each validation step's end
        """
        if preds.tensor.shape != targets.tensor.shape:
            raise ValueError("preds and targets must have the same shape")

        if shape is not None:
            unflatten_dims_name = ["Lon", "Lat"]
            unflattened_size = shape[2:4]
            dim = 2
            # reshape tensor from (batch_size, num_pred_steps, grid, nb_channels)
            # to (batch_size, num_pred_steps, lon, lat, nb_channels)
            preds.unflatten_(dim, unflattened_size, unflatten_dims_name)
            targets.unflatten_(dim, unflattened_size, unflatten_dims_name)
            lowres.unflatten_(dim, unflattened_size, unflatten_dims_name)

        if self.step_count == 0:
            self.feature_names = preds.feature_names

        # Add PSD to the sums
        self.sum_psd_pred = self.add_psd(
            preds.tensor, self.sum_psd_pred, self.pred_step
        )
        self.sum_psd_target = self.add_psd(
            targets.tensor, self.sum_psd_target, self.pred_step
        )
        self.sum_psd_lowres = self.add_psd(
            lowres.tensor, self.sum_psd_lowres, self.pred_step
        )

        # Increment count_step
        self.step_count += 1

        if shape is not None:
            # reshape tensor from (batch_size, num_pred_steps, lon, lat, nb_channels)
            # to (batch_size, num_pred_steps, grid, nb_channels)
            preds.flatten_("ngrid", 2, 3)
            targets.flatten_("ngrid", 2, 3)
            lowres.flatten_("ngrid", 2, 3)

    def compute(self, prefix: str = "val") -> dict:
        """
        Compute PSD mean for each channels/features, plot the figure, return a dict.
        Should be called at each epoch's end
        """

        # Compute PSD mean over an epoch
        mean_psd_pred = self.sum_psd_pred / self.step_count
        mean_psd_target = self.sum_psd_target / self.step_count
        mean_psd_lowres = self.sum_psd_lowres / self.step_count
        features_name = self.feature_names
        nb_channel = mean_psd_pred.shape[0]

        # Compute max wave number
        Rmax = mean_psd_pred.shape[1]
        # Compute all wave number
        k = np.linspace(2 * np.pi / 2.6, Rmax * 2 * np.pi / 2.6, Rmax)

        # dict {"feature_name" : fig}
        dict_psd_metric = dict()
        for c in range(nb_channel):
            fig = plot_log_psd_downscaling(
                k,
                mean_psd_pred[c].cpu().numpy(),
                mean_psd_target[c].cpu().numpy(),
                mean_psd_lowres[c].cpu().numpy(),
                f"PSD for {features_name[c]} at +{self.pred_step+1}h",
            )
            np.save(
                self.save_path / f"{prefix}_mean_psd_k_{features_name[c]}_{self.pred_step+1}.npy",
                np.array([k, mean_psd_pred[c].cpu().numpy(), mean_psd_target[c].cpu().numpy(), mean_psd_lowres[c].cpu().numpy()])
            )
            dict_psd_metric[f"{prefix}_mean_psd_k/{features_name[c]}"] = fig
            dest_file = (
                self.save_path
                / f"{prefix}_mean_psd_k/{features_name[c]}_{self.pred_step+1}.png"
            )
            if self.save_path.exists():
                dest_file.parent.mkdir(exist_ok=True)
                fig.savefig(dest_file)

        # Reset metric's state
        self.reset()

        return dict_psd_metric


class MetricPSDKperLeadtimes(Metric):
    """
    Compute PSD across all prediction steps from 1..pred_step+1.
    Saves PSDs for all steps, but only returns dict for last step.
    """

    def __init__(self, save_path: Path, pred_step: int = 0):
        super().__init__()
        self.save_path = save_path
        self.pred_step = pred_step

        # Now it's a tensor placeholder, no lists/None
        self.add_state("sum_psd_mean", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_psd_member", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_psd_target", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("step_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, mean: NamedTensor, member: NamedTensor, targets: NamedTensor, shape: tuple):
        if mean.tensor.shape != targets.tensor.shape:
            raise ValueError("mean and targets must have the same shape")

        if shape is not None:
            mean.unflatten_(2, shape[2:4], ["Lon", "Lat"])
            member.unflatten_(2, shape[2:4], ["Lon", "Lat"])
            targets.unflatten_(2, shape[2:4], ["Lon", "Lat"])

        if self.step_count == 0:
            self.feature_names = mean.feature_names

        # Compute PSDs for all steps
        psd_mean_all = self.compute_all_psd(mean.tensor)
        psd_member_all = self.compute_all_psd(member.tensor)
        psd_target_all = self.compute_all_psd(targets.tensor)


        # Initialize storage on first call
        if self.sum_psd_mean.ndim == 0:  
            self.sum_psd_mean = torch.zeros_like(psd_mean_all, device=self.device)
            self.sum_psd_member = torch.zeros_like(psd_member_all, device=self.device)
            self.sum_psd_target = torch.zeros_like(psd_target_all, device=self.device)

        self.sum_psd_mean += psd_mean_all
        self.sum_psd_member += psd_member_all
        self.sum_psd_target += psd_target_all

        self.step_count += 1

        if shape is not None:
            mean.flatten_("ngrid", 2, 3)
            member.flatten_("ngrid", 2, 3)
            targets.flatten_("ngrid", 2, 3)

    def compute_all_psd(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute PSD for all steps at once.
        Returns shape: (steps, channels, wave_number)
        """
        # Reorder to: (Batch, Channels, Lon, Lat, Steps)
        x = x.permute(0, -1, *range(2, x.dim() - 1), 1)
        steps = x.shape[-1]
        channels = x.shape[1]

        psds = []
        for t in range(steps):
            psd_list = []
            for c in range(channels):
                psd = power_spectral_density(
                    x[:, c : c + 1, :, :, t].cpu().numpy()
                ).squeeze()
                psd_list.append(torch.from_numpy(psd).to(device=self.device))
            psds.append(torch.stack(psd_list, dim=0))  # (channels, wave_number)

        return torch.stack(psds, dim=0)  # (steps, channels, wave_number)

    def compute(self, prefix: str = "val") -> dict:
        dict_psd_metric = {}

        # mean over batches
        mean_psd_mean = self.sum_psd_mean / self.step_count
        mean_psd_member = self.sum_psd_member / self.step_count
        mean_psd_target = self.sum_psd_target / self.step_count
        
        features_name = self.feature_names
        steps, channels, Rmax = mean_psd_mean.shape
        k = np.linspace(2 * np.pi / 2.6, Rmax * 2 * np.pi / 2.6, Rmax)

        # Save PSDs for ALL steps
        for t in range(steps):
            for c in range(channels):
                np.save(
                    self.save_path / f"{prefix}_mean_psd_k_{features_name[c]}_{t+1}.npy",
                    np.array([
                        k,
                        mean_psd_mean[t, c].cpu().numpy(),
                        mean_psd_member[t, c].cpu().numpy(),
                        mean_psd_target[t, c].cpu().numpy(),
                    ])
                )

        # Only log/plot last step
        last_step = steps - 1
        for c in range(channels):
            fig = plot_log_psd_perLeadtimes(
                k,
                mean_psd_mean[last_step, c].cpu().numpy(),
                mean_psd_member[last_step, c].cpu().numpy(),
                mean_psd_target[last_step, c].cpu().numpy(),
                f"PSD for {features_name[c]} at +{last_step+1}h",
            )
            dict_psd_metric[f"{prefix}_mean_psd_k/{features_name[c]}"] = fig

            dest_file = (
                self.save_path
                / f"{prefix}_mean_psd_k/{features_name[c]}_{last_step+1}.png"
            )
            if self.save_path.exists():
                dest_file.parent.mkdir(exist_ok=True)
                fig.savefig(dest_file)

        self.reset()
        return dict_psd_metric


class MetricPSDVar(Metric):
    """
    Compute the RMSE between the target and the prediciton PSD for each channels/features.
    """

    def __init__(self, pred_step: int = 0):
        super().__init__()
        self.pred_step = pred_step  # timestep of pred where we compute psd

        # Declaration of state, states are reset when self.reset() is called
        # Sum RMSE PSD prediction at each epoch
        self.add_state(
            "sum_rmse",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        # Step counter, needed to compute psd mean at each epoch.
        self.add_state("step_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: NamedTensor, targets: NamedTensor, shape: tuple):
        """
        compute the RSME between target and pred PSD.
        called at each end of step
        """
        if shape is not None:
            unflatten_dims_name = ["Lon", "Lat"]
            unflattened_size = shape[2:4]
            dim = 2
            # reshape tensor from (batch_size, num_pred_steps, grid, nb_channels)
            # to (batch_size, num_pred_steps, lon, lat, nb_channels)
            preds.unflatten_(dim, unflattened_size, unflatten_dims_name)
            targets.unflatten_(dim, unflattened_size, unflatten_dims_name)

        if self.step_count == 0:
            self.feature_names = preds.feature_names

        # tensor should be (Batch, channels, Lon, Lat) or (Batch, channels, ngrids)
        # to be an argument of power_spectral_density
        y_pred = preds.tensor.permute(0, -1, *range(2, preds.tensor.dim() - 1), 1)
        y_true = targets.tensor.permute(0, -1, *range(2, targets.tensor.dim() - 1), 1)

        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape")

        channels = y_true.shape[1]
        res = np.zeros((channels,))

        # Compute the RMSE for each channel of x.
        for c in range(channels):
            # Compute PSD
            psd_target = power_spectral_density(
                y_true[:, c : c + 1, :, :, self.pred_step].cpu().numpy()
            )
            psd_pred = power_spectral_density(
                y_pred[:, c : c + 1, :, :, self.pred_step].cpu().numpy()
            )
            # Compute RMSE
            res[c] = np.sqrt(np.mean((np.log10(psd_target) - np.log10(psd_pred)) ** 2))

        # Initialize sum_rmse as a tensor of dim (nb_features)
        if not self.sum_rmse.ndim:  # self.sum_rmse not yet initalized as a tensor
            self.sum_rmse = torch.zeros(channels, device=self.device)

        # Add RMSE for this channel
        self.sum_rmse += torch.from_numpy(res).to(device=self.device)

        # Increment step_count
        self.step_count += 1

        if shape is not None:
            # reshape tensor from (batch_size, num_pred_steps, lon, lat, nb_channels)
            # to (batch_size, num_pred_steps, grid, nb_channels)
            preds.flatten_("ngrid", 2, 3)
            targets.flatten_("ngrid", 2, 3)

    def compute(self, prefix: str = "val") -> dict:
        """
        Compute PSD mean for each channels/features, return a dict.
        Should be called at each epoch's end
        """
        # Compute mean RMSE over an epoch
        mean_psd_rmse = self.sum_rmse / self.step_count
        feature_names = self.feature_names

        # dict {"feature_name" : RMSE mean}
        metric_log_dict = {
            f"{prefix}_rmse_psd/{name}": mean_psd_rmse[i]
            for i, name in enumerate(feature_names)
        }

        # Reset metric's state
        self.reset()

        return metric_log_dict


# Useful function for spectral compute
def dct_2d(x):
    """
    2D dicrete_cosine_transf transform for 2D (square) numpy array
    or for each sample b of BxNxN numpy array. Applies the scipy
    dct function (Type II) along rows and columns.

    """
    assert x.ndim in [2, 3]
    if x.ndim == 3:
        res = dct(
            dct(x.transpose((0, 2, 1)), norm="ortho").transpose((0, 2, 1)),
            norm="ortho",
        )
    else:
        res = dct(dct(x.T, norm="ortho").T, norm="ortho")
    return res


def dct_var(x: np.ndarray) -> np.ndarray:
    """
    compute the bidirectional variance spectrum of the (square) numpy array x

    Inputs :
        x : numpy array, shape is B x 1 x N x N

    Return :
        sigma : 2D numpy array representing the variance spectrum
    """
    n = x.shape[-1]
    # compute the discret cosine transform in 2D of x
    fx = dct_2d(x)
    # compute the variance
    sigma = (1 / n**2) * fx**2
    return sigma


def radial_bin_dct(dct_sig: np.ndarray, center: tuple) -> np.ndarray:
    """
    radial binning on the variance spectrum obtained from the DCT

    Inputs :
        dct_sig : variance spectrum
        center : center of the radial binning (typically the center of the array)

    Return :
        radial_profile : 1D numpy array representing the radially-averaged power spectrum
    """
    y, x = np.indices(dct_sig.shape)

    # computation of radial distance from the center for each point of the spectrum
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    # maximum radial distance
    Rmax = min(x.max(), y.max(), r.max()) // 2

    # double binning for dct
    dct = (
        dct_sig.ravel()[2 * r.ravel()]
        + 0.5 * dct_sig.ravel()[2 * r.ravel() - 1]
        + 0.5 * dct_sig.ravel()[2 * r.ravel() + 1]
    )

    # aggregate binned values and compute the radial profile by averaging
    tbin = np.bincount(r.ravel()[r.ravel() < Rmax], dct[r.ravel() < Rmax])
    nr = np.bincount(r.ravel()[r.ravel() < Rmax])
    radial_profile = tbin / nr

    return radial_profile


def power_spectral_density(x: np.ndarray) -> np.ndarray:
    """
    compute the radially-averaged, sample-averaged power spectral density
    of the data x

    Inputs :
        x : numpy array, shape is B x C x N x N

    Return :
        out : numpy array, shape is (C, Rmax), with R_max defined in radial_bin_dct function
    """
    out_list = []
    channels = x.shape[1]

    for c in range(channels):
        x_c = x[:, c, :, :]

        # compute the variance spectrum and average over the batch dimension
        sig = dct_var(x_c).mean(axis=0)

        # determine the center for radial binning
        center = (sig.shape[0] // 2, sig.shape[1] // 2)

        # compute the radial profile
        out_list.append(radial_bin_dct(sig, center))

    # concatenate for all channels
    out = np.concatenate([np.expand_dims(o, axis=0) for o in out_list], axis=0)
    return out


class MetricACC(Metric):
    """
    Compute the spatially averaged, per pred step Anomaly Correlation Coefficent between the target and the prediction
    Anomalies are computed with respect to "climate" normals that must be provided beforehand
    """

    def __init__(self, dataset_info: DatasetInfo):
        super().__init__()

        # storing climate normals as a tensor
        warnings.warn(
            "You are using ACC metric, which supposes access to climate normals. "
            "From now on these normals are NOT grid point dependent, "
            "there is only one scalar value for each field. "
            "Values of ACC using spatialised climate normals "
            "may be very different from those outputted by this metric."
        )
        names = (
            dataset_info.shortnames["input_output"] + dataset_info.shortnames["output"]
        )
        self.climate_means = dataset_info.stats.to_list("mean", names).to(self.device)

        # adding sum of acc coefficient (to compute mean on each epoch)
        self.add_state(
            "sum_acc",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

        # Step counter, needed to compute psd mean at each epoch.
        self.add_state("step_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: NamedTensor, target: NamedTensor, *args):
        """
        compute the ACC between target and pred.
        prediction/target: (B, pred_steps, N_grid, d_f) or (B, pred_steps, W, H, d_f)
        called at each end of step
        """

        if self.step_count == 0 and self.climate_means.device != preds.tensor.device:
            self.climate_means = self.climate_means.to(preds.tensor.device)
        if self.step_count == 0:
            self.feature_names = preds.feature_names
            self.pred_steps = preds.tensor.shape[1]

        features = preds.tensor.shape[-1]

        # a priori unknown number of spatial dims
        # but they are all after pred_steps and before features
        spatial_dims = tuple(preds.spatial_dim_idx)
        if preds.tensor.shape != target.tensor.shape:
            raise ValueError("preds and target must have the same shape")

        # ACC computation without m
        # https://confluence.ecmwf.int/display/FUG/Section+12.A+Statistical+Concepts+-+Deterministic+Data
        num = (
            (preds.tensor - self.climate_means) * (target.tensor - self.climate_means)
        ).mean(dim=(spatial_dims))
        square_denom = ((preds.tensor - self.climate_means) ** 2).mean(
            dim=(spatial_dims)
        ) * ((target.tensor - self.climate_means) ** 2).mean(dim=(spatial_dims))
        # averageing on batch samples
        res = torch.mean(num / torch.sqrt(square_denom), dim=0)

        # Initialize sum_rmse as a tensor of dim (nb_features)
        if not self.sum_acc.ndim:  # self.sum_rmse not yet initalized as a tensor
            self.sum_acc = torch.zeros(self.pred_steps, features, device=self.device)

        # Add acc for this channel
        self.sum_acc += res.to(device=self.device)

        # Increment step_count
        self.step_count += 1

    def compute(self, prefix: str = "val") -> dict:
        """
        Compute ACC mean for each channels/features, return a dict.
        Should be called at each epoch's end
        """
        # Compute mean RMSE over an epoch
        mean_acc = self.sum_acc / self.step_count
        feature_names = self.feature_names

        # dict {"feature_name" : RMSE mean}
        metric_log_dict = {
            f"{prefix}_acc/{name}_step{j}": mean_acc[j, i]
            for i, name in enumerate(feature_names)
            for j in range(self.pred_steps)
        }

        # Reset metric's state
        self.reset()

        return metric_log_dict
        
class MetricSpread(Metric):
    """
    Compute the spatially averaged, per pred step members spread for both the target (PE-AROME) and the prediction
    """

    def __init__(self, dataset_info: DatasetInfo, pred_step: int = 0):
        super().__init__()
        self.pred_step = pred_step 
        
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        # dist_reduce_fx="cat" because we gather multiple batch before computing std

        self.dataset_name = dataset_info.name

    def update(self, pred: NamedTensor, target: NamedTensor, *args):
        """
        Assuming a batch contains the members of a same ensemble,
        compute the spread of batch=members for both target and pred.
        prediction/target: (B=Mb, pred_steps, N_grid, d_f) or (B=Mb, pred_steps, W, H, d_f)
        called at each end of step
        """

        # a priori unknown number of spatial dims
        # but they are all after pred_steps and before features
        self.spatial_dims = tuple(pred.spatial_dim_idx)
        if pred.tensor.shape != target.tensor.shape:
            raise ValueError("preds and target must have the same shape")

        self.preds.append(pred.tensor)
        self.targets.append(target.tensor)
        
    def compute(self, prefix: str = "val") -> dict:
        """
        Compute spread mean for each channels/features, return a dict.
        Should be called at each epoch's end
        """
        # Ensure concatenating into a single tensor along device dimension
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)
        # Spread computation
        preds_std = preds.std(dim=0).mean(dim=tuple([spatial_dim-1 for spatial_dim in self.spatial_dims]))
        targets_std = targets.std(dim=0).mean(dim=tuple([spatial_dim-1 for spatial_dim in self.spatial_dims]))
        # Reconvert to initial type then reset metric's state
        self.preds = []
        self.targets = []
        self.reset()

        return {"preds": preds_std,
                f"{self.dataset_name}": targets_std,
                }

# class MetricSpread(Metric):
#     """
#     Compute the spatially averaged, per pred step members spread for both the target (PE-AROME) and the prediction
#     """

#     def __init__(self, dataset_info: DatasetInfo, pred_step: int = 0):
#         super().__init__()
#         self.pred_step = pred_step 
        
#         self.add_state("preds", default=[], dist_reduce_fx="cat")
#         # dist_reduce_fx="cat" because we gather multiple batch before computing std

#         self.dataset_name = dataset_info.name

#     def update(self, pred: torch.Tensor, *args):
#         """
#         Assuming a batch contains the members of a same ensemble,
#         compute the spread of batch=members for both target and pred.
#         prediction/target: (B=Mb, pred_steps, N_grid, d_f) or (B=Mb, pred_steps, W, H, d_f)
#         called at each end of step
#         """
#         self.preds.append(pred)
        
#     def compute(self, prefix: str = "val") -> dict:
#         """
#         Compute spread mean for each channels/features, return a dict.
#         Should be called at each epoch's end
#         """
#         # Ensure concatenating into a single tensor along device dimension
#         preds = dim_zero_cat(self.preds)
#         # Spread computation
#         preds_std = preds.std(dim=1).mean(dim=(0,2,3))
#         # Reconvert to initial type then reset metric's state
#         self.preds = []
#         self.reset()

#         return {"preds": preds_std,
#                 }

class MetricMapSpread(Metric):
    """
    Compute the spatially averaged, per pred step members spread for both the target (PE-AROME) and the prediction
    """

    def __init__(self, dataset_info: DatasetInfo, pred_step: int = 0):
        super().__init__()
        self.pred_step = pred_step 
        
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
        # dist_reduce_fx="cat" because we gather multiple batch before computing std

        self.dataset_name = dataset_info.name

    def update(self, pred: torch.Tensor, target: torch.Tensor, *args):
        """
        Assuming a batch contains the members of a same ensemble,
        compute the spread of batch=members for both target and pred.
        prediction/target: (B=Mb, pred_steps, N_grid, d_f) or (B=Mb, pred_steps, W, H, d_f)
        called at each end of step
        """
        if pred.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.preds.append(pred)
        self.targets.append(target)

    def compute(self, prefix: str = "val") -> dict:
        """
        Compute spread mean for each channels/features, return a dict.
        Should be called at each epoch's end
        """
        # Ensure concatenating into a single tensor along device dimension
        preds = dim_zero_cat(self.preds)
        targets = dim_zero_cat(self.targets)
        # Spread computation
        preds_std = preds.std(dim=0)
        targets_std = targets.std(dim=0)
        # Reconvert to initial type then reset metric's state
        self.preds = []
        self.targets = []
        self.reset()

        return {"preds": preds_std,
                "pearo": targets_std,
                }
    
class MetricBias(Metric):
    """
    Compute the skill = RMSE of ensemble mean (Fortin et al. 2014) with TITAN test target rollout as ground truth
    """

    def __init__(self, dataset_info: DatasetInfo, pred_step: int = 0):
        super().__init__()
        self.pred_step = pred_step 
        
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("pearo", default=[], dist_reduce_fx="cat")
        # dist_reduce_fx="cat" because we gather multiple batch before computing std

        self.dataset_name = dataset_info.name

    def update(self, pred: torch.Tensor, target: torch.Tensor, *args):
        """
        Assuming a batch contains the members of a same ensemble,
        compute the spread of batch=members for both target and pred.
        prediction/target: (B=Mb, pred_steps, N_grid, d_f) or (B=Mb, pred_steps, W, H, d_f)
        called at each end of step
        """

        if pred.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.preds.append(pred)
        self.pearo.append(target)

    def compute(self, num_batch, prefix: str = "val") -> dict:
        """
        Compute spread mean for each channels/features, return a dict.
        Should be called at each epoch's end
        """
        # Ensure concatenating into a single tensor along device dimension
        preds = dim_zero_cat(self.preds)
        pearo = dim_zero_cat(self.pearo)
        env_path = os.environ["TITAN_PROXY_OBS"]
        target = torch.from_numpy(np.load(f"{env_path}/titan_gt_rescaled_batch_{num_batch}.npy")).to(preds.device)
        target = target[:,1:,...]  # remove time dim 0
        # print(f"target shape: {target.shape}")
        # print(f"preds shape: {preds.shape}")
        # target = target[...,:8]
        # skill computation
        preds_bias = torch.mean(preds.mean(dim=0) - target[0], dim=(1,2))
        pearo_bias = torch.mean(pearo.mean(dim=0) - target[0], dim=(1,2))
        # Reconvert to initial type then reset metric's state
        self.preds = []
        self.pearo = []
        self.reset()

        return {"preds": preds_bias, # (pred_steps, d_f)
                "pearo": pearo_bias, # (pred_steps, d_f)
                }

class MetricSquaredError(Metric):
    """
    Compute the skill = RMSE of ensemble mean (Fortin et al. 2014) with TITAN test target rollout as ground truth
    """

    def __init__(self, dataset_info: DatasetInfo, pred_step: int = 0):
        super().__init__()
        self.pred_step = pred_step 
        
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("pearo", default=[], dist_reduce_fx="cat")
        # dist_reduce_fx="cat" because we gather multiple batch before computing std

        self.dataset_name = dataset_info.name

    def update(self, pred: torch.Tensor, target: torch.Tensor, *args):
        """
        Assuming a batch contains the members of a same ensemble,
        compute the spread of batch=members for both target and pred.
        prediction/target: (B=Mb, pred_steps, N_grid, d_f) or (B=Mb, pred_steps, W, H, d_f)
        called at each end of step
        """

        if pred.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.preds.append(pred)
        self.pearo.append(target)

    def compute(self, num_batch, prefix: str = "val") -> dict:
        """
        Compute spread mean for each channels/features, return a dict.
        Should be called at each epoch's end
        """
        # Ensure concatenating into a single tensor along device dimension
        preds = dim_zero_cat(self.preds)
        pearo = dim_zero_cat(self.pearo)
        env_path = os.environ["TITAN_PROXY_OBS"]
        target = torch.from_numpy(np.load(f"{env_path}/titan_gt_rescaled_batch_{num_batch}.npy")).to(preds.device)
        target = target[:,1:,...]  # remove time dim 0
        # print(f"target shape: {target.shape}")
        # print(f"preds shape: {preds.shape}")
        # target = target[...,:8]
        # skill computation
        preds_skill = torch.mean((preds.mean(dim=0) - target[0]) ** 2, dim=(1,2))
        pearo_skill = torch.mean((pearo.mean(dim=0) - target[0]) ** 2, dim=(1,2))
        # Reconvert to initial type then reset metric's state
        self.preds = []
        self.pearo = []
        self.reset()

        return {"preds": preds_skill, # (pred_steps, d_f)
                "pearo": pearo_skill, # (pred_steps, d_f)
                }

class MetricUnbiasedVariance(Metric):
    """
    Compute the unbiased variance of the ensemble members.
    """

    def __init__(self, dataset_info: DatasetInfo, pred_step: int = 0):
        super().__init__()
        self.pred_step = pred_step 
        
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("pearo", default=[], dist_reduce_fx="cat")
        # dist_reduce_fx="cat" because we gather multiple batch before computing std

        self.dataset_name = dataset_info.name

    def update(self, pred: torch.Tensor, target: torch.Tensor, *args):
        """
        Assuming a batch contains the members of a same ensemble,
        compute the spread of batch=members for both target and pred.
        prediction/target: (B=Mb, pred_steps, N_grid, d_f) or (B=Mb, pred_steps, W, H, d_f)
        called at each end of step
        """

        if pred.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.preds.append(pred)
        self.pearo.append(target)

    def compute(self, prefix: str = "val") -> dict:
        """
        Compute spread mean for each channels/features, return a dict.
        Should be called at each epoch's end
        """
        # Ensure concatenating into a single tensor along device dimension
        preds = dim_zero_cat(self.preds)
        pearo = dim_zero_cat(self.pearo)
        # skill computation
        preds_unbVar = torch.mean(torch.var(preds, dim=0, correction=1), dim=(1,2))
        pearo_unbVar = torch.mean(torch.var(pearo, dim=0, correction=1), dim=(1,2))
        # Reconvert to initial type then reset metric's state
        self.preds = []
        self.pearo = []
        self.reset()

        return {"preds": preds_unbVar, # (pred_steps, d_f)
                "pearo": pearo_unbVar, # (pred_steps, d_f)
                }

class MetricCRPS(Metric):
    """
    Compute the CRPS with TITAN test target rollout as ground truth
    """

    def __init__(self, dataset_info: DatasetInfo, pred_step: int = 0):
        super().__init__()
        self.pred_step = pred_step

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("pearo", default=[], dist_reduce_fx="cat")
        self.dataset_name = dataset_info.name

    def update(self, pred: torch.Tensor, target: torch.Tensor, *args):
        """
        Assuming a batch contains the members of a same ensemble,
        compute the spread of batch=members for both target and pred.
        predictions: (B=Mb, pred_steps, N_grid, d_f) or (B=Mb, pred_steps, W, H, d_f)
        called at each end of step
        """
        if pred.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.preds.append(pred)
        self.pearo.append(target)

    def compute(self, num_batch, prefix: str = "val") -> dict:
        """
        Computes the CRPS score using ensemble approximation:
        CRPS = 1/N * sum_i |x_i - y| - 0.5/N^2 * sum_{i,j} |x_i - x_j|
        where x_i are ensemble members and y is the target.
        Uses a for-loop to reduce GPU memory usage when computing pairwise diffs.
        """
        preds = dim_zero_cat(self.preds)  # (ensemble_size, pred_steps, ..., d_f)
        pearo = dim_zero_cat(self.pearo)  # (ensemble_size, pred_steps, ..., d_f)
        ensemble_size = preds.shape[0]

        # Load full ground truth
        env_path = os.environ["TITAN_PROXY_OBS"]
        target = torch.from_numpy(
            np.load(f"{env_path}/titan_gt_rescaled_batch_{num_batch}.npy")
        ).to(preds.device)
        target = target[:,1:,...]  # remove time dim 0

        # Expand target to match preds
        target = target.expand_as(preds)

        # Term 1: Mean absolute error between ensemble and ground truth
        term1 = torch.mean(torch.abs(preds - target), dim=0)  # (pred_steps, ..., d_f)
        # Term 2: Mean absolute difference between ensemble members using for-loop
        term2 = torch.zeros_like(term1)
        count = ensemble_size/2 # classic CRPS, count diagonal terms / 2
        for i in range(ensemble_size):
            for j in range(i + 1, ensemble_size):
                diff_ij = torch.abs(preds[i] - preds[j])  # (pred_steps, ..., d_f)
                term2 += diff_ij
                count += 1
        term2 = term2 / count  # (pred_steps, ..., d_f)
        preds_crps = term1 - 0.5 * term2

        # Term 1: Mean absolute error between ensemble and ground truth
        term1 = torch.mean(torch.abs(pearo - target), dim=0)  # (pred_steps, ..., d_f)
        # Term 2: Mean absolute difference between ensemble members using for-loop
        term2 = torch.zeros_like(term1)
        count = ensemble_size/2 # classic CRPS 
        for i in range(ensemble_size):
            for j in range(i + 1, ensemble_size):
                diff_ij = torch.abs(pearo[i] - pearo[j])  # (pred_steps, ..., d_f)
                term2 += diff_ij
                count += 1
        term2 = term2 / count  # (pred_steps, ..., d_f)
        pearo_crps = term1 - 0.5 * term2

        # Reset internal state
        self.preds = []
        self.pearo = []
        self.reset()

        return {"preds": preds_crps,
                "pearo": pearo_crps} # (pred_steps, N_grid, d_f)