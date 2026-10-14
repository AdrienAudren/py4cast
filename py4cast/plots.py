import json
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy, error
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Literal
from datetime import timedelta
import cartopy
import einops
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from scipy.stats import wasserstein_distance
import numpy as np
import torch
from PIL import Image
from torchmetrics import Metric
from tueplots import bundles, figsizes

if TYPE_CHECKING:
    from py4cast.lightning import AutoRegressiveLightning

from mfai.torch.namedtensor import NamedTensor


@dataclass(slots=True)
class DomainInfo:
    """
    Information required for plotting.
    """

    grid_limits: Tuple[float, float, float, float]
    projection: cartopy.crs


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of the page width.
    """
    bundle = bundles.neurips2023(usetex=False, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (original_figsize[0] / fraction, original_figsize[1])
    return bundle


@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_error_map(errors, shortnames, units, title=None, step_duration=3):
    """
    Plot a heatmap of errors of different variables at different predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_duration * pred_hor_i  # Prediction horiz. in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (h)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [f"{name} ({unit})" for name, unit in zip(shortnames, units)]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


# Plot psd_pred and psd_target in function of k
def plot_log_psd(
    k: np.ndarray, psd_pred: np.ndarray, psd_target: np.ndarray, title: str = ""
):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(k, psd_pred, label="pred")
    ax.plot(k, psd_target, label="target")
    ax.set_xlabel("k")
    ax.set_ylabel("psd_k")
    ax.legend()
    ax.set_title(title)
    plt.yscale("log")
    plt.close()
    return fig

def plot_log_psd_downscaling(
    k: np.ndarray, psd_pred: np.ndarray, psd_target: np.ndarray, psd_lowres: np.ndarray, title: str = ""
):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(k, psd_pred, label="pred")
    ax.plot(k, psd_target, label="AROME")
    ax.plot(k, psd_lowres, label="ARPEGE")
    ax.set_xlabel("k")
    ax.set_ylabel("psd_k")
    ax.legend()
    ax.set_title(title)
    plt.yscale("log")
    plt.close()
    return fig

def plot_log_psd_perLeadtimes(
    k: np.ndarray, psd_mean: np.ndarray, psd_member: np.ndarray, psd_target: np.ndarray, title: str = ""
):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(k, psd_mean, label="Ensemble Mean")
    ax.plot(k, psd_member, label="Ensemble Member")
    ax.plot(k, psd_target, label="PEARO")
    ax.set_xlabel("k")
    ax.set_ylabel("psd_k")
    ax.legend()
    ax.set_title(title)
    plt.yscale("log")
    plt.close()
    return fig


@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_prediction_downscaling(
    pred,
    target,
    lowres,
    interior_mask,
    domain_info: DomainInfo,
    title=None,
    vrange=None,
    cmap="Spectral_r",
):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    # mask_reshaped = obs_mask.reshape(*grid_shape)
    pixel_alpha = interior_mask.clamp(0.7, 1).cpu().numpy()  # Faded border region

    fig, axes = plt.subplots(
        1, 3, figsize=(19, 7), subplot_kw={"projection": domain_info.projection}
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, lowres, pred)):
        # ax.coastlines()  # Add coastline outlines
        # data_grid = data.reshape(*grid_shape).cpu().numpy()
        im = ax.imshow(
            data.cpu().numpy(),
            origin="lower",
            extent=domain_info.grid_limits,
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

    # Ticks and labels
    axes[0].set_title("AROME", size=15)
    axes[1].set_title("ARPEGE", size=15)
    axes[2].set_title("8 steps Flow Matching", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig

@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_prediction_tp(
    pred,
    target,
    interior_mask,
    domain_info: DomainInfo,
    title=None,
    vrange=None,
    cmap="Spectral_r",
):
    """
    Plot example prediction and ground truth.
    Each has shape (N_grid,)
    """

    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Ensure zero values are rendered as white
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_under("white")

    # Small positive threshold so exact zeros become "under"
    norm = colors.Normalize(vmin=max(vmin, 1.0), vmax=vmax)

    # Faded border region
    pixel_alpha = interior_mask.clamp(0.7, 1).cpu().numpy()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 7),
        subplot_kw={"projection": domain_info.projection},
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        # ax.coastlines()

        im = ax.imshow(
            data.cpu().numpy(),
            origin="lower",
            extent=domain_info.grid_limits,
            alpha=pixel_alpha,
            cmap=cmap_obj,
            norm=norm,
        )

    # Titles
    axes[0].set_title("PEARO", size=15)
    axes[1].set_title("Prediction", size=15)

    # Colorbar
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig

@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_prediction(
    pred,
    target,
    interior_mask,
    domain_info: DomainInfo,
    title=None,
    vrange=None,
    cmap="plasma",
):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    # mask_reshaped = obs_mask.reshape(*grid_shape)
    pixel_alpha = interior_mask.clamp(0.7, 1).cpu().numpy()  # Faded border region

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7), subplot_kw={"projection": domain_info.projection}
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        # ax.coastlines()  # Add coastline outlines
        # data_grid = data.reshape(*grid_shape).cpu().numpy()
        im = ax.imshow(
            data.cpu().numpy(),
            origin="lower",
            extent=domain_info.grid_limits,
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

    # Ticks and labels
    axes[0].set_title("PEARO", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig

@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_spread_prediction_(
    pred,
    target,
    interior_mask,
    domain_info: DomainInfo,
    title=None,
    vrange=None,
    cmap="Spectral",
):
    """
    Plot example prediction and ground truth.
    Each has shape (N_grid,)
    """

    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Make zero values appear white
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_under("white")

    # Exact zeros fall below vmin
    norm = colors.Normalize(vmin=max(vmin, 1.0), vmax=vmax)

    # Faded border region
    pixel_alpha = interior_mask.clamp(0.7, 1).cpu().numpy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        subplot_kw={"projection": domain_info.projection},
    )

    # Plot target and prediction
    for ax, data in zip(axes, (target, pred)):
        # ax.coastlines()

        im = ax.imshow(
            data.cpu().numpy(),
            origin="lower",
            extent=domain_info.grid_limits,
            alpha=pixel_alpha,
            cmap=cmap_obj,
            norm=norm,
        )

    # Titles
    axes[0].set_title("PEARO", size=15)
    axes[1].set_title("Prediction", size=15)

    # Colorbar
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig

@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_spread_prediction(
    pred,
    target,
    interior_mask,
    domain_info: DomainInfo,
    title=None,
    vrange=None,
    cmap="viridis",
):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    # mask_reshaped = obs_mask.reshape(*grid_shape)
    pixel_alpha = interior_mask.clamp(0.7, 1).cpu().numpy()  # Faded border region

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7), subplot_kw={"projection": domain_info.projection}
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        # ax.coastlines()  # Add coastline outlines
        # data_grid = data.reshape(*grid_shape).cpu().numpy()
        im = ax.imshow(
            data.cpu().numpy(),
            origin="lower",
            extent=domain_info.grid_limits,
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

    # Ticks and labels
    axes[0].set_title("PEARO", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig

@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_1prediction(
    data,
    interior_mask,
    domain_info=None,
    title=None,
    vrange=None,
    cmap="seismic",
):
    """
    Plot a single prediction or ground truth field.

    Args:
        data (torch.Tensor): 2D field with shape (H, W).
        interior_mask (torch.Tensor): 2D mask with same shape as data, used for alpha blending.
        domain_info (object, optional): Object with attributes:
            - projection: matplotlib/cartopy projection (optional)
            - grid_limits: tuple (xmin, xmax, ymin, ymax)
        title (str, optional): Figure title.
        vrange (tuple, optional): (vmin, vmax) for color scaling.
        cmap (str, optional): Colormap for imshow.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """

    # Convert tensors to numpy
    data_np = data.detach().cpu().numpy()
    mask_np = interior_mask.clamp(0.7, 1).detach().cpu().numpy()

    # Determine value range
    if vrange is None:
        vmin = float(data_np.min())
        vmax = float(data_np.max())
    else:
        vmin, vmax = vrange

    # Create a single figure and axis
    fig, ax = plt.subplots(
        figsize=(10, 6),
        subplot_kw={"projection": getattr(domain_info, "projection", None)} if domain_info else {},
        constrained_layout=True,  # avoids colorbar/tight_layout issues
    )

    # Plot data
    im = ax.imshow(
        data_np,
        origin="lower",
        extent=getattr(domain_info, "grid_limits", None) if domain_info else None,
        alpha=mask_np,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    # Title and labels
    ax.set_title(title or "Field", fontsize=15)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", aspect=30)
    cbar.ax.tick_params(labelsize=10)

    # Return the figure object for saving or displaying
    return fig



@matplotlib.rc_context(fractional_plot_bundle(1))
def plot_spatial_error(
    error,
    obs_mask,
    domain_info: DomainInfo,
    title=None,
    vrange=None,
):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    # Set up masking of border region

    pixel_alpha = obs_mask.clamp(0.7, 1).cpu().numpy()  # Faded border region

    fig, ax = plt.subplots(
        figsize=(5, 4.8), subplot_kw={"projection": domain_info.projection}
    )

    # ax.coastlines()  # Add coastline outlines
    im = ax.imshow(
        error.cpu().numpy(),
        origin="lower",
        extent=domain_info.grid_limits,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )
    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)
    return fig


class Plotter(ABC):
    """
    Abstract class to plot errors between prediction and target.
    Prediction and target had already been computed.
    """

    @abstractmethod
    def update(
        self,
        obj: "AutoRegressiveLightning",
        prediction: NamedTensor,
        target: NamedTensor,
    ) -> None:
        """
        Do an action when "step" is trigger
        """
        pass

    @abstractmethod
    def on_step_end(self, obj: "AutoRegressiveLightning", label: str = "") -> None:
        """
        Do an action when "end" is trigger
        """
        pass


class MapPlot(Plotter):
    """
    Abstract Observer used to plot maps during training
    """

    def __init__(
        self,
        num_samples_to_plot: int,
        num_features_to_plot: Union[None, int] = None,
        prefix: str = "Test",
        save_path: Path = None,
        count: int = 0,
    ):
        self.num_samples_to_plot = num_samples_to_plot
        self.plotted_examples = 0
        self.prefix = prefix
        self.num_features_to_plot = num_features_to_plot
        self.save_path = save_path
        self.count = count

    def update(
        self,
        obj: "AutoRegressiveLightning",
        prediction: NamedTensor,
        target: NamedTensor,
    ) -> None:
        """
        Update. Should be call by "on_{training/validation/test}_step
        """
        pred = deepcopy(prediction).tensor  # In order to not modify the input
        targ = deepcopy(target).tensor  # In order to not modify the input

        # Here we reshape output from GNNS to be on the grid
        if prediction.num_spatial_dims == 1:
            pred = einops.rearrange(
                pred, "b t (x y) n -> b t x y n", x=obj.grid_shape[0]
            )
            targ = einops.rearrange(
                targ, "b t (x y) n -> b t x y n", x=obj.grid_shape[0]
            )

        if (
            obj.trainer.is_global_zero
            # and self.plotted_examples < self.num_samples_to_plot
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                pred.shape[0], self.num_samples_to_plot - self.plotted_examples
            )
            # Rescale to original data scale
            if obj.noise_strategy == "diffusion":
                # unnormalized [-1,1]
                max_stat = obj.stats.to_list("max", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                min_stat = obj.stats.to_list("min", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                prediction_rescaled = (pred + 1)*(max_stat - min_stat) / 2 + min_stat
                target_rescaled = (targ + 1)*(max_stat - min_stat) / 2 + min_stat
            else:
                std = obj.stats.to_list("std", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                mean = obj.stats.to_list("mean", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                prediction_rescaled = pred * std + mean
                target_rescaled = targ * std + mean

                # save in numpy file std and mean
                if self.save_path is not None:  
                    np.save(self.save_path / "std.npy", std.cpu().numpy())
                    np.save(self.save_path / "mean.npy", mean.cpu().numpy())
            
            # Iterate over the examples
            # We assume examples are already on grid
            for pred_slice, target_slice in zip(
                prediction_rescaled[:n_additional_examples],
                target_rescaled[:n_additional_examples],
            ):
                # Each slice is (pred_steps, Nlat, Nlon, features)
                self.plotted_examples += 1  # Increment already here

                var_vmin = target_slice.flatten(0, 2).min(dim=0)[0].cpu().numpy()
                var_vmax = target_slice.flatten(0, 2).max(dim=0)[0].cpu().numpy()
                var_vranges = list(zip(var_vmin, var_vmax))

                var_min_diff = (torch.abs(pred_slice - target_slice)).flatten(0, 2).min(dim=0)[0].cpu().numpy()
                var_max_diff = (torch.abs(pred_slice - target_slice)).flatten(0, 2).max(dim=0)[0].cpu().numpy()
                var_vranges_diff = list(zip(var_min_diff, var_max_diff))
                # np.save(self.save_path / f"var_vranges_diff.npy", np.array(var_vranges_diff))

                feature_names = (
                    prediction.feature_names[: self.num_features_to_plot]
                    if self.num_features_to_plot
                    else prediction.feature_names
                )

                self.plot_map(
                    obj,
                    pred_slice,
                    target_slice,
                    feature_names,
                    var_vranges,
                )

                # self.plot_diff_map(
                #     obj,
                #     pred_slice,
                #     target_slice,
                #     feature_names,
                #     var_vranges_diff,
                # )

    @abstractmethod
    def plot_map(
        self,
        obj: "AutoRegressiveLightning",
        prediction: torch.tensor,
        target: torch.tensor,
        feature_names: List[str],
        var_vranges: List,
    ) -> None:
        pass

    # @abstractmethod
    # def plot_diff_map(
    #     self,
    #     obj: "AutoRegressiveLightning",
    #     prediction: torch.tensor,
    #     target: torch.tensor,
    #     feature_names: List[str],
    #     var_vranges: List,
    # ) -> None:
    #     pass

    def on_step_end(self, obj: "AutoRegressiveLightning", label: str = "") -> None:
        """
        Do an action when "end" is trigger
        """
        pass
    


class PredictionTimestepPlot(MapPlot):
    """
    Observer used to plot prediction and target for each timestep.
    """

    def plot_map(
        self,
        obj: "AutoRegressiveLightning",
        prediction: torch.tensor,
        target: torch.tensor,
        feature_names: List[str],
        var_vranges: List,
    ) -> None:
        # Prediction and target: (pred_steps, Nlat, Nlon, features)
        # Iterate over prediction horizon time steps
        paths_dict = defaultdict(list)
        
        base_dt = obj.test_dataset_info.start
        
        for t_i, (pred_t, target_t) in enumerate(zip(prediction, target), start=1):
            # Create one figure per variable at this time step
            # This generate a matplotlib warning as more than 20 figures are plotted.
            dt_i = base_dt + timedelta(hours=t_i)
            
            units = [obj.dataset_info.units[name] for name in feature_names]
            var_figs = [
                plot_prediction(
                    pred_t[:, :, var_i],
                    target_t[:, :, var_i],
                    obj.interior_2d[:, :, 0],
                    title=f"{var_name} ({var_unit})\n"
                    f"[start: {base_dt}] "
                    f"t+{t_i}h {dt_i}",
                    vrange=var_vrange,
                    domain_info=obj.dataset_info.domain_info,
                ) if var_name != "pearo_tp_0m" else plot_prediction_tp(
                    pred_t[:, :, var_i],
                    target_t[:, :, var_i],
                    obj.interior_2d[:, :, 0],
                    title=f"{var_name} ({var_unit})\n"
                    f"[start: {base_dt}] "
                    f"t+{t_i}h {dt_i}",
                    vrange=var_vrange,
                    domain_info=obj.dataset_info.domain_info,
                )
                for var_i, (var_name, var_unit, var_vrange) in enumerate(
                    zip(feature_names, units, var_vranges)
                )
            ]
            # TODO : don't create all figs at once to avoid matplotlib warning
            tensorboard = obj.logger.experiment
            for var_name, fig in zip(feature_names, var_figs):
                fig_name = f"timestep_evol_per_param/{var_name}_example_{self.plotted_examples}"
                tensorboard.add_figure(fig_name, fig, t_i)
                fig_full_name = f"{fig_name}_{t_i}.png"
                if self.save_path is not None and self.save_path.exists():
                    dest_file = self.save_path / fig_full_name
                    paths_dict[var_name].append(dest_file)
                    dest_file.parent.mkdir(exist_ok=True)
                    fig.savefig(dest_file)

                if obj.mlflow_logger:
                    run_id = obj.mlflow_logger.version
                    obj.mlflow_logger.experiment.log_figure(
                        run_id=run_id,
                        figure=fig,
                        artifact_file=f"figures/{fig_full_name}",
                    )

                plt.close(fig)

        # for var_name, paths in paths_dict.items():
        #     if len(paths) > 1:
        #         make_gif(
        #             paths, self.save_path / f"timestep_evol_per_param/{var_name}.gif"
        #         )

    def plot_diff_map(
        self,
        obj: "AutoRegressiveLightning",
        prediction: torch.tensor,
        target: torch.tensor,
        feature_names: List[str],
        var_vranges: List,
    ) -> None:
        # Prediction and target: (pred_steps, Nlat, Nlon, features)
        paths_dict = defaultdict(list)

        # Start from t=1 to compute diff with t-1
        for t_i in range(1, len(prediction)):
            mae = torch.abs(prediction[t_i] - target[t_i])

            units = [obj.dataset_info.units[name] for name in feature_names]
            var_figs = [
                plot_1prediction(
                    mae[:, :, var_i],
                    obj.interior_2d[:, :, 0],
                    title=f"{var_name} ({var_unit}), MAE t={t_i} ({obj.dataset_info.pred_step*t_i} h)",
                    vrange=var_vrange,
                    domain_info=obj.dataset_info.domain_info,
                    cmap="seismic",
                )
                for var_i, (var_name, var_unit, var_vrange) in enumerate(
                    zip(feature_names, units, var_vranges)
                )
            ]

            tensorboard = obj.logger.experiment
            for var_name, fig in zip(feature_names, var_figs):
                fig_name = f"timestep_diff_per_param/{var_name}_example_{self.plotted_examples}"
                tensorboard.add_figure(fig_name, fig, t_i)
                fig_full_name = f"{fig_name}_{t_i}.png"
                if self.save_path is not None and self.save_path.exists():
                    dest_file = self.save_path / fig_full_name
                    paths_dict[var_name].append(dest_file)
                    dest_file.parent.mkdir(exist_ok=True)
                    fig.savefig(dest_file)

                if obj.mlflow_logger:
                    run_id = obj.mlflow_logger.version
                    obj.mlflow_logger.experiment.log_figure(
                        run_id=run_id,
                        figure=fig,
                        artifact_file=f"figures/{fig_full_name}",
                    )

                plt.close(fig)

class SpreadTimestepPlot(MapPlot):
    """
    Observer used to plot prediction and target spread map for each timestep.
    """
    def __init__(
        self,
        metric: Metric,
        dataset_name: str,
        num_samples_to_plot: int,
        num_features_to_plot: Union[None, int] = None,
        prefix: str = "Test",
        save_path: Path = None,
    ):
        super().__init__(
            num_samples_to_plot=num_samples_to_plot,
            num_features_to_plot=num_features_to_plot,
            prefix=prefix,
            save_path=save_path,
        )
        self.metric = metric
        self.dataset_name = dataset_name

        

    def update(
        self,
        obj: "AutoRegressiveLightning",
        prediction: NamedTensor,
        target: NamedTensor,
        
    ) -> None:
        """
        Update. Should be call by "on_{training/validation/test}_step
        """
        pred = prediction.clone().tensor  # In order to not modify the input
        targ = target.clone().tensor  # In order to not modify the input

        # Here we reshape output from GNNS to be on the grid
        if prediction.num_spatial_dims == 1:
            prediction.tensor = einops.rearrange(
                prediction.tensor, "b t (x y) n -> b t x y n", x=obj.grid_shape[0]
            )
            target.tensor = einops.rearrange(
                target.tensor, "b t (x y) n -> b t x y n", x=obj.grid_shape[0]
            )

        if self.plotted_examples < self.num_samples_to_plot:
            # Rescale to original data scale
            if obj.noise_strategy == "diffusion":
                # unnormalized [-1,1]
                max_stat = obj.stats.to_list("max", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                min_stat = obj.stats.to_list("min", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                prediction_rescaled = (pred + 1)*(max_stat - min_stat) / 2 + min_stat
                target_rescaled = (targ + 1)*(max_stat - min_stat) / 2 + min_stat
            else:
                std = obj.stats.to_list("std", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                mean = obj.stats.to_list("mean", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                prediction_rescaled = pred * std + mean
                target_rescaled = targ * std + mean

            self.metric.update(prediction_rescaled, target_rescaled)
            std_dict = self.metric.compute(prefix="test")
            prediction_std = std_dict["preds"]
            target_std = std_dict["pearo"]
            
            # save numpy target std
            env_path = os.environ["PEARO_STD_MAPS"]
            # if self.count == 0:
            #     np.save(f"{env_path}/pearo_std_map.npy", target_std.cpu().numpy())
            # save with the count in name
            if self.plotted_examples <= 90:
                np.save(f"{env_path}/pearo_std_map_{self.plotted_examples}.npy", target_std.cpu().numpy())
            elif self.plotted_examples == 93 :
                np.save(f"{env_path}/pearo_std_map_91.npy", target_std.cpu().numpy())
            
            prediction_var_vmin = prediction_std.flatten(0, 2).min(dim=0)[0].cpu().numpy() 
            prediction_var_vmax = prediction_std.flatten(0, 2).max(dim=0)[0].cpu().numpy() 
            target_var_vmin = target_std.flatten(0, 2).min(dim=0)[0].cpu().numpy()
            target_var_vmax = target_std.flatten(0, 2).max(dim=0)[0].cpu().numpy()

            var_vmin = np.minimum(prediction_var_vmin, target_var_vmin)
            var_vmax = target_var_vmax
            #var_vmax = np.maximum(target_var_vmax, prediction_var_vmax)

            var_vranges = list(zip(var_vmin, var_vmax))

            feature_names = (
                prediction.feature_names[: self.num_features_to_plot]
                if self.num_features_to_plot
                else prediction.feature_names
            )

            self.plotted_examples += 1

            if obj.trainer.is_global_zero:
                # Plot figs only on main process

                self.plot_map(
                    obj,
                    prediction_std,
                    target_std,
                    feature_names,
                    var_vranges,
                )

                # self.plot_std_diff_map(
                #     obj,
                #     prediction_std,
                #     target_std,
                #     feature_names,
                #     var_vranges,
                # )

    def plot_map(
        self,
        obj: "AutoRegressiveLightning",
        prediction: torch.tensor,
        target: torch.tensor,
        feature_names: List[str],
        var_vranges: List,
    ) -> None:
        # Prediction and target: (pred_steps, Nlat, Nlon, features)
        # Iterate over prediction horizon time steps
        paths_dict = defaultdict(list)
        
        base_dt = obj.test_dataset_info.start
        
        for t_i, (pred_t, target_t) in enumerate(zip(prediction, target), start=1):
            # Create one figure per variable at this time step
            # This generate a matplotlib warning as more than 20 figures are plotted.
            dt_i = base_dt + timedelta(hours=t_i)
            
            units = [obj.dataset_info.units[name] for name in feature_names]
            var_figs = [
                plot_spread_prediction(
                    pred_t[:, :, var_i],
                    target_t[:, :, var_i],
                    obj.interior_2d[:, :, 0],
                    title=f"{var_name} 16-member spread\n"
                    f"[start: {base_dt}] "
                    f"t+{t_i}h {dt_i}",
                    vrange=var_vrange,
                    domain_info=obj.dataset_info.domain_info,
                    cmap="viridis",
                )
                for var_i, (var_name, var_unit, var_vrange) in enumerate(
                    zip(feature_names, units, var_vranges)
                )
            ]
            # TODO : don't create all figs at once to avoid matplotlib warning
            tensorboard = obj.logger.experiment
            for var_name, fig in zip(feature_names, var_figs):
                fig_name = f"timestep_spread_evol_per_param/{var_name}_example_{self.plotted_examples}"
                tensorboard.add_figure(fig_name, fig, t_i)
                fig_full_name = f"{fig_name}_{t_i}.png"
                if self.save_path is not None and self.save_path.exists():
                    dest_file = self.save_path / fig_full_name
                    paths_dict[var_name].append(dest_file)
                    dest_file.parent.mkdir(exist_ok=True)
                    fig.savefig(dest_file)

                if obj.mlflow_logger:
                    run_id = obj.mlflow_logger.version
                    obj.mlflow_logger.experiment.log_figure(
                        run_id=run_id,
                        figure=fig,
                        artifact_file=f"figures/{fig_full_name}",
                    )

                plt.close(fig)

    def plot_std_diff_map(
        self,
        obj: "AutoRegressiveLightning",
        prediction_std: torch.tensor,
        target_std: torch.tensor,
        feature_names: List[str],
        var_vranges: List,
    ) -> None:
        # Prediction_std and target_std: (pred_steps, Nlat, Nlon, features)
        paths_dict = defaultdict(list)

        # Start from t=1 to compute diff with t-1
        for t_i in range(1, len(prediction_std)):
            pred_diff = prediction_std[t_i] - prediction_std[t_i - 1]
            target_diff = target_std[t_i] - target_std[t_i - 1]

            units = [obj.dataset_info.units[name] for name in feature_names]
            var_figs = [
                plot_prediction(
                    pred_diff[:, :, var_i],
                    target_diff[:, :, var_i],
                    obj.interior_2d[:, :, 0],
                    title=f"{var_name} ({var_unit}), Δstd t={t_i} ({obj.dataset_info.pred_step*t_i} h)",
                    vrange=var_vrange,
                    domain_info=obj.dataset_info.domain_info,
                    cmap="seismic",
                )
                for var_i, (var_name, var_unit, var_vrange) in enumerate(
                    zip(feature_names, units, var_vranges)
                )
            ]

            tensorboard = obj.logger.experiment
            for var_name, fig in zip(feature_names, var_figs):
                fig_name = f"timestep_spread_diff_per_param/{var_name}_example_{self.plotted_examples}"
                tensorboard.add_figure(fig_name, fig, t_i)
                fig_full_name = f"{fig_name}_{t_i}.png"
                if self.save_path is not None and self.save_path.exists():
                    dest_file = self.save_path / fig_full_name
                    paths_dict[var_name].append(dest_file)
                    dest_file.parent.mkdir(exist_ok=True)
                    fig.savefig(dest_file)

                if obj.mlflow_logger:
                    run_id = obj.mlflow_logger.version
                    obj.mlflow_logger.experiment.log_figure(
                        run_id=run_id,
                        figure=fig,
                        artifact_file=f"figures/{fig_full_name}",
                    )

                plt.close(fig)

    

class EnScorePlot(MapPlot):
    """
    Observer used to plot ensemble scores for each timestep.
    """
    def __init__(
        self,
        bias_metric: Metric,
        error_metric: Metric,
        unbVar_metric: Metric,
        crps_metric: Metric,
        ES_metric: Metric,
        vario_metric: Metric,
        dataset_name: str,
        num_samples_to_plot: int,
        num_features_to_plot: Union[None, int] = None,
        prefix: str = "Test",
        save_path: Path = None,
        num_pred_steps_test: int = None,
    ):
        super().__init__(
            num_samples_to_plot=num_samples_to_plot,
            num_features_to_plot=num_features_to_plot,
            prefix=prefix,
            save_path=save_path,
        )
        self.bias_metric = bias_metric
        self.error_metric = error_metric
        self.unbVar_metric = unbVar_metric
        self.crps_metric = crps_metric
        self.ES_metric = ES_metric
        self.vario_metric = vario_metric
        self.dataset_name = dataset_name

        self.crps_pred = np.zeros((self.num_samples_to_plot, num_pred_steps_test, self.num_features_to_plot))
        self.crps_pearo = np.zeros((self.num_samples_to_plot, num_pred_steps_test, self.num_features_to_plot))
        self.fcrps_pred = np.zeros((self.num_samples_to_plot, num_pred_steps_test, self.num_features_to_plot))
        self.fcrps_pearo = np.zeros((self.num_samples_to_plot, num_pred_steps_test, self.num_features_to_plot))
        
        self.ES_pred = np.zeros((self.num_samples_to_plot, num_pred_steps_test))
        self.ES_pearo = np.zeros((self.num_samples_to_plot, num_pred_steps_test))
        self.fES_pred = np.zeros((self.num_samples_to_plot, num_pred_steps_test))
        self.fES_pearo = np.zeros((self.num_samples_to_plot, num_pred_steps_test))

        self.bias_pred = np.zeros((self.num_samples_to_plot, num_pred_steps_test, self.num_features_to_plot))
        self.bias_pearo = np.zeros((self.num_samples_to_plot, num_pred_steps_test, self.num_features_to_plot))

        self.mean_unbVar_pred = np.zeros((self.num_samples_to_plot, num_pred_steps_test, self.num_features_to_plot))
        self.mean_unbVar_pearo = np.zeros((self.num_samples_to_plot, num_pred_steps_test, self.num_features_to_plot))
        
        self.mean_error_pred = np.zeros((self.num_samples_to_plot, num_pred_steps_test, self.num_features_to_plot))
        self.mean_error_pearo = np.zeros((self.num_samples_to_plot, num_pred_steps_test, self.num_features_to_plot))

        self.vario_pred = np.zeros((self.num_samples_to_plot, num_pred_steps_test))
        self.vario_pearo = np.zeros((self.num_samples_to_plot, num_pred_steps_test))
        
    def update(
        self,
        obj: "AutoRegressiveLightning",
        prediction: NamedTensor,
        target: NamedTensor,
        
    ) -> None:
        """
        Update. Should be call by "on_{training/validation/test}_step
        """
        pred = prediction.clone().tensor  # In order to not modify the input
        targ = target.clone().tensor  # In order to not modify the input

        # Here we reshape output from GNNS to be on the grid
        if prediction.num_spatial_dims == 1:
            prediction.tensor = einops.rearrange(
                prediction.tensor, "b t (x y) n -> b t x y n", x=obj.grid_shape[0]
            )
            target.tensor = einops.rearrange(
                target.tensor, "b t (x y) n -> b t x y n", x=obj.grid_shape[0]
            )

        if self.plotted_examples < self.num_samples_to_plot: 
            # Rescale to original data scale
            if obj.noise_strategy == "diffusion":
                # unnormalized [-1,1]
                max_stat = obj.stats.to_list("max", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                min_stat = obj.stats.to_list("min", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                prediction_rescaled = (pred + 1)*(max_stat - min_stat) / 2 + min_stat
                target_rescaled = (targ + 1)*(max_stat - min_stat) / 2 + min_stat
            else:
                std = obj.stats.to_list("std", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                mean = obj.stats.to_list("mean", prediction.feature_names).to(
                    pred, non_blocking=True
                )
                prediction_rescaled = pred * std + mean
                target_rescaled = targ * std + mean

            self.crps_metric.update(prediction_rescaled, target_rescaled)
            crps_dict = self.crps_metric.compute(num_batch=self.plotted_examples, prefix="test")
            
            self.ES_metric.update(prediction_rescaled, target_rescaled)
            ES_dict = self.ES_metric.compute(num_batch=self.plotted_examples, prefix="test")

            self.unbVar_metric.update(prediction_rescaled, target_rescaled)
            unbVar_dict = self.unbVar_metric.compute(num_batch=self.plotted_examples, prefix="test")
            
            self.error_metric.update(prediction_rescaled, target_rescaled)
            error_dict = self.error_metric.compute(num_batch=self.plotted_examples, prefix="test")

            self.bias_metric.update(prediction_rescaled, target_rescaled)
            bias_dict = self.bias_metric.compute(num_batch=self.plotted_examples, prefix="test")

            self.vario_metric.update(prediction_rescaled, target_rescaled)
            vario_dict = self.vario_metric.compute(num_batch=self.plotted_examples, prefix="test")

            feature_names = (
                prediction.feature_names[: self.num_features_to_plot]
                if self.num_features_to_plot
                else prediction.feature_names
            )
            units = [obj.dataset_info.units[name] for name in feature_names]

            if obj.trainer.is_global_zero:

                self.crps_pred[self.plotted_examples] = crps_dict["preds"].mean(dim=(1, 2)).cpu().numpy() # mean over spatial dimensions
                self.crps_pearo[self.plotted_examples] = crps_dict["pearo"].mean(dim=(1, 2)).cpu().numpy()
                self.fcrps_pred[self.plotted_examples] = crps_dict["preds_fair"].mean(dim=(1, 2)).cpu().numpy()
                self.fcrps_pearo[self.plotted_examples] = crps_dict["pearo_fair"].mean(dim=(1, 2)).cpu().numpy()

                self.vario_pred[self.plotted_examples] = vario_dict["preds"].mean(dim=(1, 2)).cpu().numpy() # mean over spatial dimensions
                self.vario_pearo[self.plotted_examples] = vario_dict["pearo"].mean(dim=(1, 2)).cpu().numpy()
                
                self.ES_pred[self.plotted_examples] = ES_dict["preds"].cpu().numpy()
                self.ES_pearo[self.plotted_examples] = ES_dict["pearo"].cpu().numpy()
                self.fES_pred[self.plotted_examples] = ES_dict["preds_fair"].cpu().numpy()
                self.fES_pearo[self.plotted_examples] = ES_dict["pearo_fair"].cpu().numpy()

                self.mean_unbVar_pred[self.plotted_examples] = unbVar_dict["preds"].cpu().numpy()
                self.mean_unbVar_pearo[self.plotted_examples] = unbVar_dict["pearo"].cpu().numpy()

                self.mean_error_pred[self.plotted_examples] = error_dict["preds"].cpu().numpy()
                self.mean_error_pearo[self.plotted_examples] = error_dict["pearo"].cpu().numpy()

                self.bias_pred[self.plotted_examples] = bias_dict["preds"].cpu().numpy()
                self.bias_pearo[self.plotted_examples] = bias_dict["pearo"].cpu().numpy()

            self.plotted_examples += 1

        if self.plotted_examples == self.num_samples_to_plot: # end of epoch

            if obj.trainer.is_global_zero:

                # Average numpy arrays over all examples
                mean_crps_pred = np.mean(self.crps_pred[:self.plotted_examples], axis=0) # shape (num_pred_steps_test, self.num_features_to_plot)
                mean_crps_pearo = np.mean(self.crps_pearo[:self.plotted_examples], axis=0) # shape (num_pred_steps_test, self.num_features_to_plot)
                mean_fcrps_pred = np.mean(self.fcrps_pred[:self.plotted_examples], axis=0) # shape (num_pred_steps_test, self.num_features_to_plot)
                mean_fcrps_pearo = np.mean(self.fcrps_pearo[:self.plotted_examples], axis=0) # shape (num_pred_steps_test, self.num_features_to_plot
                
                mean_vario_pred = np.mean(self.vario_pred[:self.plotted_examples], axis=0) # shape (num_pred_steps_test, self.num_features_to_plot)
                mean_vario_pearo = np.mean(self.vario_pearo[:self.plotted_examples], axis=0) # shape (num_pred_steps_test, self.num_features_to_plot)
                
                mean_ES_pred = np.mean(self.ES_pred[:self.plotted_examples], axis=0) # shape (num_pred_steps_test)
                mean_ES_pearo = np.mean(self.ES_pearo[:self.plotted_examples], axis=0) # shape (num_pred_steps_test)
                mean_fES_pred = np.mean(self.fES_pred[:self.plotted_examples], axis=0) # shape (num_pred_steps_test)
                mean_fES_pearo = np.mean(self.fES_pearo[:self.plotted_examples], axis=0) # shape (num_pred_steps_test)

                mean_unbVar_pred = np.sqrt(np.mean(self.mean_unbVar_pred[:self.plotted_examples], axis=0))
                mean_unbVar_pearo = np.sqrt(np.mean(self.mean_unbVar_pearo[:self.plotted_examples], axis=0))

                mean_rmse_pred = np.sqrt(np.mean(self.mean_error_pred[:self.plotted_examples], axis=0))
                mean_rmse_pearo = np.sqrt(np.mean(self.mean_error_pearo[:self.plotted_examples], axis=0))

                ssr_pred = np.sqrt(17/16)*np.divide(mean_unbVar_pred, mean_rmse_pred)
                ssr_pearo = np.sqrt(17/16)*np.divide(mean_unbVar_pearo, mean_rmse_pearo)

                mean_bias_pred = np.mean(self.bias_pred[:self.plotted_examples], axis=0) # shape (num_pred_steps_test, self.num_features_to_plot)
                mean_bias_pearo = np.mean(self.bias_pearo[:self.plotted_examples], axis=0) # shape (num_pred_steps_test, self.num_features_to_plot)

                # Plot figs only on main process
                for metric_name in ["crps", "fcrps", "ES", "fES", "bias", "spread_error_ratio", "spread", "error", "vario"]:
                    if metric_name == "crps":
                        self.plot_curves(
                            obj,
                            metric_name,
                            mean_crps_pred,
                            mean_crps_pearo,
                            feature_names,
                        )
                    elif metric_name == "fcrps":
                        self.plot_curves(
                            obj,
                            metric_name,
                            mean_fcrps_pred,
                            mean_fcrps_pearo,
                            feature_names,
                        )
                    elif metric_name == "vario":
                        self.plot_curves(
                            obj,
                            metric_name,
                            mean_vario_pred,
                            mean_vario_pearo,
                            feature_names,
                        )
                    elif metric_name == "ES":
                        self.plot_curves(
                            obj,
                            metric_name,
                            mean_ES_pred,
                            mean_ES_pearo,
                            feature_names,
                        )
                    elif metric_name == "fES":
                        self.plot_curves(
                            obj,
                            metric_name,
                            mean_fES_pred,
                            mean_fES_pearo,
                            feature_names,
                        )
                    elif metric_name == "bias":
                        self.plot_curves(
                            obj,
                            metric_name,
                            mean_bias_pred,
                            mean_bias_pearo,
                            feature_names,
                        )
                    elif metric_name == "spread_error_ratio":
                        self.plot_curves(
                            obj,
                            metric_name,
                            ssr_pred,
                            ssr_pearo,
                            feature_names,
                        )
                    elif metric_name == "spread":
                        self.plot_curves(
                            obj,
                            metric_name,
                            mean_unbVar_pred,
                            mean_unbVar_pearo,
                            feature_names,
                        )
                    elif metric_name == "error":
                        self.plot_curves(
                            obj,
                            metric_name,
                            mean_rmse_pred,
                            mean_rmse_pearo,
                            feature_names,
                        )

    def plot_curves(
        self,
        obj: "AutoRegressiveLightning",
        metric_name: Literal["crps", "fcrps", "ES", "fES", "bias", "spread_error_ratio", "spread", "error", "vario"],
        mean_metric_preds: np.ndarray,  # (pred_steps, features)
        mean_metric_pearo: np.ndarray,  # (pred_steps, features)
        feature_names: List[str],
    ) -> None:

        sns.set(style="whitegrid", context="talk", font_scale=1.2)
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12
        })

        paths_dict = defaultdict(list)
        units = [obj.dataset_info.units[name] for name in feature_names]
        var_figs = []
        
        if metric_name == "ES" or metric_name == "fES" or metric_name == "vario":
            
            lead_times = np.arange(1, mean_metric_preds.shape[0] + 1)
            plt.figure(figsize=(8, 6))
            sns.lineplot(x=lead_times, y=mean_metric_preds, marker="o", label="preds", linewidth=2.5, color="purple")
            np.save(self.save_path / f"preds_{metric_name}_plot_per_param.npy", mean_metric_preds)
            
            sns.lineplot(x=lead_times, y=mean_metric_pearo, marker="o", label="pearo", linewidth=2.5, color="green")
            np.save(self.save_path / f"pearo_{metric_name}_plot_per_param.npy", mean_metric_pearo)

            # Labels and title
            plt.xlabel("Lead Time")
            plt.ylabel(metric_name.upper())
            plt.title(f"{metric_name.upper()} Across Forecast Lead Times")
            plt.legend(loc="best", frameon=True)
            plt.tight_layout()

            var_figs.append(plt.gcf())
            
            tensorboard = obj.logger.experiment
            fig_name = f"{metric_name}_plot_per_param/ES"
            tensorboard.add_figure(fig_name, var_figs[0])
            fig_full_name = f"{fig_name}.png"
            if self.save_path is not None and self.save_path.exists():
                dest_file = self.save_path / fig_full_name
                paths_dict["ES"].append(dest_file)
                dest_file.parent.mkdir(exist_ok=True)
                var_figs[0].savefig(dest_file)
            
            if obj.mlflow_logger:
                run_id = obj.mlflow_logger.version
                obj.mlflow_logger.experiment.log_figure(
                    run_id=run_id,
                    figure=var_figs[0],
                    artifact_file=f"figures/{fig_full_name}",
                )
                
            plt.close(var_figs[0])
        
        else:

            for var_i, (var_name, var_unit) in enumerate(zip(feature_names, units)):

                lead_times = np.arange(1, mean_metric_preds.shape[0] + 1)
                plt.figure(figsize=(8, 6))
                sns.lineplot(x=lead_times, y=mean_metric_preds[:, var_i], marker="o", label="preds", linewidth=2.5, color="purple")
                np.save(self.save_path / f"preds_{metric_name}_plot_per_param_{var_name}.npy", mean_metric_preds[:, var_i])
                
                sns.lineplot(x=lead_times, y=mean_metric_pearo[:, var_i], marker="o", label="pearo", linewidth=2.5, color="green")
                np.save(self.save_path / f"pearo_{metric_name}_plot_per_param_{var_name}.npy", mean_metric_pearo[:, var_i])

                # Labels and title
                plt.xlabel("Lead Time")
                plt.ylabel(metric_name.upper())
                plt.title(f"{var_name} {metric_name.upper()} Across Forecast Lead Times")
                plt.legend(loc="best", frameon=True)
                plt.tight_layout()

                var_figs.append(plt.gcf())
            
            # TODO : don't create all figs at once to avoid matplotlib warning
            tensorboard = obj.logger.experiment
            for var_name, fig in zip(feature_names, var_figs):
                fig_name = f"{metric_name}_plot_per_param/{var_name}"
                tensorboard.add_figure(fig_name, fig)
                fig_full_name = f"{fig_name}.png"
                if self.save_path is not None and self.save_path.exists():
                    dest_file = self.save_path / fig_full_name
                    paths_dict[var_name].append(dest_file)
                    dest_file.parent.mkdir(exist_ok=True)
                    fig.savefig(dest_file)

                if obj.mlflow_logger:
                    run_id = obj.mlflow_logger.version
                    obj.mlflow_logger.experiment.log_figure(
                        run_id=run_id,
                        figure=fig,
                        artifact_file=f"figures/{fig_full_name}",
                    )

                plt.close(fig)

    
    def plot_map(
        self,
        obj,
        prediction,
        target,
        feature_names,
        var_vranges,
    ):
        # This plotter may not need maps—so just ignore inputs.
        # But the method must exist for the class to be instantiable.
        return


class PredictionEpochPlot(MapPlot):
    """
    Observer used to plot prediction and target for max timestep at each epoch.
    Used to visualize improvement of model over epochs
    """

    def plot_map(
        self,
        obj: "AutoRegressiveLightning",
        prediction: torch.tensor,
        target: torch.tensor,
        feature_names: List[str],
        var_vranges: List,
    ) -> None:
        # We keep only the max timestep of the pred and target: shape (Nlat, Nlon, features)
        max_step = prediction.shape[0]
        pred_t, target_t = prediction[max_step - 1], target[max_step - 1]

        # Create one figure per variable at this time step
        # This generate a matplotlib warning as more than 20 figures are plotted.
        leadtime = obj.dataset_info.pred_step * max_step
        var_figs = [
            plot_prediction(
                pred_t[:, :, var_i],
                target_t[:, :, var_i],
                obj.interior_2d[:, :, 0],
                title=f"{var_name} ({var_unit}), "
                f"t={max_step} ({leadtime} h) - epoch {obj.current_epoch}",
                vrange=var_vrange,
                domain_info=obj.dataset_info.domain_info,
            )
            for var_i, (var_name, var_unit, var_vrange) in enumerate(
                zip(
                    feature_names,
                    [obj.dataset_info.units[name] for name in feature_names],
                    var_vranges,
                )
            )
        ]

        tensorboard = obj.logger.experiment
        for var_name, fig in zip(feature_names, var_figs):
            fig_name = (
                f"epoch_evol_per_param/{var_name}_example_{self.plotted_examples}"
            )
            tensorboard.add_figure(fig_name, fig, obj.current_epoch)
            fig_full_name = f"{fig_name}_{obj.current_epoch}.png"
            if self.save_path is not None:
                dest_file = self.save_path / fig_full_name
                dest_file.parent.mkdir(exist_ok=True)
                fig.savefig(dest_file)

            if obj.mlflow_logger:
                run_id = obj.mlflow_logger.version
                obj.mlflow_logger.experiment.log_figure(
                    run_id=run_id, figure=fig, artifact_file=f"figures/{fig_full_name}"
                )

        plt.close("all")  # Close all figs for this time step, saves memory


class StateErrorPlot(Plotter):
    """
    Produce a figure where the error for each variable is shown
    with respect to step
    """

    def __init__(
        self,
        metrics: Dict[str, Metric],
        prefix: str = "Test",
        save_path: Path = None,
    ):
        self.metrics = metrics
        self.prefix = prefix
        self.losses = {}
        self.shortnames = []
        self.units = []
        self.initialized = False
        self.save_path = save_path
        for metric in self.metrics:
            self.losses[metric] = []
        self.count=1
        self.epoch_dict_loss = {}

    def update(
        self,
        obj: "AutoRegressiveLightning",
        prediction: NamedTensor,
        target: NamedTensor,
    ) -> None:
        """
        Compute the metric. Append to a dictionnary
        """
        for name in self.metrics:

            if name == "std":
                if not self.initialized:
                    self.metrics[name].update(prediction, target)
                    self.epoch_dict_loss = self.metrics[name].compute(prefix="test")
                else:
                    self.metrics[name].update(prediction, target)
                    dict_loss = self.metrics[name].compute(prefix="test")
                    for key, value in dict_loss.items():
                        self.epoch_dict_loss[key] += value
                    self.count+=1
            else:
                self.losses[name].append(
                    obj.trainer.strategy.reduce(
                        self.metrics[name](prediction, target), reduce_op="mean"
                    ).cpu()
                )
        if not self.initialized:
            self.shortnames = prediction.feature_names
            self.units = [
                obj.dataset_info.units[name] for name in prediction.feature_names
            ]
            self.initialized = True

    def on_step_end(self, obj: "AutoRegressiveLightning", label: str = "") -> None:
        """
        Make the summary figure
        """
        tensorboard = obj.logger.experiment
        if obj.trainer.is_global_zero:
            for name in self.metrics: 

                if name == "std":
                    for key, value in self.epoch_dict_loss.items():
                        self.epoch_dict_loss[key] /= self.count

                    # Log metrics in tensorboard, with x axis as forecast timestep
                    for t in range(loss.shape[0]):
                        for k in range(loss.shape[1]):
                            scalar_name = f"{label}_{name}/timestep_{self.shortnames[k]}"
                            values = {legend: array[t][k] for legend, array in self.epoch_dict_loss.items()}
                            tensorboard.add_scalars(scalar_name, values, t + 1)
                else:
                    loss_tensor = torch.cat(self.losses[name], dim=0)
                    loss = torch.mean(loss_tensor, dim=0)

                    # Log metrics in tensorboard, with x axis as forecast timestep
                    loss_dict = {self.shortnames[k]: [] for k in range(loss.shape[1])}
                    for t in range(loss.shape[0]):
                        for k in range(loss.shape[1]):
                            scalar_name = f"{label}_{name}/timestep_{self.shortnames[k]}"
                            tensorboard.add_scalar(scalar_name, loss[t][k], t + 1)
                            loss_dict[self.shortnames[k]].append(loss[t][k].item())

                    # Plot the score card
                    if not obj.trainer.sanity_checking:
                        fig = plot_error_map(
                            loss,
                            self.shortnames,
                            self.units,
                            step_duration=obj.dataset_info.pred_step,
                        )

                        # log it in tensorboard
                        fig_name = f"score_cards/{self.prefix}_{name}"
                        tensorboard.add_figure(fig_name, fig, obj.current_epoch)
                        fig_full_name = f"{fig_name}.png"
                        if self.save_path is not None:
                            dest_file = self.save_path / fig_full_name
                            dest_file.parent.mkdir(exist_ok=True)
                            fig.savefig(dest_file)

                        if obj.mlflow_logger:
                            run_id = obj.mlflow_logger.version
                            obj.mlflow_logger.experiment.log_figure(
                                run_id=run_id,
                                figure=fig,
                                artifact_file=f"figures/{fig_full_name}",
                            )
                        plt.close(fig)

                        if self.save_path is not None:
                            # Save metrics in json file
                            with open(
                                self.save_path / f"{label}_{name}_scores.json", "w"
                            ) as json_file:
                                json.dump(loss_dict, json_file)
            # Free memory
            [self.losses[name].clear() for name in self.metrics]


class SpatialErrorPlot(Plotter):
    """
    Produce a map which shows where the error are accumulating (all variables together).
    """

    def __init__(self, prefix: str = "Test"):
        self.spatial_loss_maps = []
        self.prefix = prefix

    def update(
        self,
        obj: "AutoRegressiveLightning",
        prediction: NamedTensor,
        target: NamedTensor,
    ) -> None:
        spatial_loss = obj.loss(prediction, target, reduce_spatial_dim=False)
        # Getting only spatial loss for the required val_step_errors
        if prediction.num_spatial_dims == 1:
            spatial_loss = einops.rearrange(
                spatial_loss, "b t (x y) -> b t x y ", x=obj.grid_shape[0]
            )
        self.spatial_loss_maps.append(
            obj.trainer.strategy.reduce(spatial_loss, reduce_op="mean").cpu()
        )  # (B, N_log, N_lat, N_lon)

    def on_step_end(self, obj: "AutoRegressiveLightning", label: str = "") -> None:
        """
        Make the summary figure
        """
        # (N_test, N_log, N_lat, N_lon)
        if obj.trainer.is_global_zero:
            spatial_loss_tensor = torch.cat(self.spatial_loss_maps, dim=0)
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, N_lat, N_lon)

            loss_map_figs = [
                plot_spatial_error(
                    loss_map,
                    obj.interior_2d[:, :, 0],
                    title=f"{self.prefix} loss, t={t_i} ({obj.dataset_info.pred_step*t_i} h)",
                    domain_info=obj.dataset_info.domain_info,
                )
                for t_i, loss_map in enumerate(mean_spatial_loss)
            ]
            tensorboard = obj.logger.experiment

            for t_i, fig in enumerate(loss_map_figs):
                fig_full_name = f"spatial_error_{label}/{self.prefix}_loss"
                tensorboard.add_figure(fig_full_name, fig, t_i)

                if obj.mlflow_logger:
                    run_id = obj.mlflow_logger.version
                    obj.mlflow_logger.experiment.log_figure(
                        run_id=run_id,
                        figure=fig,
                        artifact_file=f"figures/{fig_full_name}.png",
                    )
            plt.close()

        self.spatial_loss_maps.clear()