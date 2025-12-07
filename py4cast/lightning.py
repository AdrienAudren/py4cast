import json
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import einops
import matplotlib
import mlflow.pytorch
import torch
import torch.nn.functional as F
import math
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities import rank_zero_only
from mfai.torch.models.base import ModelType
from mfai.torch.models.utils import (
    expand_to_batch,
    features_last_to_second,
    features_second_to_last,
)
from mlflow.models.signature import infer_signature
from torchinfo import summary
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup

from py4cast.datasets import get_datasets
from py4cast.datasets.base import DatasetInfo, ItemBatch, NamedTensor, Statics
from py4cast.io.outputs import GribSavingSettings, save_named_tensors_to_grib
from py4cast.losses import ScaledLoss, WeightedLoss, MultiScaleLoss
from py4cast.metrics import MetricACC, MetricPSDK, MetricPSDKdownscaling, MetricPSDKperLeadtimes, MetricPSDVar, MetricSpread, MetricMapSpread, MetricBias, MetricSquaredError, MetricUnbiasedVariance, MetricCRPS
from py4cast.models import build_model_from_settings, get_model_kls_and_settings
from py4cast.models import registry as model_registry
from py4cast.plots import (
    PredictionEpochPlot,
    PredictionTimestepPlot,
    SpreadTimestepPlot,
    EnScorePlot,
    SpatialErrorPlot,
    StateErrorPlot,
)
from py4cast.utils import str_to_dtype

import numpy as np

PLOT_PERIOD: int = 1

TIMESTEPS: int = 1000 


@dataclass
class PlDataModule(LightningDataModule):
    """
    DataModule to encapsulate data splits and data loading.
    """

    def __init__(
        self,
        dataset_name: str = "dummy",
        num_input_steps: int = 1,
        num_pred_steps_train: int = 1,
        num_pred_steps_val_test: int = 1,
        batch_size: int = 2,
        num_workers: int = 1,
        prefetch_factor: int | None = None,
        pin_memory: bool = False,
        dataset_conf: Path | None = None,
        predict_conf: Dict | None = None,
        ensemble_metrics: bool = False,
        noise_members: int = 0,
        noise_strategy: Literal["forcing", "stochastic_EPA", "ddim", "CondLayerNorm"] = "forcing",
        shortcut_length: int = None,
        dynamical_downscaling: bool = False,
        features_to_plot: List[str] = ["t2m"],
    ):
        super().__init__()
        self.num_input_steps = num_input_steps
        self.num_pred_steps_train = num_pred_steps_train
        self.num_pred_steps_val_test = num_pred_steps_val_test
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.noise_members = noise_members
        self.noise_strategy = noise_strategy
        self.shortcut_length = shortcut_length
        self.dynamical_downscaling = dynamical_downscaling
        self.ensemble_metrics = ensemble_metrics
        self.features_to_plot = features_to_plot


        # Get dataset in initialisation to have access to this attribute before method trainer.fit
        self.train_ds, self.val_ds, self.test_ds = get_datasets(
            dataset_name,
            num_input_steps,
            num_pred_steps_train,
            num_pred_steps_val_test,
            noise_members,
            noise_strategy,
            dynamical_downscaling,
            shortcut_length,
            dataset_conf,
            predict_conf,
        )

    @property
    def len_train_dl(self):
        return len(
            self.train_ds.torch_dataloader(
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                prefetch_factor=self.prefetch_factor,
                pin_memory=self.pin_memory,
            )
        )

    @property
    def train_dataset_info(self) -> DatasetInfo:
        return self.train_ds.dataset_info

    @property
    def infer_ds(self):
        return self.test_ds

    def train_dataloader(self):
        return self.train_ds.torch_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self):
        return self.val_ds.torch_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self):
        return self.test_ds.torch_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def predict_dataloader(self):
        return self.test_ds.torch_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
        )


@rank_zero_only
def rank_zero_init(model_kls, model_settings, statics: Statics):
    if hasattr(model_kls, "rank_zero_setup"):
        model_kls.rank_zero_setup(model_settings, statics.meshgrid)


class AutoRegressiveLightning(LightningModule):
    """
    Auto-regressive lightning module for predicting meteorological fields.
    """

    def __init__(
        self,
        settings_init_args: dict,
        # args linked from trainer and datamodule
        dataset_info,  # Don't put type hint here or CLI doesn't work
        len_train_loader: int,
        infer_ds,
        dataset_name: str = "dummy",
        dataset_conf: Path | None = None,  # TO DELETE ?
        num_input_steps: int = 1,
        num_pred_steps_train: int = 1,
        num_pred_steps_val_test: int = 1,
        batch_size: int = 2,
        noise_members: int = 0,
        noise_strategy: Literal["forcing", "stochastic_EPA", "ddim", "CondLayerNorm"] = "forcing",
        shortcut_length: int = None,
        dynamical_downscaling: bool = False,
        ensemble_metrics: bool = False,
        features_to_plot: List[str] = ["t2m"],
        # non-linked args
        model_name: Literal[tuple(model_registry.keys())] = "HalfUNet",
        model_conf: Path | None = None,
        loss_name: Literal["mse", "mae", "afcrps", "ClassicCRPS"] = "mse",
        num_inter_steps: int = 1,
        num_samples_to_plot: int = 1,
        training_strategy: Literal["diff_ar", "scaled_ar"] = "diff_ar",
        channels_last: bool = False,
        io_conf: Path | None = None,
        mask_ratio: float = 0,
        learning_rate: float = 1e-4,
        min_learning_rate: float = 1e-6,
        num_warmup_steps: int = 0,
        betas: tuple = (0.9, 0.999),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.infer_ds = infer_ds
        self.settings_init_args = settings_init_args
        self.dataset_name = dataset_name
        self.dataset_conf = dataset_conf
        self.dataset_info = dataset_info
        self.batch_size = batch_size
        self.noise_members = noise_members
        self.noise_strategy = noise_strategy
        self.shortcut_length = shortcut_length
        self.dynamical_downscaling = dynamical_downscaling
        self.ensemble_metrics = ensemble_metrics
        self.features_to_plot = features_to_plot
        self.model_conf = model_conf
        self.model_name = model_name
        self.num_input_steps = num_input_steps
        self.num_pred_steps_train = num_pred_steps_train
        self.num_pred_steps_val_test = num_pred_steps_val_test
        self.num_inter_steps = num_inter_steps
        self.num_samples_to_plot = num_samples_to_plot
        self.training_strategy = training_strategy
        self.len_train_loader = len_train_loader
        self.channels_last = channels_last
        self.io_conf = io_conf
        self.mask_ratio = mask_ratio
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.betas = betas

        if self.num_inter_steps > 1 and self.num_input_steps > 1:
            raise AttributeError(
                "It is not possible to have multiple input steps when num_inter_steps > 1."
                f"Get num_input_steps :{self.num_input_steps} and num_inter_steps: {self.num_inter_steps}"
            )
        ALLOWED_STRATEGIES = ("diff_ar", "scaled_ar")
        if self.training_strategy not in ALLOWED_STRATEGIES:
            raise AttributeError(
                f"Unknown strategy {self.training_strategy}, allowed strategies are {ALLOWED_STRATEGIES}"
            )

        self.save_hyperparameters()  # write hparams.yaml in save folder

        # Load static features for grid/data
        # We do not want to change dataset statics inplace
        # Otherwise their is some problem with transform_statics and parameters_saving
        # when relaoding from checkpoint
        statics = deepcopy(dataset_info.statics)
        # Init object of register_dict
        self.diff_stats = dataset_info.diff_stats
        self.stats = dataset_info.stats

        # Keeping track of grid shape
        self.grid_shape = statics.grid_shape
        # For example plotting
        self.plotted_examples = 0

        # For storing spatial loss maps during evaluation
        self.spatial_loss_maps = []

        # class variables to log loss during training, for tensorboad custom scalar
        self.training_step_losses = []
        self.validation_step_losses = []

        # Set model input/output grid features based on dataset tensor shapes
        num_grid_static_features = statics.grid_statics.dim_size("features")

        # Compute the number of input features for the neural network
        # Should be directly supplied by datasetinfo ?

        num_input_features = (
            num_input_steps * dataset_info.weather_dim
            + num_grid_static_features
            + dataset_info.forcing_dim
            + (1 if self.noise_strategy == "forcing" else 0)
        )
        # print(f"num_input_features : {num_input_features}")
        # print(f"num_grid_static_features : {num_grid_static_features}")
        # print(f"num_forcing_features : {dataset_info.forcing_dim}")
        # print(f"num_weather_features : {dataset_info.weather_dim}")

        num_output_features = dataset_info.weather_dim

        model_kls, model_settings = get_model_kls_and_settings(
            model_name, self.settings_init_args
        )

        # All processes should wait until rank zero
        # has done the initialization (like creating a graph)
        rank_zero_init(model_kls, model_settings, statics)

        self.model, model_settings = build_model_from_settings(
            model_name,
            num_input_features,
            num_output_features,
            self.settings_init_args,
            statics.grid_shape,
        )
        if channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # We transform and register the statics after the model has been set up
        # This change the dimension of all statics
        if self.model.model_type == ModelType.GRAPH:
            # Graph model, we flatten the statics spatial dims
            statics.grid_statics.flatten_("ngrid", 0, 1)
            statics.border_mask = statics.border_mask.flatten(0, 1)
            statics.interior_mask = statics.interior_mask.flatten(0, 1)

        # Register interior and border mask.
        statics.register_buffers(self)

        self.num_spatial_dims = statics.grid_statics.num_spatial_dims

        self.register_buffer(
            "grid_static_features",
            expand_to_batch(statics.grid_statics.tensor, batch_size),
            persistent=False,
        )

        self._kernel_cache = {}  # cache kernels per (n, device, dtype)

        # We need to instantiate the loss after statics had been transformed.
        # Indeed, the statics used should be in the right dimensions.
        # MSE loss, need to do reduction ourselves to get proper weighting
        if loss_name == "mse":
            self.loss = WeightedLoss("MSELoss", reduction="none")
            # self.loss = MultiScaleLoss("MSELoss", reduction="none")
        elif loss_name == "mae":
            self.loss = WeightedLoss("L1Loss", reduction="none")
        elif loss_name == "afcrps":
            # self.loss = WeightedLoss("AFCRPS", reduction="none")
            self.loss = MultiScaleLoss("AFCRPS", reduction="none")
        elif loss_name == "ClassicCRPS":
            self.loss = MultiScaleLoss("ClassicCRPS", reduction="none")
            # self.loss = WeightedLoss("AFCRPS", reduction="none")
        else:
            raise TypeError(f"Unknown loss function: {loss_name}")
        self.loss.prepare(self, statics.interior_mask, dataset_info)

        exp_summary(self)

    def setup(self, stage=None):
        if self.logging_enabled:
            self.save_path = Path(self.trainer.logger.log_dir)
            max_pred_step = self.num_pred_steps_val_test - 1
            self.rmse_psd_plot_metric = MetricPSDVar(pred_step=max_pred_step)
            # self.psd_plot_metric = MetricPSDK(self.save_path, pred_step=max_pred_step)
            self.psd_plot_metric = MetricPSDKperLeadtimes(self.save_path, pred_step=max_pred_step)
            # self.downscale_psd_plot_metric = MetricPSDKdownscaling(self.save_path, pred_step=max_pred_step)
            self.acc_metric = MetricACC(self.dataset_info)
            self.spread_metric = MetricSpread(self.dataset_info, pred_step=max_pred_step+1)
            self.spread_map_metric = MetricMapSpread(self.dataset_info, pred_step=max_pred_step+1)
            self.bias_metric = MetricBias(self.dataset_info, pred_step=max_pred_step+1)
            self.error_metric = MetricSquaredError(self.dataset_info, pred_step=max_pred_step+1)
            self.unbVar_metric = MetricUnbiasedVariance(self.dataset_info, pred_step=max_pred_step+1)
            self.crps_metric = MetricCRPS(self.dataset_info, pred_step=max_pred_step+1)
            self.configure_loggers()

    def configure_loggers(self):
        layout = {
            "Check Overfit": {
                "loss": [
                    "Multiline",
                    ["mean_loss_epoch/train", "mean_loss_epoch/validation"],
                ],
            },
        }
        self.logger.experiment.add_custom_scalars(layout)

    @property
    def logging_enabled(self) -> bool:
        """
        Check if logging is enabled
        """
        return self.trainer.logger.log_dir is not None

    @property
    def dtype(self):
        """
        Return the appropriate torch dtype for the desired precision in hparams.
        """
        return str_to_dtype[self.trainer.precision]

    def print_summary_model(self):
        self.dataset_info.summary()
        print(f"Number of input_steps : {self.num_input_steps}")
        print(f"Number of pred_steps (training) : {self.num_pred_steps_train}")
        print(f"Number of pred_steps (test/val) : {self.num_pred_steps_val_test}")
        print(f"Number of intermediary steps :{self.num_inter_steps}")
        print(f"Training strategy :{self.training_strategy}")
        print(
            f"Model step duration : {self.dataset_info.pred_step /self.num_inter_steps}"
        )
        print(f"Model conf {self.model_conf}")
        print("---------------------")
        print(f"Loss {self.loss}")
        print(f"Batch size {self.batch_size}")
        print("---------------------------")

    @rank_zero_only
    def inspect_tensors(self):
        """
        Prints all tensor parameters and buffers
        of the model with name, shape and dtype.
        """
        # trainable parameters
        for name, param in self.named_parameters():
            print(name, param.shape, param.dtype)
        # buffers
        for name, buffer in self.named_buffers():
            print(name, buffer.shape, buffer.dtype)

    @rank_zero_only
    def log_hparams_tb(self):
        if self.logging_enabled and self.logger:
            # Save model & dataset conf as files
            if self.dataset_conf is not None:
                shutil.copyfile(self.dataset_conf, self.save_path / "dataset_conf.json")
            # Write commit and state of git repo in log file
            dest_git_log = self.save_path / "git_log.txt"
            out_log = (
                subprocess.check_output(["git", "log", "-n", "1"])
                .strip()
                .decode("utf-8")
            )
            out_status = (
                subprocess.check_output(["git", "status"]).strip().decode("utf-8")
            )
            with open(dest_git_log, "w") as f:
                f.write(out_log)
                f.write(out_status)

    def on_fit_start(self):
        self.log_hparams_tb()
        self.print_summary_model()

    def configure_optimizers(self):
        """
        Configure the optimizer and scheduler
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
        )

        # Scheduler
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            min_lr=self.hparams.min_learning_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward_noising(
        self, x_0: torch.Tensor, t: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        beta = torch.linspace(0.0001, 0.02, TIMESTEPS)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar = torch.cat((torch.tensor([1.], device=beta.device), alpha_bar[:-1]), dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        one_minus_sqrt_alpha_bar = torch.sqrt(1 - alpha_bar)
        
        noise = torch.randn_like(x_0)
        # indexing with t
        sqrt_alpha_bar_t = sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        one_minus_sqrt_alpha_bar_t = one_minus_sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        
        noisy_image = sqrt_alpha_bar_t * x_0 + one_minus_sqrt_alpha_bar_t * noise
        snr = alpha_bar[t] / (1 - alpha_bar[t])
        return noisy_image, noise, snr

    def compute_snr(
        self, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the SNR for the given timestep t.
        """
        beta = torch.linspace(0.0001, 0.02, TIMESTEPS)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar = torch.cat((torch.tensor([1.], device=beta.device), alpha_bar[:-1]), dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        one_minus_sqrt_alpha_bar = torch.sqrt(1 - alpha_bar)

        alpha_bar_t = alpha_bar[t]
        return alpha_bar_t / (1 - alpha_bar_t)
    

    def backward_denoising_ddim(
        self, x_t: torch.Tensor, pred_x: torch.Tensor, t: int, eta: float = 0.0
        ) -> torch.Tensor:

        t = torch.full((x_t.shape[0],1,1,1), t, dtype=torch.int64, device="cpu")
        beta = torch.linspace(0.0001, 0.02, TIMESTEPS)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        alpha_bar = torch.cat((torch.tensor([1.], device=beta.device), alpha_bar[:-1]), dim=0)
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        one_minus_sqrt_alpha_bar = torch.sqrt(1 - alpha_bar)

        alpha_bar_t = alpha_bar[t]
        alpha_t_minus_one = alpha[t - 1]

        sigma_t = torch.sqrt(beta[t] * (1 - alpha_bar[t - 1]) / (1 - alpha_bar_t)) * eta

        pred = torch.sqrt(alpha_t_minus_one) * pred_x
        pred_noise = (x_t - torch.sqrt(alpha_bar_t) * pred_x) / torch.sqrt(1 - alpha_bar_t)
        pred = pred + torch.sqrt(1 - alpha_t_minus_one - sigma_t ** 2) * pred_noise

        eps_t = torch.randn_like(x_t)
        pred = pred + sigma_t * eps_t

        return pred
    
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
        B, N_lat, N_lon, d_f = x.shape
        aro_inputs = x.view(B, N_lat, N_lon, d_f).permute(0, 3, 1, 2)

        g_1d = self._get_gaussian_1d(n, aro_inputs.device, aro_inputs.dtype)

        # Build separable kernels
        g_row = g_1d.view(1, 1, 1, -1).repeat(d_f, 1, 1, 1)
        g_col = g_1d.view(1, 1, -1, 1).repeat(d_f, 1, 1, 1)

        pad = g_1d.numel() // 2
        aro_inputs_padded = F.pad(aro_inputs, (pad, pad, pad, pad), mode="reflect")

        # Apply separable conv
        out = F.conv2d(aro_inputs_padded, g_row, groups=d_f)
        out = F.conv2d(out, g_col, groups=d_f)

        aro_outputs_tensor = out.permute(0, 2, 3, 1).view(B, N_lat, N_lon, d_f)
        return aro_outputs_tensor

    # def _next_x(
    #     self, batch: ItemBatch, prev_states: NamedTensor, step_idx: int, scale_factor: int = 1
    # ) -> torch.Tensor:
    #     """
    #     Build the next x input for the model at timestep step_idx using the :
    #     - previous states
    #     - forcing
    #     - static features
    #     """

    #     forcing = batch.forcing.select_dim("timestep", step_idx, bare_tensor=False)
    #     forcing_tensor = batch.forcing.select_dim("timestep", step_idx)

    #     last_smoothed = self.smooth(prev_states.select_dim("timestep", -1), n=scale_factor)

    #     x = torch.cat(
    #         [
    #             self.smooth(prev_states.select_dim("timestep", idx), n=scale_factor)
    #             for idx in range(batch.num_input_steps-1)   
    #         ]
    #         + [last_smoothed,]
    #         + [self.grid_static_features[: batch.batch_size], forcing_tensor],
    #         dim=forcing.dim_index("features"),
    #     )
                
    #     return x, last_smoothed

    # def _next_x(
    #     self, batch: ItemBatch, prev_states: NamedTensor, step_idx: int
    # ) -> torch.Tensor:
    #     """
    #     Build the next x input for the model at timestep step_idx using the :
    #     - previous states
    #     - forcing
    #     - static features
    #     """

    #     forcing = batch.forcing.select_dim("timestep", step_idx, bare_tensor=False)
    #     forcing_tensor = batch.forcing.select_dim("timestep", step_idx)

    #     last_smoothed = self.smooth(prev_states.select_dim("timestep", -1), n=10)

    #     x = torch.cat(
    #         [
    #             prev_states.select_dim("timestep", idx)
    #             for idx in range(batch.num_input_steps)   
    #         ]
    #         + [self.grid_static_features[: batch.batch_size], forcing_tensor],
    #         dim=forcing.dim_index("features"),
    #     )
                
    #     return x, last_smoothed

    # def _next_x(
    #     self, batch: ItemBatch, prev_states: NamedTensor, step_idx: int
    # ) -> torch.Tensor:
    #     """
    #     Build the next x input for the model at timestep step_idx using the :
    #     - previous states
    #     - forcing
    #     - static features
    #     """

    #     forcing = batch.forcing.select_dim("timestep", step_idx, bare_tensor=False)
    #     forcing_tensor = batch.forcing.select_dim("timestep", step_idx)

    #     x = torch.cat(
    #         [self.grid_static_features[: batch.batch_size], forcing_tensor],
    #         dim=forcing.dim_index("features"),
    #     )
                
    #     return x
    def _next_x(
        self, batch: ItemBatch, prev_states: NamedTensor, step_idx: int
    ) -> torch.Tensor:
        """
        Build the next x input for the model at timestep step_idx using the :
        - previous states
        - forcing
        - static features
        """

        forcing = batch.forcing.select_dim("timestep", step_idx, bare_tensor=False)
        forcing_tensor = batch.forcing.select_dim("timestep", step_idx)

        x = torch.cat(
            [prev_states.select_dim("timestep", idx)
             for idx in range(batch.num_input_steps)] 
            +
            [self.grid_static_features[: batch.batch_size], forcing_tensor],
            dim=forcing.dim_index("features"),
        )
                
        return x

    def _step_diffs(
        self, feature_names: List[str], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mean and std of the differences between two consecutive states on the desired device.
        """
        step_diff_std = self.diff_stats.to_list("std", feature_names).to(
            device,
            non_blocking=True,
        )
        step_diff_mean = self.diff_stats.to_list("mean", feature_names).to(
            device, non_blocking=True
        )
        return step_diff_std, step_diff_mean

    def _strategy_params(self) -> Tuple[bool, bool, int]:
        """
        Return the parameters for the desired strategy:
        - force_border
        - scale_y
        - num_inter_steps
        """
        force_border: bool = True if self.training_strategy == "scaled_ar" else False
        scale_y: bool = True if self.training_strategy == "scaled_ar" else False
        # raise if mismatch between strategy and num_inter_steps
        if self.training_strategy == "diff_ar":
            if self.num_inter_steps != 1:
                raise ValueError(
                    "Diff AR strategy requires exactly 1 intermediary step."
                )

        return force_border, scale_y, self.num_inter_steps

    def common_step(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        Handling autocast subtelty for mixed precision on GPU and CPU (only bf16 for the later).
        """
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=self.dtype):
                return self._common_step(batch, inference)
        else:
            if not inference and "bf16" in self.trainer.precision:
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    return self._common_step(batch, inference)
            else:
                return self._common_step(batch, inference)
    def common_step_FM_training(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        Handling autocast subtelty for mixed precision on GPU and CPU (only bf16 for the later).
        """
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=self.dtype):
                return self._common_step_FM_training(batch, inference)
        else:
            if not inference and "bf16" in self.trainer.precision:
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    return self._common_step_FM_training(batch, inference)
            else:
                return self._common_step_FM_training(batch, inference)
    def common_step_FM_sampling(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        Handling autocast subtelty for mixed precision on GPU and CPU (only bf16 for the later).
        """
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=self.dtype):
                return self._common_step_FM_sampling(batch, inference)
        else:
            if not inference and "bf16" in self.trainer.precision:
                with torch.amp.autocast("cuda", dtype=self.dtype):
                    return self._common_step_FM_sampling(batch, inference)
            else:
                return self._common_step_FM_sampling(batch, inference)
            
    def f10(self, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Logarithmic growth function with k=10,
        normalized to the range [0, 1).

        Parameters:
            n (torch.Tensor): Non-negative integer tensor of any shape.

        Returns:
            torch.Tensor: float32 tensor with values in [0, 1), same shape as n.
            torch.Tensor: int64 tensor with values in [1, 128), same shape as n.
        """
        k = 10.0
        scale = 31.0 #127.0  # original growth factor
        
        n = n.to(torch.float32)  # works elementwise across batch
        vals = 1 + scale * torch.log1p(n / k) / (1 + torch.log1p(n / k))

        ints = vals.to(torch.int64)           # [1, 128)
        floats = (ints / (scale+1)).to(torch.float32)  # [0, 1)

        return floats, ints

    def _common_step(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        Two Autoregressive strategies are implemented here for train, val, test and inference:
        - scaled_ar:
            * Boundary forcing with y_true/true_state
            * Scaled Differential update next_state = prev_state + y * std + mean
            * Intermediary steps for which we have no y_true data

        - diff_ar:
            * No Boundary forcing
            * Differential update next_state = prev_state + y
            * No Intermediary steps

        Derived/Inspired from https://github.com/joeloskarsson/neural-lam/

        In inference mode, we assume batch.outputs is None and we disable output based border forcing.
        """
        force_border, scale_y, num_inter_steps = self._strategy_params()
        # Right now we postpone that we have a single input/output/forcing

        self.original_shape = None

        if self.model.model_type == ModelType.GRAPH:
            # Stack original shape to reshape later
            self.original_shape = batch.inputs.tensor.shape
            # Graph model, we flatten the batch spatial dims
            batch.inputs.flatten_("ngrid", *batch.inputs.spatial_dim_idx)

            if not inference:
                batch.outputs.flatten_("ngrid", *batch.outputs.spatial_dim_idx)

            batch.forcing.flatten_("ngrid", *batch.forcing.spatial_dim_idx)

        prev_states = batch.inputs
        prediction_list = []

        # Here we do the autoregressive prediction looping
        # for the desired number of ar steps.
        for i in range(batch.num_pred_steps):
            if not inference:
                border_state = batch.outputs.select_dim("timestep", i)

            if scale_y:
                step_diff_std, step_diff_mean = self._step_diffs(
                    (
                        self.output_feature_names
                        if inference
                        else batch.outputs.feature_names
                    ),
                    prev_states.device,
                )
            
            # t = torch.tensor(2) #torch.randint(low=0, high=50, size=(1,), device=prev_states.device) # random t FOR TRAINING ONLY ! 
            # scale_factor_cond, scale_factor = self.f10(t)
            # scale_factor_cond = torch.ones(self.batch_size*self.noise_members, device=prev_states.device) * scale_factor_cond

            # Intermediary steps for which we have no y_true data
            # Should be greater or equal to 1 (otherwise nothing is done).
            for k in range(num_inter_steps):
                # x, last_smoothed = self._next_x(batch, prev_states, i, scale_factor=scale_factor) # aro_t smoothed according to scale factor
                # x, last_smoothed = self._next_x(batch, prev_states, i) # aro_t smoothed according to scale factor
                x = self._next_x(batch, prev_states, i) 
                if self.noise_strategy == "CondLayerNorm":
                    # generate (32,) noise vectors for stochastic conditional layer normalization
                    # skillful joint probabilistic weather forecasting from marginals: https://arxiv.org/pdf/2506.10772
                    epsilon = torch.randn(self.batch_size*self.noise_members, 4096, device=prev_states.device)

                # Graph (B, N_grid, d_f) or Conv (B, N_lat,N_lon d_f)
                if self.channels_last:
                    x = x.to(memory_format=torch.channels_last)

                # Here we adapt our tensors to the order of dimensions of CNNs and ViTs
                if self.model.features_second:
                    x = features_last_to_second(x)
                    if self.noise_strategy == "CondLayerNorm":
                        x = x.unsqueeze(1).expand(-1, self.noise_members, -1, -1, -1).reshape(self.batch_size*self.noise_members, *x.shape[1:])
                        y = self.model(x, cond_z=epsilon)
                    else:   
                        y = self.model(x)
                    y = features_second_to_last(y)
                else:
                    if self.noise_strategy == "CondLayerNorm":
                        x = x.unsqueeze(1).expand(-1, self.noise_members, -1, -1, -1).reshape(self.batch_size*self.noise_members, *x.shape[1:])
                        y = self.model(x, cond_z=epsilon)
                    else:
                        y = self.model(x)

                # We update the latest of our prev_states with the network output
                if scale_y:

                    last_prev_state = prev_states.select_dim("timestep", -1)
                    if self.noise_strategy in ["CondLayerNorm", "forcing"]:
                        last_prev_state = last_prev_state.unsqueeze(1).expand(-1, self.noise_members, -1, -1, -1).reshape(self.batch_size*self.noise_members, *last_prev_state.shape[1:])
                    
                    # last_prev_state = last_smoothed
                    # if self.noise_strategy in ["CondLayerNorm", "forcing"]:
                    #     last_prev_state = last_prev_state.unsqueeze(1).expand(-1, self.noise_members, -1, -1, -1).reshape(self.batch_size*self.noise_members, *last_prev_state.shape[1:])


                    predicted_state = (
                        # select the last timestep
                        last_prev_state
                        + y * step_diff_std
                        + step_diff_mean
                    )

                    # predicted_state = y
                else:
                    predicted_state = prev_states.select_dim("timestep", -1) + y

                # Overwrite border with true state
                # Force it to true state for all intermediary step
                if not inference and force_border:
                    new_state = (
                        self.border_mask.expand_as(predicted_state) * border_state.repeat(self.noise_members, 1, 1, 1)
                        + self.interior_mask.expand_as(predicted_state) * predicted_state
                    )
                else:
                    new_state = predicted_state

                # Only update the prev_states if we are not at the last step
                if i < batch.num_pred_steps - 1 or k < num_inter_steps - 1:
                    # Update input states for next iteration: drop oldest, append new_state
                    timestep_dim_index = batch.inputs.dim_index("timestep")
                    new_prev_states_tensor = torch.cat(
                        [
                            # Drop the oldest timestep (select all but the first)
                            prev_states.index_select_dim(
                                "timestep",
                                range(1, prev_states.dim_size("timestep")),
                            ),
                            # Add the timestep dimension to the new state
                            new_state.unsqueeze(timestep_dim_index),
                        ],
                        dim=timestep_dim_index,
                    )

                    # Make a new NamedTensor with the same dim and
                    # feature names as the original prev_states
                    prev_states = NamedTensor.new_like(
                        new_prev_states_tensor, prev_states
                    )
            # Append prediction to prediction list only "normal steps"
            prediction_list.append(new_state)

        prediction = torch.stack(
            prediction_list, dim=1
        )  # Stacking is done on time step. (B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)

        # In inference mode we use a "trained" module which MUST have the output feature names
        # and the output dim names attributes set.
        if inference:
            pred_out = NamedTensor(
                prediction.type(self.output_dtype),
                self.output_dim_names,
                self.output_feature_names,
            )
        else:
            pred_out = NamedTensor.new_like(
                prediction.type_as(batch.outputs.tensor), batch.outputs
            )

        target_tensor = batch.outputs.tensor
        target_shifted = torch.zeros_like(target_tensor)
        target_shifted[:, 1:, :, :, 2] = target_tensor[:, :-1, :, :, 2]  # shift everything forward by one step, for precipitation only

        corrected_tensor = target_tensor - target_shifted
        corrected_target = NamedTensor.new_like(
            corrected_tensor.type_as(batch.outputs.tensor), batch.outputs
        )

        return pred_out, corrected_target # batch.outputs # aro_outputs_upsampled_nt

    def _common_step_FM_training(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        We extend here the _common_step diff_ar strategy, which is basically one step flow matching, 
        to multi step flow matching combined with shortcut models. We followed training implementation of Frans et al.
        ONE STEP DIFFUSION VIA SHORTCUT MODELS https://arxiv.org/pdf/2410.12557
        """
            
        force_border, scale_y, num_inter_steps = self._strategy_params()
        # Right now we postpone that we have a single input/output/forcing

        self.original_shape = None

        if self.model.model_type == ModelType.GRAPH:
            # Stack original shape to reshape later
            self.original_shape = batch.inputs.tensor.shape
            # Graph model, we flatten the batch spatial dims
            batch.inputs.flatten_("ngrid", *batch.inputs.spatial_dim_idx)

            if not inference:
                batch.outputs.flatten_("ngrid", *batch.outputs.spatial_dim_idx)

            batch.forcing.flatten_("ngrid", *batch.forcing.spatial_dim_idx)

        prev_states = batch.inputs
        prediction_list = []

        # Draw t and dt and epsilon ~ N(0,I)
        # t = torch.rand(self.batch_size, device=prev_states.device)
        # dt = 1/(2**torch.floor(torch.rand(self.batch_size, device=prev_states.device)*6))
        epsilon = torch.randn_like(prev_states.select_dim("timestep", 0))

        border_state = batch.outputs.select_dim("timestep", 0)

        # Graph (B, N_grid, d_f) or Conv (B, N_lat,N_lon d_f)
        if self.channels_last:
            x_t = x_t.to(memory_format=torch.channels_last)

        ########################################################################################################
        # draw a random variable in [0,1] and verify it has value < 0.75, 3/4 of the time -> velocity matching #
        ########################################################################################################
        if torch.rand(1).item() < 1: # 0.9

            t = torch.randint(low=0, high=64, size=(self.batch_size,), device=prev_states.device)
            t = t.float() / 64 # Normalize t [0,1] range
            dt = torch.zeros(self.batch_size, device=prev_states.device) # torch.ones(self.batch_size, device=prev_states.device) * (1 / 64)  # dt = 1/64 for flow matching target

            x_t = self._next_x(batch, prev_states, 0, t=t, ft=epsilon)

            # Here we adapt our tensors to the order of dimensions of CNNs and ViTs
            if self.model.features_second:
                x_t = features_last_to_second(x_t)
                y = self.model(x_t, t=t, dt=dt)
                y = features_second_to_last(y)
            else:
                y = self.model(x_t, t=t, dt=dt)

            # match velocity y = x_1-x_0
            if self.dynamical_downscaling:
                num_dynamic_features = batch.inputs.dim_size("features")
                forcing_tensor = batch.forcing.select_dim("timestep", -1)
                forcing_tobedownscaled = forcing_tensor[...,:num_dynamic_features]
                if self.noise_strategy == "ddim":
                    predicted_state = forcing_tobedownscaled + epsilon + y # arp_t+1 + epsilon + y
                else:
                #     step_diff_std, step_diff_mean = self._step_diffs(
                #     (
                #         self.output_feature_names
                #         if inference
                #         else batch.outputs.feature_names
                #     ),
                #     prev_states.device,
                # )
                #     predicted_state = (
                #         forcing_tobedownscaled
                #         + ((1 - 1e-5)*epsilon + y) * step_diff_std
                #         + step_diff_mean
                #     )
                    # # Get shape
                    # B, N_lat, N_lon, d_f = prev_states.select_dim("timestep", -1).shape

                    # # Select timestep 0 and keep on GPU
                    # aro_t = prev_states.select_dim("timestep", -1)  # shape: (B, N_lat, N_lon, d_f)

                    # # Rearrange to (B, d_f, N_lat, N_lon) for interpolation
                    # aro_t = aro_t.permute(0, 3, 1, 2)  # (B, d_f, H, W)

                    # # Compute low-res dimensions
                    # N_lat_low = N_lat // 4
                    # N_lon_low = N_lon // 4

                    # # Downsample
                    # aro_t_lowres = torch.nn.functional.interpolate(
                    #     aro_t, size=(N_lat_low, N_lon_low), mode='bilinear', align_corners=False
                    # )

                    # # Upsample back to original resolution
                    # aro_t_upsampled = torch.nn.functional.interpolate(
                    #     aro_t_lowres, size=(N_lat, N_lon), mode='bilinear', align_corners=False
                    # )

                    # # Rearrange back to (B, N_lat, N_lon, d_f)
                    # aro_t_upsampled_tensor = aro_t_upsampled.permute(0, 2, 3, 1)  # shape: (B, N_lat, N_lon, d_f)
                    
                    predicted_state = (
                        (1 - 1e-5)*prev_states.select_dim("timestep", -1) + y
                    )
            else:
                predicted_state = prev_states.select_dim("timestep", -1) + y # aro_t + y

            # Overwrite border with true state
            # Force it to true state for all intermediary step
            if not inference and force_border:
                prediction = (
                    self.border_mask * border_state
                    + self.interior_mask * predicted_state
                )
            else:
                prediction = predicted_state

            # no A-R loop, requires additionnal dim for timesteps
            prediction = prediction.unsqueeze(1)
            #(B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)

            pred_out = NamedTensor.new_like(
            prediction.type_as(batch.outputs.tensor), batch.outputs
            )
            return pred_out, batch.outputs

        ######################################################################################################
        # Else if random variable has value > 0.75, 1/4 of the time, consistency loss term is backpropagated #
        ######################################################################################################
        else:
            # Sample dt_base ∈ {1, 2, 3, 4, 6}
            dt_base = torch.randint(low=1, high=7, size=(self.batch_size,), device=prev_states.device)
            dt_sections = torch.pow(2, dt_base)
            # dt_sections ∈ {1, 2, 4, 8, 16, 32, 64}
            dt = 1.0 / dt_sections
            # dt = 1 / 2^dt_base ∈ {1, 1/2, ..., 1/64}
            t = (torch.rand(self.batch_size, device=prev_states.device) * (dt_sections-1)).floor().long()
            t = t / dt_sections  # Normalize t [0,1] range

            x_t = self._next_x(batch, prev_states, 0, t=t, ft=epsilon)

            # Here we adapt our tensors to the order of dimensions of CNNs and ViTs
            if self.model.features_second:
                x_t = features_last_to_second(x_t)
                with torch.no_grad():
                    s_t = self.model(x_t, t=t, dt=dt)
                    s_t = features_second_to_last(s_t)
                    x_td = self._next_x(batch, prev_states, 0, t=t, dt=dt, st=s_t, ft=epsilon)
                    x_td = features_last_to_second(x_td)
                    s_td = self.model(x_td, t=t+dt, dt=dt)
                    s_td = features_second_to_last(s_td)

                s_t_2d = self.model(x_t, t=t, dt=2*dt)
                s_t_2d = features_second_to_last(s_t_2d)
                
            else:
                with torch.no_grad():
                    s_t = self.model(x_t, t=t, dt=dt)
                    x_td = self._next_x(batch, prev_states, 0, t=t, dt=dt, st=s_t, ft=epsilon)
                    s_td = self.model(x_td, t=t+dt, dt=dt)

                s_t_2d = self.model(x_t, t=t, dt=2*dt)

            # self-consistency target
            # no A-R loop, requires additionnal dim for timesteps
            v_target = torch.clip((s_t+s_td)/2, -4, 4)
            target = torch.unsqueeze(v_target, 1)
            pred_out = torch.unsqueeze(s_t_2d, 1)
            #(B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)

            target = NamedTensor.new_like(
                target.type_as(batch.outputs.tensor), batch.outputs
            )
            pred_out = NamedTensor.new_like(
                pred_out.type_as(batch.outputs.tensor), batch.outputs
            )
            return pred_out, target

    def _common_step_FM_sampling(
        self, batch: ItemBatch, inference: bool = False
    ) -> Tuple[NamedTensor, NamedTensor]:
        """
        We extend here the _common_step diff_ar strategy, which is basically one step flow matching, 
        to multi step flow matching combined with shortcut models. We followed sampling implementation of Frans et al.
        ONE STEP DIFFUSION VIA SHORTCUT MODELS https://arxiv.org/pdf/2410.12557
        """
        force_border, scale_y, num_inter_steps = self._strategy_params()
        # Right now we postpone that we have a single input/output/forcing

        self.original_shape = None

        if self.model.model_type == ModelType.GRAPH:
            # Stack original shape to reshape later
            self.original_shape = batch.inputs.tensor.shape
            # Graph model, we flatten the batch spatial dims
            batch.inputs.flatten_("ngrid", *batch.inputs.spatial_dim_idx)

            if not inference:
                batch.outputs.flatten_("ngrid", *batch.outputs.spatial_dim_idx)

            batch.forcing.flatten_("ngrid", *batch.forcing.spatial_dim_idx)

        prev_states = batch.inputs
        prediction_list = []

        num_dynamic_features = batch.inputs.dim_size("features")
        
        # Here we do the autoregressive prediction looping
        # for the desired number of ar steps.
        for i in range(batch.num_pred_steps):
            if not inference:
                border_state = batch.outputs.select_dim("timestep", i)

            # Intermediary steps for which we have no y_true data
            # Should be greater or equal to 1 (otherwise nothing is done).
            for k in range(num_inter_steps):
                dt = torch.tensor(1/self.shortcut_length, device=prev_states.device)*torch.ones(self.batch_size, device=prev_states.device)
                t = torch.zeros(self.batch_size, device=prev_states.device)
                x = self._next_x(batch, prev_states, i)

                # Graph (B, N_grid, d_f) or Conv (B, N_lat,N_lon d_f)
                if self.channels_last:
                    x = x.to(memory_format=torch.channels_last)

                for n in range(self.shortcut_length):
                    dt = torch.zeros(
                    self.batch_size, device=prev_states.device)
                    # Here we adapt our tensors to the order of dimensions of CNNs and ViTs
                    if self.model.features_second:
                        x = features_last_to_second(x)
                        s_t = self.model(x, t=t, dt=dt)
                        s_t = features_second_to_last(s_t)
                        x = features_second_to_last(x)
                    else:
                        s_t = self.model(x, t=t, dt=dt)
                    
                    # remove forcings + static features 
                    # retrieves the number of features of batch.inputs
                    x = x[...,:num_dynamic_features] # flow matching dynamic state x(t) always at beginning of the tensor
                    
                    if self.dynamical_downscaling:
                        dt = torch.tensor(1/self.shortcut_length, device=prev_states.device)*torch.ones(self.batch_size, device=prev_states.device)
                        x = self._next_x(batch, prev_states, i, t=None, dt=dt, st=s_t, ft=x) 
                    else:
                        # requires timestep dimension to match prev_states dimensions
                        x = x.unsqueeze(1)
                        x = NamedTensor.new_like(x.type_as(prev_states.tensor), prev_states)
                        x = self._next_x(batch, x, i, t=None, dt=dt, st=s_t)

                    t = t + dt
                
                    
                # remove forcings + static features from
                if self.noise_strategy == "ddim":
                    predicted_state = x[...,:num_dynamic_features] + batch.forcing.select_dim("timestep", i)[...,:num_dynamic_features] # pred residual + arp_t+1
                else:
                #     step_diff_std, step_diff_mean = self._step_diffs(
                #     (
                #         self.output_feature_names
                #         if inference
                #         else batch.outputs.feature_names
                #     ),
                #     prev_states.device,
                # )
                #     predicted_state = (
                #         batch.forcing.select_dim("timestep", i)[...,:num_dynamic_features]
                #         + x[...,:num_dynamic_features] * step_diff_std
                #         + step_diff_mean
                #     )
                    predicted_state = (
                        x[...,:num_dynamic_features]
                    )

                # Overwrite border with true state
                # Force it to true state for all intermediary step
                if not inference and force_border:
                    new_state = (
                        self.border_mask * border_state
                        + self.interior_mask * predicted_state
                    )
                else:
                    new_state = predicted_state

                # Only update the prev_states if we are not at the last step
                if i < batch.num_pred_steps - 1 or k < num_inter_steps - 1:
                    # Update input states for next iteration: drop oldest, append new_state
                    timestep_dim_index = batch.inputs.dim_index("timestep")
                    new_prev_states_tensor = torch.cat(
                        [
                            # Drop the oldest timestep (select all but the first)
                            prev_states.index_select_dim(
                                "timestep",
                                range(1, prev_states.dim_size("timestep")),
                            ),
                            # Add the timestep dimension to the new state
                            new_state.unsqueeze(timestep_dim_index),
                        ],
                        dim=timestep_dim_index,
                    )

                    # Make a new NamedTensor with the same dim and
                    # feature names as the original prev_states
                    prev_states = NamedTensor.new_like(
                        new_prev_states_tensor, prev_states
                    )
            # Append prediction to prediction list only "normal steps"
            prediction_list.append(new_state)

        prediction = torch.stack(
            prediction_list, dim=1
        )  # Stacking is done on time step. (B, pred_steps, N_grid, d_f) or (B, pred_steps, N_lat, N_lon, d_f)

        # In inference mode we use a "trained" module which MUST have the output feature names
        # and the output dim names attributes set.
        if inference:
            pred_out = NamedTensor(
                prediction.type(self.output_dtype),
                self.output_dim_names,
                self.output_feature_names,
            )
        else:
            pred_out = NamedTensor.new_like(
                prediction.type_as(batch.outputs.tensor), batch.outputs
            )

            lowres_tensor = batch.forcing.tensor[...,:num_dynamic_features]
            lowres = NamedTensor.new_like(
                lowres_tensor.type_as(batch.outputs.tensor), batch.outputs
            )
        return pred_out, batch.outputs, lowres

    def on_train_start(self):
        self.train_plotters = []

    def _shared_epoch_end(self, outputs: List[torch.Tensor], label: str) -> None:
        """Computes and logs the averaged metrics at the end of an epoch.
        Step shared by training and validation epochs.
        """
        if self.logging_enabled:
            avg_loss = torch.stack([x for x in outputs]).mean()
            tb = self.logger.experiment
            tb.add_scalar(f"mean_loss_epoch/{label}", avg_loss, self.current_epoch)
        
    def training_step(self, batch: ItemBatch, batch_idx: int) -> torch.Tensor:
        """
        Train on single batch
        """
        # we save the feature names at the first batch
        # to check at inference time if the feature names are the same
        # also useful to build NamedTensor outputs with same feature and dim names
        if batch_idx == 0:
            self.input_feature_names = batch.inputs.feature_names
            self.output_feature_names = batch.outputs.feature_names
            self.output_dim_names = batch.outputs.names
            self.output_dtype = batch.outputs.tensor.dtype

        if self.shortcut_length is not None:
            prediction, target = self.common_step_FM_training(batch)
            batch_loss = torch.mean(self.loss(prediction, target, epoch=self.current_epoch))
        else:
            prediction, target = self.common_step(batch)
            batch_loss = torch.mean(self.loss(prediction, target, epoch=self.current_epoch))

        self.training_step_losses.append(batch_loss)

        # Notify every plotters
        if self.logging_enabled:
            for plotter in self.train_plotters:
                plotter.update(self, prediction=self.prediction, target=self.target)

        return batch_loss

    def on_save_checkpoint(self, checkpoint):
        """
        We store our feature and dim names in the checkpoint
        """
        checkpoint["input_feature_names"] = self.input_feature_names
        checkpoint["output_feature_names"] = self.output_feature_names
        checkpoint["output_dim_names"] = self.output_dim_names
        checkpoint["output_dtype"] = self.output_dtype

    def on_load_checkpoint(self, checkpoint):
        """
        We load our feature and dim names from the checkpoint
        """
        self.input_feature_names = checkpoint["input_feature_names"]
        self.output_feature_names = checkpoint["output_feature_names"]
        self.output_dim_names = checkpoint["output_dim_names"]
        self.output_dtype = checkpoint["output_dtype"]

    def predict_step(self, batch: ItemBatch, batch_idx: int) -> torch.Tensor:
        """
        Check if the feature names are the same as the one used during training
        and make a prediction and accumulate if io_conf =/= none.
        """
        if batch_idx == 0:
            if self.input_feature_names != batch.inputs.feature_names:
                raise ValueError(
                    f"Input Feature names mismatch between training and inference. "
                    f"Training: {self.input_feature_names}, Inference: {batch.inputs.feature_names}"
                )
        preds = self.forward(batch)
        if not (self.io_conf is None):
            self.grib_writing(preds)
        return preds

    def grib_writing(self, preds):
        with open(self.io_conf, "r") as f:
            save_settings = GribSavingSettings(**json.load(f))
            ph = len(save_settings.output_fmt.split("{}")) - 1
            kw = len(save_settings.output_kwargs)
            fi = len(save_settings.sample_identifiers)
            if ph != (fi + kw):
                raise ValueError(
                    f"Filename fmt has {ph} placeholders,\
                    but {kw} output_kwargs and {fi} sample identifiers."
                )
        for sample, pred in zip(self.infer_ds.sample_list, preds):
            save_named_tensors_to_grib(pred, self.infer_ds, sample, save_settings)

    def forward(self, x: ItemBatch) -> NamedTensor:
        """
        Forward pass of the model
        """
        if self.shortcut_length is not None:
            return self.common_step_FM_sampling(x, inference=True)[0]
        else:
            return self.common_step_(x, inference=True)[0]
        

    def on_train_epoch_end(self):
        outputs = self.training_step_losses
        self._shared_epoch_end(outputs, "train")
        self.training_step_losses.clear()  # free memory

    def on_train_end(self):
        if self.mlflow_logger:
            # Get random sample to infer the signature of the model
            dataloader = self.trainer.datamodule.test_dataloader()
            data = next(iter(dataloader))
            signature = infer_signature(
                data.inputs.tensor.detach().numpy(),
                data.outputs.tensor.detach().numpy(),
            )

            # Manually log the trained model in Mlflow style with its signature
            run_id = self.mlflow_logger.version
            with mlflow.start_run(run_id=run_id):
                mlflow.pytorch.log_model(
                    pytorch_model=self.trainer.model,
                    artifact_path="model",
                    signature=signature,
                )

    def on_validation_start(self):
        """
        Add some plots when starting validation
        """
        if self.logging_enabled:
            l1_loss = ScaledLoss("L1Loss", reduction="none")
            l1_loss.prepare(self, self.interior_mask, self.dataset_info)
            metrics = {"mae": l1_loss}

            self.valid_plotters = [
                PredictionTimestepPlot(
                    num_samples_to_plot=1,
                    num_features_to_plot=8,
                    prefix="Validation",
                    save_path=self.save_path,
                ),
                PredictionEpochPlot(
                    num_samples_to_plot=1,
                    num_features_to_plot=8,
                    prefix="Validation",
                    save_path=self.save_path,
                ),
            ]

    def _shared_val_test_step(self, batch: ItemBatch, batch_idx, label: str):
        with torch.no_grad():
            
            if self.shortcut_length is not None:
                loss_prediction, loss_target = self.common_step_FM_training(batch)
                prediction, target, lowres = self.common_step_FM_sampling(batch)
                time_step_loss = torch.mean(
                    self.loss(loss_prediction, loss_target, epoch=self.current_epoch),
                    dim=0,
                )
            else:
                prediction, target = self.common_step(batch)

                # # select mean of members 
                # prediction_tensor = prediction.tensor.reshape(self.batch_size, self.noise_members, *prediction.tensor.shape[1:])
                # member = torch.mean(prediction_tensor, dim=1)
                # preds = NamedTensor.new_like(member.type_as(prediction.tensor), prediction)

                lowres = None
                time_step_loss = torch.mean(self.loss(prediction, target, epoch=self.current_epoch), dim=0)

        mean_loss = torch.mean(time_step_loss)

        if self.logging_enabled:
            # Log loss per timestep
            loss_dict = {
                f"timestep_losses/{label}_step_{step}": time_step_loss[step]
                for step in range(time_step_loss.shape[0])
            }
            self.log_dict(loss_dict, on_epoch=True, sync_dist=True)
            self.log(
                f"{label}_mean_loss",
                mean_loss,
                on_epoch=True,
                sync_dist=True,
                prog_bar=(label == "val"),
            )
        return prediction, target, lowres, mean_loss

    def validation_step(self, batch: ItemBatch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, lowres, mean_loss = self._shared_val_test_step(
            batch, batch_idx, "val"
        )
        self.validation_step_losses.append(mean_loss)

        self.val_mean_loss = mean_loss

        if self.noise_members > 1:
            # select random member for preds
            prediction_tensor = prediction.tensor.reshape(self.batch_size, self.noise_members, *prediction.tensor.shape[1:])
            member = prediction_tensor[:, torch.randint(0, self.noise_members, (1,)).item()]
            prediction = NamedTensor.new_like(member.type_as(prediction.tensor), prediction)

            # select mean of members 
            # prediction_tensor = prediction.tensor.reshape(self.batch_size, self.noise_members, *prediction.tensor.shape[1:])
            # member = torch.mean(prediction_tensor, dim=1)
            # prediction = NamedTensor.new_like(member.type_as(prediction.tensor), prediction)

        if self.logging_enabled:
            # Notify every plotters
            if self.current_epoch % PLOT_PERIOD == 0:
                for plotter in self.valid_plotters:
                    if lowres is not None and hasattr(plotter, "requires_lowres"):
                        plotter.update(self, prediction=prediction, target=target, lowres=lowres)
                    else:
                        plotter.update(self, prediction=prediction, target=target)
            if self.dynamical_downscaling:
                self.downscale_psd_plot_metric.update(prediction, target, lowres, self.original_shape)
            else:
                self.psd_plot_metric.update(prediction, target, self.original_shape)
            self.rmse_psd_plot_metric.update(prediction, target, self.original_shape)
            self.acc_metric.update(prediction, target)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """

        if self.logging_enabled:
            # Get dict of metrics' results
            dict_metrics = dict()
            if self.dynamical_downscaling:
                dict_metrics.update(self.downscale_psd_plot_metric.compute())
            else:
                dict_metrics.update(self.psd_plot_metric.compute())
            dict_metrics.update(self.rmse_psd_plot_metric.compute())
            dict_metrics.update(self.acc_metric.compute())
            for name, elmnt in dict_metrics.items():
                if isinstance(elmnt, matplotlib.figure.Figure):
                    # Tensorboard logger
                    self.logger.experiment.add_figure(
                        f"{name}", elmnt, self.current_epoch
                    )
                    # If MLFlowLogger activated
                    if self.mlflow_logger:
                        run_id = self.mlflow_logger.version
                        self.mlflow_logger.experiment.log_figure(
                            run_id=run_id,
                            figure=elmnt,
                            artifact_file=f"figures/{name}.png",
                        )
                elif isinstance(elmnt, torch.Tensor):
                    self.log_dict(
                        {name: elmnt},
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

        outputs = self.validation_step_losses
        self._shared_epoch_end(outputs, "validation")

        # free memory
        self.validation_step_losses.clear()

        if self.logging_enabled:
            # Notify every plotters
            if self.current_epoch % PLOT_PERIOD == 0:
                for plotter in self.valid_plotters:
                    plotter.on_step_end(self, label="Valid")

    def on_test_start(self):
        """
        Attach observer when starting test
        """
        if self.logging_enabled:
            metrics = {}
            for torch_loss, alias in ("L1Loss", "mae"), ("MSELoss", "rmse"):
                loss = ScaledLoss(torch_loss, reduction="none")
                loss.prepare(self, self.interior_mask, self.dataset_info)
                metrics[alias] = loss

            if self.ensemble_metrics:
                metrics["std"] = self.spread_metric

            self.test_plotters = [
                StateErrorPlot(metrics, save_path=self.save_path),
                SpatialErrorPlot(),
                PredictionTimestepPlot(
                    num_samples_to_plot=self.num_samples_to_plot,
                    num_features_to_plot=6,
                    prefix="Test",
                    save_path=self.save_path,
                )
            ]

            if self.ensemble_metrics:
                self.test_plotters.append(SpreadTimestepPlot(
                    metric=self.spread_map_metric,
                    dataset_name = self.dataset_info.name,
                    num_samples_to_plot=self.num_samples_to_plot,
                    num_features_to_plot=6,
                    prefix="Test",
                    save_path=self.save_path,
                ))
                self.test_plotters.append(EnScorePlot(
                    bias_metric=self.bias_metric,
                    error_metric=self.error_metric,
                    unbVar_metric=self.unbVar_metric,
                    crps_metric=self.crps_metric,
                    dataset_name = self.dataset_info.name,
                    num_samples_to_plot=31,
                    num_features_to_plot=6,
                    prefix="Test",
                    save_path=self.save_path,
                    num_pred_steps_test=self.num_pred_steps_val_test,
                ))

    def test_step(self, batch: ItemBatch, batch_idx):
        """
        Run test on single batch
        """
        prediction, target, lowres, _ = self._shared_val_test_step(batch, batch_idx, "test")

        # if self.noise_members > 1:
        #     # select random member for preds
        #     # prediction_tensor = prediction.tensor.reshape(self.batch_size, self.noise_members, *prediction.tensor.shape[1:])
        #     # member = prediction_tensor[:, torch.randint(0, self.noise_members, (1,)).item()]
        #     # prediction = NamedTensor.new_like(member.type_as(prediction.tensor), prediction)

        #     # select mean of members 
        #     prediction_tensor = prediction.tensor.reshape(self.batch_size, self.noise_members, *prediction.tensor.shape[1:])
        #     member = torch.mean(prediction_tensor, dim=1)
        #     prediction = NamedTensor.new_like(member.type_as(prediction.tensor), prediction)

        if self.logging_enabled:
            
            if "full" in self.features_to_plot:

                # If we want to plot all features, we do not slice the prediction and target
                sliced_prediction = prediction
                sliced_target = target
                sliced_lowres = lowres if lowres is not None else None
            else:

                desired_indices = [prediction.feature_names_to_idx[feature] for feature in self.features_to_plot]
                # Use index_select_dim to create a new NamedTensor with only the desired features
                sliced_prediction = NamedTensor(
                    tensor=prediction.index_select_dim(prediction.feature_dim_name, torch.tensor(desired_indices), bare_tensor=True),
                    names=prediction.names.copy(),
                    feature_names=self.features_to_plot,
                    feature_dim_name=prediction.feature_dim_name
                )
                sliced_target = NamedTensor(
                    tensor=target.index_select_dim(target.feature_dim_name, torch.tensor(desired_indices), bare_tensor=True),
                    names=target.names.copy(),
                    feature_names=self.features_to_plot,
                    feature_dim_name=target.feature_dim_name
                )

                if lowres is not None:
                    sliced_lowres = NamedTensor(
                        tensor=lowres.index_select_dim(lowres.feature_dim_name, torch.tensor(desired_indices), bare_tensor=True),
                        names=lowres.names.copy(),
                        feature_names=self.features_to_plot,
                        feature_dim_name=lowres.feature_dim_name
                    )

            # Notify plotters & metrics
            for plotter in self.test_plotters:
                if lowres is not None and hasattr(plotter, "requires_lowres"):
                    plotter.update(self, prediction=sliced_prediction, target=sliced_target, lowres=sliced_lowres)
                else:
                    plotter.update(self, prediction=sliced_prediction, target=sliced_target)

            self.acc_metric.update(prediction, target)
            if self.dynamical_downscaling:
                self.downscale_psd_plot_metric.update(prediction, target, lowres, self.original_shape)
            else:
                # select random member for preds
                prediction_tensor = prediction.tensor.reshape(self.batch_size, self.noise_members, *prediction.tensor.shape[1:])
                member_tensor = prediction_tensor[:, torch.randint(0, self.noise_members, (1,)).item()]
                member = NamedTensor.new_like(member_tensor.type_as(prediction.tensor), prediction)

                # select mean of members 
                prediction_tensor = prediction.tensor.reshape(self.batch_size, self.noise_members, *prediction.tensor.shape[1:])
                mean_tensor = torch.mean(prediction_tensor, dim=1)
                mean = NamedTensor.new_like(mean_tensor.type_as(prediction.tensor), prediction)

                self.psd_plot_metric.update(mean, member, target, self.original_shape)
                
            self.rmse_psd_plot_metric.update(prediction, target, self.original_shape)

    @cached_property
    def interior_2d(self) -> torch.Tensor:
        """
        Get the interior mask as a 2d mask.
        Usefull when stored as 1D in statics.
        """
        if self.num_spatial_dims == 1:
            return einops.rearrange(
                self.interior_mask, "(x y) h -> x y h", x=self.grid_shape[0]
            )
        return self.interior_mask

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        """
        if self.logging_enabled:
            dict_metrics = {}
            if self.dynamical_downscaling:
                dict_metrics.update(self.downscale_psd_plot_metric.compute(prefix="test"))
            else:
                dict_metrics.update(self.psd_plot_metric.compute(prefix="test"))
            dict_metrics.update(self.rmse_psd_plot_metric.compute(prefix="test"))
            dict_metrics.update(self.acc_metric.compute(prefix="test"))

            for name, elmnt in dict_metrics.items():
                if isinstance(elmnt, matplotlib.figure.Figure):
                    # Tensorboard logger
                    self.logger.experiment.add_figure(
                        f"{name}", elmnt, self.current_epoch
                    )
                    # If MLFlowLogger activated
                    if self.mlflow_logger:
                        run_id = self.mlflow_logger.version
                        self.mlflow_logger.experiment.log_figure(
                            run_id=run_id,
                            figure=elmnt,
                            artifact_file=f"figures/{name}.png",
                        )
                elif isinstance(elmnt, torch.Tensor):
                    self.log_dict(
                        {name: elmnt},
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )

            # Notify plotters that the test epoch end
            for plotter in self.test_plotters:
                plotter.on_step_end(self, label="Test")

    @cached_property
    def mlflow_logger(self) -> Union[MLFlowLogger, None]:
        """
        Get the MLFlowLogger if it has been set.
        """
        return next(
            iter([o for o in self.loggers if isinstance(o, MLFlowLogger)]), None
        )


@rank_zero_only
def exp_summary(model: AutoRegressiveLightning):
    model.print_summary_model()
    summary(model.model)
