from typing import Literal

import torch
import numpy as np
from tqdm import tqdm
from py4cast.datasets.base import DatasetInfo, ItemBatch, NamedTensor, Statics
from py4cast.datasets.base import DatasetABC
from py4cast.utils import torch_save
from pathlib import Path
import torch.nn.functional as F
import math


def compute_mean_std_min_max(
    dataset: DatasetABC, type_tensor: Literal["inputs", "outputs", "forcing"]
):
    """
    Compute mean and standard deviation for this dataset.
    """
    random_batch = next(iter(dataset.torch_dataloader()))
    named_tensor = getattr(random_batch, type_tensor)
    n_features = len(named_tensor.feature_names)
    sum_means = torch.zeros(n_features)
    sum_squares = torch.zeros(n_features)
    ndim_features = len(named_tensor.tensor.shape) - 1
    flat_input = named_tensor.tensor.flatten(0, ndim_features - 1)  # (X, Features)
    best_min = torch.min(flat_input, dim=0).values
    best_max = torch.max(flat_input, dim=0).values

    if torch.isnan(best_min).any() or torch.isnan(best_max).any(): 
        raise ValueError(
            "Your dataset contain NaN values, which prevent the calculation of statistics."
        )

    counter = 0
    if dataset.settings.standardize:
        raise ValueError("Your dataset should not be standardized.")

    for batch in tqdm(
        dataset.torch_dataloader(),
        desc=f"Computing {type_tensor} stats",
    ):
        tensor = getattr(batch, type_tensor).tensor
        tensor = tensor.flatten(1, 3)  # Flatten to be (Batch, X, Features)
        counter += tensor.shape[0]  # += batch size

        sum_means += torch.sum(tensor.mean(dim=1), dim=0)  # (d_features)
        sum_squares += torch.sum((tensor**2).mean(dim=1), dim=0)  # (d_features)

        mini = torch.min(tensor, 1).values[0]
        stack_mini = torch.stack([best_min, mini], dim=0)
        best_min = torch.min(stack_mini, dim=0).values  # (d_features)

        maxi = torch.max(tensor, 1).values[0]
        stack_maxi = torch.stack([best_max, maxi], dim=0)
        best_max = torch.max(stack_maxi, dim=0).values  # (d_features)

    mean = sum_means / counter
    second_moment = sum_squares / counter
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    stats = {}
    for i, name in enumerate(named_tensor.feature_names):
        stats[name] = {
            "mean": mean[i],
            "std": std[i],
            "min": best_min[i],
            "max": best_max[i],
        }
    return stats


def compute_parameters_stats(dataset: DatasetABC):
    """
    Compute mean and standard deviation for this dataset.
    """
    all_stats = {}
    for type_tensor in ["inputs", "outputs", "forcing"]:
        stats_dict = compute_mean_std_min_max(dataset, type_tensor)
        for feature, stats in stats_dict.items():
            # If feature was computed multiple times we keep only first occurence
            if feature not in all_stats.keys():
                all_stats[feature] = stats

    dest_file = "/py4cast_root_dir/parameters_stats.pt"
    torch_save(all_stats, Path(dest_file))
    np.save("/py4cast_root_dir/parameters_stats.npy", all_stats)
    print(f"Parameters statistics saved in {dest_file}")


# def compute_time_step_stats(dataset: DatasetABC):
#     random_inputs = next(iter(dataset.torch_dataloader())).inputs
#     n_features = len(random_inputs.feature_names)
#     sum_means = torch.zeros(n_features)
#     sum_squares = torch.zeros(n_features)
#     counter = 0
#     if not dataset.settings.standardize:
#         raise ValueError("Your dataset should be standardized.")

#     for batch in tqdm(dataset.torch_dataloader(), desc="Computing diff stats"):
#         # Here we assume that data are in 2 or 3 D
#         inputs = batch.inputs.tensor
#         outputs = batch.outputs.tensor

#         in_out = torch.cat([inputs, outputs], dim=1)
#         # diff = in_out[:, 1:] - in_out[:, :-1]  # Substract information on time dimension
#         # print("in_out.shape", in_out.shape)
#         # diff = in_out[:, 2] - 2*in_out[:, 1] + in_out[:, 0] # 2nd order difference
#         # diff = -3*in_out[:, 2] + 4*in_out[:, 1] - in_out[:, 0] # Backward Difference (3-point, first-order accurate) formula for the second derivative
#         # diff = in_out[:, 2] - in_out[:, 0] # centered first derivative
#         # diff = in_out[:, 3] - 3*in_out[:, 2] + 3*in_out[:, 1] - in_out[:, 0] # 3rd order difference
#         # diff = in_out[:, 4] - 4*in_out[:, 3] + 6*in_out[:, 2] - 4*in_out[:, 1] + in_out[:, 0] # 4th order difference

#         diff = diff.flatten(1, 3)  # Flatten everybody to be (Batch, X, Features)

#         counter += in_out.shape[0]  # += batch size
#         sum_means += torch.sum(diff.mean(dim=1), dim=0)  # (d_features)
#         sum_squares += torch.sum((diff**2).mean(dim=1), dim=0)  # (d_features)

#     diff_mean = sum_means / counter
#     diff_second_moment = sum_squares / counter
#     diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)
#     store_d = {}

#     # Storing variable statistics
#     for i, name in enumerate(batch.inputs.feature_names):
#         store_d[name] = {
#             "mean": diff_mean[i],
#             "std": diff_std[i],
#         }
#     # Diff mean and std of forcing variables are not used during training so we
#     # store fixed values : mean = 0, std = 1
#     for name in batch.forcing.feature_names:
#         store_d[name] = {"mean": torch.tensor(0), "std": torch.tensor(1)}
#     dest_file =  "/py4cast_root_dir/multi_scale_stats.pt"
#     torch_save(store_d, Path(dest_file))
#     np.save("/py4cast_root_dir/multi_scale_stats.npy", store_d)
#     print(f"Parameters time diff stats saved in {dest_file}")

def smooth(x: torch.Tensor, n: int = 1) -> NamedTensor:
    # Unpack shape
    B, T, N_lat, N_lon, d_f = x.tensor.shape
    # Rearrange for interpolation: (B*T, d_f, N_lat, N_lon)
    aro_inputs = x.tensor.view(B * T, N_lat, N_lon, d_f).permute(0, 3, 1, 2)

    # Map scale factor n -> equivalent Gaussian sigma
    sigma = n / math.sqrt(3.0)

    # Build Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:  # ensure odd
        kernel_size += 1

    coords = torch.arange(kernel_size, device=aro_inputs.device, dtype=aro_inputs.dtype) - kernel_size // 2
    g_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g_1d /= g_1d.sum()


    # Separable 2D kernel
    g_2d = torch.outer(g_1d, g_1d)
    g_2d /= g_2d.sum()

    # Depthwise conv kernel
    g_kernel = g_2d.view(1, 1, kernel_size, kernel_size).to(aro_inputs.dtype).to(aro_inputs.device)
    g_kernel = g_kernel.repeat(d_f, 1, 1, 1)  # one kernel per channel

    # Padding for "same" output size
    pad = kernel_size // 2
    aro_inputs_padded = F.pad(aro_inputs, (pad, pad, pad, pad), mode="reflect")

    # Apply depthwise convolution
    aro_outputs = F.conv2d(aro_inputs_padded, g_kernel, groups=d_f)

    # Rearrange back to (B, T, N_lat, N_lon, d_f)
    aro_outputs_tensor = aro_outputs.permute(0, 2, 3, 1).view(B, T, N_lat, N_lon, d_f)

    # Wrap into NamedTensor
    aro_outputs_nt = NamedTensor.new_like(
        aro_outputs_tensor.type_as(x.tensor), x
    )
    return aro_outputs_nt

# def compute_time_step_stats(dataset: DatasetABC, max_scale_power: int = 5):
#     """
#     Compute per-feature multi-scale time-step statistics.

#     Instead of a single std, we compute std across each frequency band
#     defined by scales = [2^max_scale_power, ..., 2^0].
#     """

#     random_inputs = next(iter(dataset.torch_dataloader())).inputs
#     n_features = len(random_inputs.feature_names)

#     # storage for accumulating band stats
#     n_scales = max_scale_power + 1
#     sum_means = [torch.zeros(n_features) for _ in range(n_scales)]
#     sum_squares = [torch.zeros(n_features) for _ in range(n_scales)]
#     counter = 0

#     if not dataset.settings.standardize:
#         raise ValueError("Your dataset should be standardized.")

#     for batch in tqdm(dataset.torch_dataloader(), desc="Computing multi-scale stats"):
#         outputs = batch.outputs  # NamedTensor (B, T, N_lat, N_lon, d_f)

#         # generate smoothing scales (e.g., [32,16,8,4,2,1] for max_scale_power=5)
#         scales = [2**i for i in reversed(range(max_scale_power + 1))]

#         # smooth diffs at each scale
#         smooths = [smooth(outputs, n=s).tensor if s != 1 else outputs.tensor
#                         for s in scales]

#         # compute band-pass signals
#         bands = []
#         for i in range(len(scales) - 1):
#             band = smooths[i + 1] - smooths[i]
#             bands.append(band)
#         # add lowest frequency (most smoothed)
#         bands.insert(0, smooths[0])

#         # flatten to (Batch*Time*Space, d_f)
#         for i, band in enumerate(bands):
#             flat = band.flatten(0, -2)  # merge B,T,N_lat,N_lon
#             means = flat.mean(dim=0)   # (d_f,)
#             sq_means = (flat**2).mean(dim=0)

#             batch_size = flat.shape[0]
#             counter += batch_size
#             sum_means[i] += means * batch_size
#             sum_squares[i] += sq_means * batch_size

#     # finalize stats
#     diff_means = [sum_means[i] / counter for i in range(n_scales)]
#     diff_second_moments = [sum_squares[i] / counter for i in range(n_scales)]
#     diff_stds = [torch.sqrt(diff_second_moments[i] - diff_means[i] ** 2) for i in range(n_scales)]

#     store_d = {}
#     for f, name in enumerate(batch.inputs.feature_names):
#         store_d[name] = {
#             "mean": diff_means[-1][f],  # keep last (finest scale) mean
#             "std": [diff_stds[i][f] for i in range(n_scales)]
#         }

#     # forcing variables: fixed mean/std
#     for name in batch.forcing.feature_names:
#         store_d[name] = {"mean": torch.tensor(0), "std": [torch.tensor(1) for _ in range(n_scales)]}

#     # save
#     dest_file = "/py4cast_root_dir/multi_scale_gk_stats.pt"
#     torch_save(store_d, Path(dest_file))
#     np.save("/py4cast_root_dir/multi_scale_gk_stats.npy", store_d)
#     print(f"Parameters time diff stats saved in {dest_file}")

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


def compute_time_step_stats(dataset, max_scale_power: int = 5):
    """
    Compute per-feature multi-scale integral histograms.
    For each band and each feature, compute integral of 2D fields over N_lat,N_lon,
    then build histogram distributions over batches.
    """
    random_inputs = next(iter(dataset.torch_dataloader())).inputs
    n_features = len(random_inputs.feature_names)
    n_scales = max_scale_power + 1

    # Storage for integrals: list of [list of values for each feature]
    # Shape: (n_scales, n_features)
    integrals = [[[] for _ in range(n_features)] for _ in range(n_scales)]

    if not dataset.settings.standardize:
        raise ValueError("Your dataset should be standardized.")

    for batch in tqdm(dataset.torch_dataloader(), desc="Computing multi-scale integrals"):
        outputs = batch.outputs  # (B, T, N_lat, N_lon, d_f)

        # generate smoothing scales (e.g., [32,16,8,4,2,1] for max_scale_power=5)
        scales = [2**i for i in reversed(range(max_scale_power + 1))]

        # smooth diffs at each scale
        smooths = [smooth(outputs, n=s).tensor if s != 1 else outputs.tensor for s in scales]

        # compute band-pass signals
        bands = []
        for i in range(len(scales) - 1):
            band = smooths[i + 1] - smooths[i]
            bands.append(band)
        # add lowest frequency (most smoothed)
        bands.insert(0, smooths[0])

        # loop over bands
        for i, band in enumerate(bands):
            # band: (B, T, N_lat, N_lon, d_f)
            # integrate over spatial dims (N_lat, N_lon)
            integrals_batch = band.sum(dim=(-3, -2))  # (B, T, d_f)
            integrals_batch = integrals_batch.reshape(-1, n_features)  # (B*T, d_f)

            for f in range(n_features):
                integrals[i][f].extend(integrals_batch[:, f].cpu().numpy().tolist())

    # Now plot histograms for each (feature, band)
    out_dir = Path("/py4cast_root_dir/multi_scale_density_plots/")
    out_dir.mkdir(parents=True, exist_ok=True)

    means_over_bands = np.zeros((n_features, n_scales))

    for i in range(n_scales):
        for f, name in enumerate(random_inputs.feature_names):
            values = np.array(integrals[i][f])
            mean_val, std_val = values.mean(), values.std()
            means_over_bands[f, i] = mean_val

            plt.figure(figsize=(6, 4))
            plt.hist(values, bins=50, alpha=0.7, color="blue")
            plt.axvline(mean_val, color="red", linestyle="--", label=f"mean={mean_val:.2f}")
            plt.axvline(mean_val + std_val, color="green", linestyle=":", label=f"std={std_val:.2f}")
            plt.axvline(mean_val - std_val, color="green", linestyle=":")
            plt.title(f"Histogram of integrals - Feature: {name}, Band {i}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"hist_feature-{name}_band-{i}.png")
            plt.close()

    # Plot mean integrals vs band for each feature
    plt.figure(figsize=(8, 6))
    scales = [2**i for i in reversed(range(max_scale_power + 1))]
    for f, name in enumerate(random_inputs.feature_names):
        plt.plot(range(n_scales), means_over_bands[f], marker="o", label=name)
    plt.xticks(range(n_scales), [str(s) for s in scales])
    plt.xlabel("Scale (2^i)")
    plt.ylabel("Mean integral value")
    plt.title("Mean integral values over bands")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mean_integrals_over_bands.png")
    plt.close()

    print(f"Saved histograms and summary figure to {out_dir}")


