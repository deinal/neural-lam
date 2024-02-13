# Standard library
import os
from argparse import ArgumentParser
from pathlib import Path

# Third-party
import numpy as np
import torch
from tqdm import tqdm

# First-party
from neural_lam import constants
from neural_lam.weather_dataset import WeatherDataset


def calculate_static_features(data, output_dir):
    """
    Calculates static features based on the input data and saves them.
    """
    earth_mask = data[0, :, :, constants.PARAM_NAMES.index("rho")] == 0
    np.save(Path(output_dir, "earth_mask.npy"), earth_mask)

    y_indices, x_indices = np.indices(earth_mask.shape)
    nwp_xy = np.array([x_indices, y_indices])
    np.save(Path(output_dir, "nwp_xy.npy"), nwp_xy)

    y_coords, x_coords = y_indices[earth_mask], x_indices[earth_mask]
    x_center = (np.min(x_coords) + np.max(x_coords)) / 2
    y_center = (np.min(y_coords) + np.max(y_coords)) / 2
    earth_center = np.array([y_center, x_center])
    grid_coords = np.stack((y_indices, x_indices), axis=-1)
    geopotential = np.linalg.norm(grid_coords - earth_center, axis=-1)
    np.save(Path(output_dir, "surface_geopotential.npy"), geopotential)

    num_features = constants.GRID_STATE_DIM
    parameter_weights = np.ones(num_features, dtype=np.float32)
    np.save(Path(output_dir, "parameter_weights.npy"), parameter_weights)


def main():
    """
    Create parameter weights.
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="space_weather",
        help="Dataset to compute weights for (default: space_weather)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=1,
        help="Step length in minutes to consider single time step (default: 1)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    args = parser.parse_args()

    # Create static features
    static_dir_path = Path("data", args.dataset, "static")
    os.makedirs(static_dir_path, exist_ok=True)
    train_samples_dir_path = Path("data", args.dataset, "samples", "train")
    file_paths = sorted(train_samples_dir_path.glob("*.npy"))
    data = np.load(next(iter(file_paths)))
    calculate_static_features(data, static_dir_path)

    # Load dataset without any subsampling
    ds = WeatherDataset(
        args.dataset,
        split="train",
        subsample_step=1,
        pred_length=3,
        standardize=False,
    )  # Without standardization
    loader = torch.utils.data.DataLoader(
        ds, args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    # Compute mean and std.-dev. of each parameter across full dataset
    print("Computing mean and std.-dev. for parameters...")
    means = []
    squares = []
    for init_batch, target_batch in tqdm(loader):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t, N_grid, d_features)
        means.append(torch.mean(batch, dim=(1, 2)))  # (N_batch, d_features,)
        squares.append(
            torch.mean(batch**2, dim=(1, 2))
        )  # (N_batch, d_features,)

    mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    print("Saving mean, std.-dev. for parameters...")
    torch.save(mean, Path(static_dir_path, "parameter_mean.pt"))
    torch.save(std, Path(static_dir_path, "parameter_std.pt"))

    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        args.dataset,
        split="train",
        subsample_step=1,
        pred_length=3,
        standardize=True,
    )  # Re-load with standardization
    loader_standard = torch.utils.data.DataLoader(
        ds_standard, args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    used_subsample_len = (
        constants.SAMPLE_LEN["train"] // args.step_length
    ) * args.step_length

    diff_means = []
    diff_squares = []
    for init_batch, target_batch in tqdm(loader_standard):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t', N_grid, d_features)
        # Note: batch contains only 1min-steps
        stepped_batch = torch.cat(
            [
                batch[:, ss_i : used_subsample_len : args.step_length]
                for ss_i in range(args.step_length)
            ],
            dim=0,
        )
        # (N_batch', N_t, N_grid, d_features),
        # N_batch' = args.step_length*N_batch

        batch_diffs = stepped_batch[:, 1:] - stepped_batch[:, :-1]
        # (N_batch', N_t-1, N_grid, d_features)

        diff_means.append(
            torch.mean(batch_diffs, dim=(1, 2))
        )  # (N_batch', d_features,)
        diff_squares.append(
            torch.mean(batch_diffs**2, dim=(1, 2))
        )  # (N_batch', d_features,)

    diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
    diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

    print("Saving one-step difference mean and std.-dev...")
    torch.save(diff_mean, Path(static_dir_path, "diff_mean.pt"))
    torch.save(diff_std, Path(static_dir_path, "diff_std.pt"))


if __name__ == "__main__":
    main()
