# Third-party
import numpy as np

WANDB_PROJECT = "spacecast-alpha"

# Variable names
PARAM_NAMES = [
    "B",
    "Bx",
    "By",
    "Bz",
    "E",
    "Ex",
    "Ey",
    "Ez",
    "v",
    "vx",
    "vy",
    "vz",
    "rho",
    "pressure",
    "temperature",
    "agyrotropy",
    "anisotropy",
    "reconnection",
]

PARAM_NAMES_SHORT = [
    "B",
    "Bx",
    "By",
    "Bz",
    "E",
    "Ex",
    "Ey",
    "Ez",
    "v",
    "vx",
    "vy",
    "vz",
    "rho",
    "pressure",
    "temperature",
    "agyrotropy",
    "anisotropy",
    "reconnection",
]

PARAM_UNITS = [
    "T",
    "T",
    "T",
    "T",
    "V/m",
    "V/m",
    "V/m",
    "V/m",
    "km/s",
    "km/s",
    "km/s",
    "km/s",
    "1/cm³",
    "Pa",
    "K",
    "-",
    "-",
    "-",
]

# Projection and grid
# Hard coded for now, but should eventually be part of dataset desc. files
GRID_SHAPE = (428, 642)  # (y, x)

GRID_LIMITS = [-20, 10, -10, 10]

# Data dimensions
GRID_STATE_DIM = 18

# Sample lengths
SAMPLE_LEN = {
    "train": 5,
    "val": 5,
    "test": 32,
}


# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 2, 3])

# Log these metrics to wandb as scalar values for
# specific variables and lead times
# List of metrics to watch, including any prefix (e.g. val_rmse)
METRICS_WATCH = []
# Dict with variables and lead times to log watched metrics for
# Format is a dictionary that maps from a variable index to
# a list of lead time steps
VAR_LEADS_METRICS_WATCH = {
    PARAM_NAMES.index("B"): [1, 2, 3],
    PARAM_NAMES.index("E"): [1, 2, 3],
    PARAM_NAMES.index("v"): [1, 2, 3],
}
