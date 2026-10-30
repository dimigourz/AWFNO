#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled Burgers 1D reviewer experiments for AWFNO.

Default suite (REVIEWER_SUITE=core):
    1. AWFNO local-Fourier-local (LFL)
    2. local-only local-identity-local (LIL)
    3. in-block Fourier-only identity-Fourier-identity (IFI)
    4. standard WFNO at width 32
    5. parameter-matched WFNO at width 42

Additional suites:
    REVIEWER_SUITE=ordering
        compares LFL, LLF, and FLL with identical learned operators.
    REVIEWER_SUITE=sensitivity
        sweeps the local radius, retained Fourier modes, and window length.
    REVIEWER_SUITE=all
        runs every unique configuration.

All controlled models use exactly the same:
    - disjoint train/validation/test split;
    - window protocol;
    - physical window-plus-final-state loss;
    - AdamW optimizer and cosine learning-rate schedule;
    - gradient clipping, epochs, and checkpoint-selection metric.

The script automatically reports:
    - physical full-trajectory and final-time L2 errors;
    - low/high-frequency error decomposition;
    - error in the top-gradient (shock) region and its complement;
    - parameter counts, training time, forward latency, rollout latency,
      network-call count, and peak allocated GPU memory.

Input HDF5 solutions are converted to U with shape (N, Nx, Nt). At test time,
only the true initial condition is used; subsequent rollout inputs are previous
predictions.
"""

import os
import gc
import re
import csv
import json
import copy
import time
import random
import hashlib
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. ENV HELPERS / CONFIG
# ============================================================

def env_bool(name, default=False):
    value = os.environ.get(name, None)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name, default):
    value = os.environ.get(name, None)
    return int(value) if value not in (None, "") else int(default)


def env_float(name, default):
    value = os.environ.get(name, None)
    return float(value) if value not in (None, "") else float(default)


def env_list_int(name, default_list):
    value = os.environ.get(name, None)
    if value is None or str(value).strip() == "":
        return list(default_list)
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def env_str(name, default):
    value = os.environ.get(name, None)
    return str(default) if value in (None, "") else str(value)


def env_list_str(name, default_list):
    value = os.environ.get(name, None)
    if value is None or str(value).strip() == "":
        return list(default_list)
    return [x.strip() for x in str(value).split(",") if x.strip()]


SEED = env_int("SEED", 1234)

# HDF5 input. Copy the file to a fast local path when possible.
DATA_PATH = Path(os.environ.get(
    "DATA_PATH",
    "/content/drive/MyDrive/anus/1D_Burgers_Sols_Nu0.001.hdf5",
))

OUTPUT_ROOT = Path(os.environ.get(
    "OUTPUT_ROOT",
    "/content/drive/MyDrive/anus/burgers_reviewer_ablations",
))

TARGET_NX = env_int("TARGET_NX", 512)
TIME_SUBSAMPLE = env_int("TIME_SUBSAMPLE", 1)

# Optional explicit HDF5 keys. Leave empty for automatic detection.
HDF5_SOLUTION_KEY = os.environ.get("HDF5_SOLUTION_KEY", "") or None
HDF5_X_KEY = os.environ.get("HDF5_X_KEY", "") or None
HDF5_T_KEY = os.environ.get("HDF5_T_KEY", "") or None

SPLIT_MODE = os.environ.get("SPLIT_MODE", "contiguous")  # "contiguous" or "random"
SPLIT_SEED = env_int("SPLIT_SEED", 2026)

# Use ALL_SAMPLE_SIZES for stable split/cache paths. RUN_SAMPLE_SIZES selects what to run now.
ALL_SAMPLE_SIZES = env_list_int("ALL_SAMPLE_SIZES", [25, 50, 100, 200, 400, 800])
SAMPLE_SIZES = env_list_int("RUN_SAMPLE_SIZES", ALL_SAMPLE_SIZES)
N_TRAIN_POOL = max(ALL_SAMPLE_SIZES)
N_VAL = env_int("N_VAL", 50)
N_TEST = env_int("N_TEST", 100)
LOW_RAM_HDF5 = env_bool("LOW_RAM_HDF5", True)

WINDOW_K = env_int("WINDOW_K", 5)
WINDOW_STRIDE = env_int("WINDOW_STRIDE", 5)
N_TIME_MODES = env_int("N_TIME_MODES", 5)

BATCH_SIZE = env_int("BATCH_SIZE", 128)
EPOCHS = env_int("EPOCHS", 200)
LR = env_float("LR", 1.0e-3)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 1.0e-6)

WIDTH = env_int("WIDTH", 32)
N_LAYERS = env_int("N_LAYERS", 4)
MODES_X = env_int("MODES_X", 16)
LOCAL_RADIUS = env_int("LOCAL_RADIUS", 3)

FINAL_LOSS_WEIGHT = env_float("FINAL_LOSS_WEIGHT", 1.0)
ENFORCE_MEAN = env_bool("ENFORCE_MEAN", False)

USE_LIBRARY_FNO = env_bool("USE_LIBRARY_FNO", True)
FORCE_RETRAIN = env_bool("FORCE_RETRAIN", False)
RUN_EXACT_HASH_OVERLAP_CHECK = env_bool("RUN_EXACT_HASH_OVERLAP_CHECK", True)
RUN_DISTANCE_OVERLAP_CHECK = env_bool("RUN_DISTANCE_OVERLAP_CHECK", False)
OVERLAP_TOL = env_float("OVERLAP_TOL", 1.0e-10)

SAVE_PLOTS = env_bool("SAVE_PLOTS", True)
SHOW_PLOTS = env_bool("SHOW_PLOTS", False)
MAKE_TRAINING_CURVE_PLOTS = env_bool("MAKE_TRAINING_CURVE_PLOTS", True)
MAKE_L2_PER_TIME_PLOT = env_bool("MAKE_L2_PER_TIME_PLOT", True)
MAKE_PER_TEST_L2_PLOTS = env_bool("MAKE_PER_TEST_L2_PLOTS", False)
MAKE_SOLUTION_PLOTS = env_bool("MAKE_SOLUTION_PLOTS", True)
MAKE_BEST_WORST_PLOTS = env_bool("MAKE_BEST_WORST_PLOTS", True)
N_BEST_WORST = env_int("N_BEST_WORST", 2)
MAX_TEST_CASES_TO_PLOT = os.environ.get("MAX_TEST_CASES_TO_PLOT", "10")
MAX_TEST_CASES_TO_PLOT = None if str(MAX_TEST_CASES_TO_PLOT).lower() == "none" else int(MAX_TEST_CASES_TO_PLOT)
PLOT_DPI = env_int("PLOT_DPI", 140)
L2_CHUNK_SIZE = env_int("L2_CHUNK_SIZE", 50)

USE_TRAPEZOIDAL_TIME_L2 = env_bool("USE_TRAPEZOIDAL_TIME_L2", True)
INCLUDE_T0_IN_FULL_L2 = env_bool("INCLUDE_T0_IN_FULL_L2", True)

# Comma-separated entries may include integers or "final".
_plot_times_raw = os.environ.get("PLOT_TIME_INDICES", "25,50,final")
PLOT_TIME_INDICES_MANUAL = []
for item in _plot_times_raw.split(","):
    item = item.strip()
    if not item:
        continue
    PLOT_TIME_INDICES_MANUAL.append("final" if item.lower() == "final" else int(item))

EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "burgers_reviewer_ablations")

# Reviewer experiments. The default "core" suite deliberately runs only the
# highest-priority controlled comparisons. Select "ordering", "sensitivity",
# or "all" after the core suite has completed.
REVIEWER_SUITE = env_str("REVIEWER_SUITE", "core").strip().lower()
REVIEWER_N_TRAIN = env_int("REVIEWER_N_TRAIN", 400)
REVIEWER_SEEDS = env_list_int("REVIEWER_SEEDS", [SEED])
REVIEWER_MODEL_FILTER = env_list_str("REVIEWER_MODELS", [])
WFNO_PM_WIDTH = env_int("WFNO_PM_WIDTH", 42)

R_SWEEP = env_list_int("R_SWEEP", [1, 3, 5])
MODES_SWEEP = env_list_int("MODES_SWEEP", [8, 16, 32])
K_SWEEP = env_list_int("K_SWEEP", [1, 5, 10])

COMMON_FINAL_LOSS_WEIGHT = env_float(
    "COMMON_FINAL_LOSS_WEIGHT",
    FINAL_LOSS_WEIGHT,
)
SHARP_GRADIENT_QUANTILE = env_float("SHARP_GRADIENT_QUANTILE", 0.90)
FREQUENCY_CUTOFF = env_int("FREQUENCY_CUTOFF", MODES_X)

PROFILE_INFERENCE = env_bool("PROFILE_INFERENCE", True)
PROFILE_WARMUP = env_int("PROFILE_WARMUP", 5)
PROFILE_FORWARD_REPEATS = env_int("PROFILE_FORWARD_REPEATS", 30)
PROFILE_ROLLOUT_REPEATS = env_int("PROFILE_ROLLOUT_REPEATS", 5)
SAVE_REVIEWER_ROLLOUTS = env_bool("SAVE_REVIEWER_ROLLOUTS", True)
PRINT_ROLLOUT_BLOCKS = env_bool("PRINT_ROLLOUT_BLOCKS", False)

CACHE_ROOT = OUTPUT_ROOT / (
    f"cache_{EXPERIMENT_NAME}_"
    f"sizes{'-'.join(map(str, ALL_SAMPLE_SIZES))}_"
    f"K{WINDOW_K}_M{N_TIME_MODES}_W{WIDTH}_mx{MODES_X}_R{LOCAL_RADIUS}"
)
MODEL_DIR = CACHE_ROOT / "models"
PLOT_DIR = CACHE_ROOT / "plots"
DATA_DIR = CACHE_ROOT / "data"
RESULTS_DIR = CACHE_ROOT / "results"

for d in [MODEL_DIR, PLOT_DIR, DATA_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcdefaults()
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.dpi": 120,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "-",
    "axes.spines.top": True,
    "axes.spines.right": True,
})


# ============================================================
# 1. REPRODUCIBILITY / DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

    # GPU speed settings. TF32 is useful on Ampere+ GPUs and harmless elsewhere.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = device.type == "cuda"
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. GLOBAL GEOMETRY
# ============================================================

x_grid_np = None
t_grid = None
Nx = None
Nt = None
DX = None
DT_SNAPSHOT = 1.0
BASE_TAG = "uninitialized"


def set_geometry_from_arrays(U_shape, x_grid, t=None):
    global x_grid_np, t_grid, Nx, Nt, DX, DT_SNAPSHOT

    if len(U_shape) != 3:
        raise ValueError(f"Expected U shape (N,Nx,Nt). Got {U_shape}")

    _, Nx, Nt = U_shape
    x_grid_np = np.asarray(x_grid, dtype=np.float32)

    if len(x_grid_np) != Nx:
        raise ValueError(f"x_grid length {len(x_grid_np)} does not match Nx={Nx}")

    DX = float(np.mean(np.diff(x_grid_np.astype(np.float64)))) if Nx > 1 else 1.0

    if t is None:
        t_grid = np.arange(Nt, dtype=np.float32)
        DT_SNAPSHOT = 1.0
    else:
        t_grid = np.asarray(t, dtype=np.float32)
        if len(t_grid) != Nt:
            print("Warning: t_grid length mismatch. Using snapshot index as time grid.")
            t_grid = np.arange(Nt, dtype=np.float32)
            DT_SNAPSHOT = 1.0
        elif Nt > 1:
            DT_SNAPSHOT = float(np.mean(np.diff(t_grid.astype(np.float64))))
        else:
            DT_SNAPSHOT = 1.0

    print("\nGeometry")
    print("--------")
    print("N, Nx, Nt   :", U_shape[0], Nx, Nt)
    print("x range     :", float(x_grid_np.min()), float(x_grid_np.max()))
    print("DX          :", DX)
    print("DT_SNAPSHOT :", DT_SNAPSHOT)


# ============================================================
# 3. BASIC HELPERS
# ============================================================

def short_hash(obj):
    txt = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(txt.encode("utf-8")).hexdigest()[:10]


def clean_state_dict(sd):
    sd = copy.deepcopy(sd)
    if isinstance(sd, dict) and "_metadata" in sd:
        del sd["_metadata"]
    if hasattr(sd, "_metadata"):
        try:
            del sd._metadata
        except Exception:
            pass
    return OrderedDict((k, v) for k, v in sd.items() if k != "_metadata")


def save_model_checkpoint(path, model, meta=None):
    torch.save({"state_dict": clean_state_dict(model.state_dict()), "meta": meta or {}}, path)


def load_model_checkpoint(path, model):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(clean_state_dict(payload["state_dict"]), strict=True)
    return payload.get("meta", {})


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def to_device(x):
    return x.to(device, non_blocking=PIN_MEMORY)


# ============================================================
# 4. HDF5 DATA LOADING
# ============================================================

def _require_h5py():
    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise ImportError("Install h5py with: pip install h5py") from exc


def _list_hdf5_datasets(h5file):
    datasets = []
    def visitor(name, obj):
        if hasattr(obj, "shape") and obj.shape is not None:
            datasets.append((name, tuple(obj.shape), str(getattr(obj, "dtype", ""))))
    h5file.visititems(visitor)
    return datasets


def _find_first_matching_key(h5file, candidates):
    all_names = []
    def visitor(name, obj):
        if hasattr(obj, "shape") and obj.shape is not None:
            all_names.append(name)
    h5file.visititems(visitor)

    lower_to_name = {name.lower(): name for name in all_names}
    for cand in candidates:
        if cand.lower() in lower_to_name:
            return lower_to_name[cand.lower()]

    for cand in candidates:
        c = cand.lower()
        for name in all_names:
            n = name.lower()
            if n.endswith("/" + c) or n == c:
                return name
    return None


def _auto_find_solution_key(h5file):
    """Find the solution dataset without reading large arrays into RAM."""
    preferred = ["tensor", "U", "u", "solution", "solutions", "data", "sols", "fields", "trajectories"]
    key = _find_first_matching_key(h5file, preferred)
    if key is not None:
        dset = h5file[key]
        if len(dset.shape) in (3, 4):
            return key

    candidates = []
    for name, shape, dtype in _list_hdf5_datasets(h5file):
        if len(shape) in (3, 4) and np.prod(shape) > 0:
            candidates.append((name, shape, dtype, int(np.prod(shape))))
    if not candidates:
        raise KeyError("Could not automatically find a 3D/4D solution dataset in the HDF5 file.")
    candidates.sort(key=lambda item: item[-1], reverse=True)
    return candidates[0][0]


def _auto_find_grid_key(h5file, kind):
    if kind == "x":
        candidates = ["x", "x_grid", "x-coordinate", "x_coordinate", "space", "grid/x", "coordinates/x"]
    elif kind == "t":
        candidates = ["t", "time", "t_grid", "t-coordinate", "t_coordinate", "grid/t", "coordinates/t"]
    else:
        raise ValueError(kind)

    key = _find_first_matching_key(h5file, candidates)
    if key is None:
        return None
    arr = np.asarray(h5file[key])
    return key if arr.ndim == 1 else None


def _squeeze_solution_array(arr):
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 3 and 1 in arr.shape:
        axis = arr.shape.index(1)
        arr = np.squeeze(arr, axis=axis)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array after squeezing singleton channels. Got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _convert_solution_to_nxt(arr, x=None, t=None):
    arr = _squeeze_solution_array(arr)
    shape = arr.shape
    axes = list(range(3))
    x_len = None if x is None else int(len(x))
    t_len = None if t is None else int(len(t))

    if x_len is not None and t_len is not None:
        x_axes = [a for a in axes if shape[a] == x_len]
        t_axes = [a for a in axes if shape[a] == t_len]
        for ax_x in x_axes:
            for ax_t in t_axes:
                if ax_x == ax_t:
                    continue
                ax_n = [a for a in axes if a not in (ax_x, ax_t)][0]
                return np.transpose(arr, (ax_n, ax_x, ax_t)).astype(np.float32)

    if x_len is not None:
        x_axes = [a for a in axes if shape[a] == x_len]
        if x_axes:
            ax_x = x_axes[0]
            remaining = [a for a in axes if a != ax_x]
            if ax_x == 1:
                ax_n, ax_t = 0, 2
            elif ax_x == 2:
                ax_n, ax_t = 0, 1
            else:
                ax_n, ax_t = remaining[0], remaining[1]
            return np.transpose(arr, (ax_n, ax_x, ax_t)).astype(np.float32)

    if t_len is not None:
        t_axes = [a for a in axes if shape[a] == t_len]
        if t_axes:
            ax_t = t_axes[0]
            remaining = [a for a in axes if a != ax_t]
            if ax_t == 2:
                ax_n, ax_x = 0, 1
            elif ax_t == 1:
                ax_n, ax_x = 0, 2
            else:
                ax_n, ax_x = remaining[0], remaining[1]
            return np.transpose(arr, (ax_n, ax_x, ax_t)).astype(np.float32)

    return arr.astype(np.float32)


def _read_solution_subset_low_ram(dset, n_rows_needed):
    """
    Read only the trajectories needed for the configured split.

    This assumes the HDF5 solution dataset uses the standard PDEBench convention
    where the first axis is the trajectory/sample axis. This is true for the
    usual Burgers files, e.g. tensor.shape = (N, T, Nx) or (N, Nx, T).
    """
    total_rows = int(dset.shape[0])
    n_read = min(int(n_rows_needed), total_rows)

    if SPLIT_MODE == "random":
        rng = np.random.default_rng(SPLIT_SEED)
        chosen = rng.permutation(total_rows)[:n_read]
        order = np.argsort(chosen)
        chosen_sorted = chosen[order]
        raw_sorted = np.asarray(dset[chosen_sorted, ...], dtype=np.float32)
        inv_order = np.argsort(order)
        raw = raw_sorted[inv_order]
        print(f"LOW_RAM_HDF5: read {n_read}/{total_rows} randomly selected trajectories from HDF5.")
    else:
        raw = np.asarray(dset[:n_read, ...], dtype=np.float32)
        print(f"LOW_RAM_HDF5: read first {n_read}/{total_rows} trajectories from HDF5.")

    return raw


def load_hdf5_1d_dataset(path):
    _require_h5py()
    import h5py

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"HDF5 data file not found: {path}\n"
            "Set DATA_PATH to the correct local path before running."
        )

    n_rows_needed = N_TRAIN_POOL + N_VAL + N_TEST

    with h5py.File(path, "r") as f:
        print("\nHDF5 datasets")
        print("-------------")
        for name, shape, dtype in _list_hdf5_datasets(f):
            print(f"  {name}: shape={shape}, dtype={dtype}")

        solution_key = HDF5_SOLUTION_KEY or _auto_find_solution_key(f)
        x_key = HDF5_X_KEY or _auto_find_grid_key(f, "x")
        t_key = HDF5_T_KEY or _auto_find_grid_key(f, "t")

        print("\nSelected HDF5 keys")
        print("------------------")
        print("solution:", solution_key)
        print("x       :", x_key)
        print("t       :", t_key)

        dset = f[solution_key]
        print("solution shape:", tuple(dset.shape), "dtype:", dset.dtype)
        print("LOW_RAM_HDF5:", LOW_RAM_HDF5)
        print("rows needed for split:", n_rows_needed)

        x = None if x_key is None else np.asarray(f[x_key], dtype=np.float32).reshape(-1)
        t = None if t_key is None else np.asarray(f[t_key], dtype=np.float32).reshape(-1)

        if LOW_RAM_HDF5:
            if len(dset.shape) not in (3, 4):
                raise ValueError(f"Expected 3D/4D solution dataset. Got shape {tuple(dset.shape)}")
            raw = _read_solution_subset_low_ram(dset, n_rows_needed)
        else:
            print("WARNING: LOW_RAM_HDF5=False; reading the entire solution dataset into RAM.")
            raw = np.asarray(dset, dtype=np.float32)

    U = _convert_solution_to_nxt(raw, x=x, t=t)
    del raw
    gc.collect()

    if x is None:
        x = np.arange(U.shape[1], dtype=np.float32)
    if t is None:
        t = np.arange(U.shape[2], dtype=np.float32)

    if TIME_SUBSAMPLE > 1:
        U = U[:, :, ::TIME_SUBSAMPLE].copy()
        t = t[::TIME_SUBSAMPLE].copy()
        gc.collect()

    if not np.all(np.isfinite(U)):
        raise ValueError("Loaded solution array contains NaN or Inf values.")

    U = U.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    t = t.astype(np.float32, copy=False)

    print("\nLoaded data")
    print("-----------")
    print("U shape:", U.shape, " internal layout = (N,Nx,Nt)")
    print("x shape:", x.shape)
    print("t shape:", t.shape)
    print("U memory GB:", U.nbytes / 1024**3)
    print("U min/max:", float(U.min()), float(U.max()))

    return U, x, t

def downsample_space_stride(U, x, target_nx):
    if target_nx is None or int(target_nx) <= 0:
        return U, x

    target_nx = int(target_nx)
    U = np.asarray(U, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    N, Nx0, Nt0 = U.shape

    if len(x) != Nx0:
        raise ValueError(f"len(x)={len(x)} does not match Nx={Nx0}")
    if target_nx > Nx0:
        raise ValueError(f"TARGET_NX={target_nx} is larger than current Nx={Nx0}")
    if Nx0 % target_nx != 0:
        raise ValueError(f"Stride downsampling requires Nx={Nx0} divisible by TARGET_NX={target_nx}")

    factor = Nx0 // target_nx
    U_down = U[:, ::factor, :]
    x_down = x[::factor]

    print("\nSpatial downsampling")
    print("--------------------")
    print("original U shape:", U.shape)
    print("target Nx       :", target_nx)
    print("stride factor   :", factor)
    print("new U shape     :", U_down.shape)

    return U_down.astype(np.float32), x_down.astype(np.float32)


def make_train_val_test_splits(U_all):
    needed = N_TRAIN_POOL + N_VAL + N_TEST
    if U_all.shape[0] < needed:
        raise ValueError(
            f"Need at least {needed} trajectories: train_pool={N_TRAIN_POOL}, "
            f"val={N_VAL}, test={N_TEST}, but data has {U_all.shape[0]}. "
            "Reduce ALL_SAMPLE_SIZES/RUN_SAMPLE_SIZES/N_VAL/N_TEST."
        )

    ids = np.arange(U_all.shape[0], dtype=np.int64)
    if SPLIT_MODE == "random":
        rng = np.random.default_rng(SPLIT_SEED)
        ids = rng.permutation(ids)
    elif SPLIT_MODE != "contiguous":
        raise ValueError(f"Unknown SPLIT_MODE={SPLIT_MODE!r}")

    selected = ids[:needed]
    train_ids = selected[:N_TRAIN_POOL]
    val_ids = selected[N_TRAIN_POOL:N_TRAIN_POOL + N_VAL]
    test_ids = selected[N_TRAIN_POOL + N_VAL:N_TRAIN_POOL + N_VAL + N_TEST]

    U_train_pool = U_all[train_ids]
    U_val = U_all[val_ids]
    U_test = U_all[test_ids]

    print("\nData split")
    print("----------")
    print("split mode       :", SPLIT_MODE)
    print("U_train_pool     :", U_train_pool.shape)
    print("U_val            :", U_val.shape)
    print("U_test           :", U_test.shape)
    print("train ids first  :", train_ids[:5])
    print("val ids first    :", val_ids[:5])
    print("test ids first   :", test_ids[:5])

    return U_train_pool, train_ids, U_val, val_ids, U_test, test_ids


def save_split_dataset_npz(path, U_train_pool, train_ids, U_val, val_ids, U_test, test_ids, x, t):
    np.savez_compressed(
        path,
        U_train_pool=U_train_pool,
        train_ids=train_ids,
        U_val=U_val,
        val_ids=val_ids,
        U_test=U_test,
        test_ids=test_ids,
        x_grid=x,
        t_grid=t,
        sample_sizes=np.array(ALL_SAMPLE_SIZES, dtype=np.int64),
        run_sample_sizes=np.array(SAMPLE_SIZES, dtype=np.int64),
        data_path=np.array(str(DATA_PATH)),
    )
    print("Saved split dataset:", path)


def trajectory_hashes(U):
    U32 = np.ascontiguousarray(U.astype(np.float32, copy=False))
    return [hashlib.sha1(U32[i].tobytes()).hexdigest() for i in range(U32.shape[0])]


def check_exact_hash_overlap(U_a, U_b, name_a, name_b):
    h_a = set(trajectory_hashes(U_a))
    h_b = set(trajectory_hashes(U_b))
    inter = h_a.intersection(h_b)
    print(f"\nExact hash overlap check {name_a} vs {name_b}")
    print("------------------------------------------")
    print("duplicates:", len(inter))
    if inter:
        raise RuntimeError(f"Exact duplicate trajectory found between {name_a} and {name_b}.")


def check_dataset_overlap(U_a, U_b, name_a="A", name_b="B", tol=1.0e-10):
    A = U_a.reshape(U_a.shape[0], -1)
    B = U_b.reshape(U_b.shape[0], -1)
    min_dist = float("inf")
    min_pair = None
    for i in range(A.shape[0]):
        d = np.linalg.norm(B - A[i][None, :], axis=1)
        j = int(np.argmin(d))
        if d[j] < min_dist:
            min_dist = float(d[j])
            min_pair = (i, j)
    print(f"\nDistance overlap check {name_a} vs {name_b}: min distance={min_dist:.6e}, pair={min_pair}")
    if min_dist < tol:
        raise RuntimeError(f"Possible duplicate trajectory between {name_a} and {name_b}.")
    return min_dist, min_pair


# ============================================================
# 5. METRICS
# ============================================================

def _time_weights_np(nt):
    w = np.ones(nt, dtype=np.float64)
    if USE_TRAPEZOIDAL_TIME_L2 and nt > 1:
        w[0] = 0.5
        w[-1] = 0.5
    return w


def _time_weights_torch(nt, device_, dtype):
    w = torch.ones(nt, device=device_, dtype=dtype)
    if USE_TRAPEZOIDAL_TIME_L2 and nt > 1:
        w[0] = 0.5
        w[-1] = 0.5
    return w


def abs_l2_window_batch(pred, true):
    w = _time_weights_torch(pred.shape[-1], pred.device, pred.dtype).view(1, 1, -1)
    err2 = (pred - true) ** 2
    return torch.sqrt(torch.sum(err2 * w, dim=(1, 2)) * DX * DT_SNAPSHOT + 1.0e-30)


def abs_l2_field_batch(pred, true):
    return torch.sqrt(torch.sum((pred - true) ** 2, dim=1) * DX + 1.0e-30)


def l2_per_timestep_np(pred, true, chunk_size=L2_CHUNK_SIZE):
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")
    n, nx, nt = pred.shape
    if nx != Nx:
        raise ValueError(f"Geometry mismatch: array has Nx={nx}, global Nx={Nx}")

    out = np.zeros((n, nt), dtype=np.float64)
    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        err = pred[i0:i1].astype(np.float32, copy=False) - true[i0:i1].astype(np.float32, copy=False)
        out[i0:i1] = np.sqrt(np.sum(err * err, axis=1, dtype=np.float64) * DX + 1.0e-30)
        del err
        gc.collect()
    return out


def abs_l2_full_traj_np(pred, true):
    if INCLUDE_T0_IN_FULL_L2:
        pred_use, true_use = pred, true
    else:
        pred_use, true_use = pred[:, :, 1:], true[:, :, 1:]
    l2_t = l2_per_timestep_np(pred_use, true_use)
    w = _time_weights_np(l2_t.shape[1]).reshape(1, -1)
    return np.sqrt(np.sum(w * l2_t ** 2, axis=1) * DT_SNAPSHOT + 1.0e-30)


def abs_l2_final_np(pred, true):
    return l2_per_timestep_np(pred, true)[:, -1]


def rollout_l2_metrics(pred, true):
    full = abs_l2_full_traj_np(pred, true)
    final = abs_l2_final_np(pred, true)
    l2_t = l2_per_timestep_np(pred, true)
    return {
        "full_l2_mean": float(np.mean(full)),
        "full_l2_std": float(np.std(full)),
        "full_l2_median": float(np.median(full)),
        "final_l2_mean": float(np.mean(final)),
        "final_l2_std": float(np.std(final)),
        "final_l2_median": float(np.median(final)),
        "mean_t": np.mean(l2_t, axis=0),
        "median_t": np.median(l2_t, axis=0),
        "std_t": np.std(l2_t, axis=0),
        "all_full_l2": full,
        "all_final_l2": final,
        "all_l2_t": l2_t,
    }


# ============================================================
# 6. WINDOWS / TEMPORAL BASIS
# ============================================================

def make_window_start_indices(nt, K, stride):
    if nt <= K:
        raise ValueError(f"Need Nt > K. Got Nt={nt}, K={K}")
    return list(range(0, nt - K, stride))


def build_window_tensors(U, starts):
    N, nx, nt = U.shape
    u0 = np.zeros((N * len(starts), nx), dtype=np.float32)
    Uf = np.zeros((N * len(starts), nx, WINDOW_K), dtype=np.float32)
    q = 0
    for i in range(N):
        for s in starts:
            u0[q] = U[i, :, s]
            Uf[q] = U[i, :, s + 1:s + WINDOW_K + 1]
            q += 1
    return u0, Uf


def compute_temporal_basis_from_windows(u_start, U_future, n_modes):
    R = U_future - u_start[:, :, None]
    X = R.reshape(-1, WINDOW_K).astype(np.float64)
    print("\nComputing temporal basis from training windows")
    print("------------------------------------------------")
    print("basis matrix shape:", X.shape)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    r = min(n_modes, Vt.shape[0])
    basis = Vt[:r].T.astype(np.float32)
    energy = S ** 2
    ratio = float(np.sum(energy[:r]) / (np.sum(energy) + 1.0e-12))
    print(f"Effective rank  : {r}")
    print(f"Energy captured : {ratio:.10f}")
    print("Singular values :", S[:min(20, len(S))])
    return basis, ratio, S.astype(np.float32)


def reconstruct_window_torch(u_start, coeff, basis, enforce_mean=False):
    R = torch.einsum("bxm,km->bxk", coeff, basis)
    Uf = u_start.unsqueeze(-1) + R
    if enforce_mean:
        mean0 = u_start.mean(dim=1, keepdim=True).unsqueeze(-1)
        Uf = Uf + mean0 - Uf.mean(dim=1, keepdim=True)
    return Uf


def enforce_mean_future(U_future, u_start):
    mean0 = u_start.mean(dim=1, keepdim=True).unsqueeze(-1)
    return U_future + mean0 - U_future.mean(dim=1, keepdim=True)


# ============================================================
# 7. DATASET / NORMALIZER
# ============================================================

class WindowDataset1D(Dataset):
    def __init__(self, u_start, U_future):
        self.u_start = torch.tensor(u_start, dtype=torch.float32)
        self.U_future = torch.tensor(U_future, dtype=torch.float32)

    def __len__(self):
        return self.u_start.shape[0]

    def __getitem__(self, i):
        return {"u_start": self.u_start[i], "U_future": self.U_future[i]}


class FullTrajectoryDataset1D(Dataset):
    def __init__(self, U):
        U = np.asarray(U, dtype=np.float32)
        if U.ndim != 3:
            raise ValueError(f"Expected U with shape (N,Nx,Nt). Got {U.shape}")
        if U.shape[-1] < 2:
            raise ValueError("Need at least two saved time snapshots for full-trajectory prediction.")
        self.u0 = torch.tensor(U[:, :, 0], dtype=torch.float32)
        self.U_future = torch.tensor(U[:, :, 1:], dtype=torch.float32)

    def __len__(self):
        return self.u0.shape[0]

    def __getitem__(self, i):
        return {"u0": self.u0[i], "U_future": self.U_future[i]}


class GaussianNormalizer:
    def __init__(self, x, dims=None, eps=1.0e-6):
        self.eps = eps
        if dims is None:
            self.mean = x.mean()
            self.std = x.std()
        else:
            self.mean = x.mean(dim=dims, keepdim=True)
            self.std = x.std(dim=dims, keepdim=True)

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

    def to(self, device_):
        self.mean = self.mean.to(device_)
        self.std = self.std.to(device_)
        return self


# ============================================================
# 8. MODELS
# ============================================================

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x):
        super().__init__()
        self.modes_x = modes_x
        scale = 1.0 / np.sqrt(in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, dtype=torch.cfloat))

    @staticmethod
    def compl_mul1d(a, b):
        return torch.einsum("bix,iox->box", a, b)

    def forward(self, x):
        B, C, nx = x.shape
        x_ft = torch.fft.rfft(x, norm="ortho")
        out_ft = torch.zeros(B, self.weights.shape[1], nx // 2 + 1, dtype=torch.cfloat, device=x.device)
        mx = min(self.modes_x, nx // 2 + 1)
        out_ft[:, :, :mx] = self.compl_mul1d(x_ft[:, :, :mx], self.weights[:, :, :mx])
        return torch.fft.irfft(out_ft, n=nx, norm="ortho")


class LocalOperator1d(nn.Module):
    def __init__(self, width, radius):
        super().__init__()
        self.op = nn.Conv1d(width, width, kernel_size=2 * radius + 1, padding=radius, padding_mode="circular", bias=True)

    def forward(self, z):
        return self.op(z)


class FourierOperator1d(nn.Module):
    def __init__(self, width, modes_x):
        super().__init__()
        self.op = SpectralConv1d(width, width, modes_x)

    def forward(self, z):
        return self.op(z)


class AlternatingLayer1d(nn.Module):
    def __init__(self, width, modes_x, local_radius):
        super().__init__()
        self.L_l1 = LocalOperator1d(width, local_radius)
        self.K_l = FourierOperator1d(width, modes_x)
        self.L_l2 = LocalOperator1d(width, local_radius)
        self.sigma = nn.GELU()

    def forward(self, v_l):
        v_l_1 = v_l + self.sigma(self.L_l1(v_l))
        v_l_2 = v_l_1 + self.sigma(self.K_l(v_l_1))
        return v_l_2 + self.sigma(self.L_l2(v_l_2))


class AlternatingEncoder1d(nn.Module):
    def __init__(self, width, n_layers, modes_x, local_radius, x_grid):
        super().__init__()
        x = torch.tensor(x_grid, dtype=torch.float32)
        xn = (x - x.min()) / (x.max() - x.min() + 1.0e-12)
        grid = torch.stack([xn, torch.sin(2.0 * np.pi * xn), torch.cos(2.0 * np.pi * xn)], dim=-1)
        self.register_buffer("grid", grid.unsqueeze(0), persistent=False)
        self.input_proj = nn.Linear(4, width)
        self.blocks = nn.ModuleList([AlternatingLayer1d(width, modes_x, local_radius) for _ in range(n_layers)])

    def forward(self, u_start_norm):
        B, nx = u_start_norm.shape
        grid = self.grid.expand(B, -1, -1)
        z = self.input_proj(torch.cat([u_start_norm.unsqueeze(-1), grid], dim=-1))
        z = z.permute(0, 2, 1).contiguous()
        for block in self.blocks:
            z = block(z)
        return z.permute(0, 2, 1).contiguous()


class TBAWFNO1d(nn.Module):
    def __init__(self, n_modes, x_grid):
        super().__init__()
        self.encoder = AlternatingEncoder1d(WIDTH, N_LAYERS, MODES_X, LOCAL_RADIUS, x_grid)
        self.fc1 = nn.Linear(WIDTH, 128)
        self.fc2 = nn.Linear(128, n_modes)
        self.act = nn.GELU()

    def forward(self, u_start_norm):
        z = self.encoder(u_start_norm)
        return self.fc2(self.act(self.fc1(z))).contiguous()


class AWFNO1d(nn.Module):
    def __init__(self, window_k, x_grid):
        super().__init__()
        self.encoder = AlternatingEncoder1d(WIDTH, N_LAYERS, MODES_X, LOCAL_RADIUS, x_grid)
        self.fc1 = nn.Linear(WIDTH, 128)
        self.fc2 = nn.Linear(128, window_k)
        self.act = nn.GELU()

    def forward(self, u_start_norm):
        z = self.encoder(u_start_norm)
        return self.fc2(self.act(self.fc1(z))).contiguous()


class WFNO1d(nn.Module):
    def __init__(self, window_k, x_grid):
        super().__init__()
        x = torch.tensor(x_grid, dtype=torch.float32)
        xn = (x - x.min()) / (x.max() - x.min() + 1.0e-12)
        grid = torch.stack([xn, torch.sin(2.0 * np.pi * xn), torch.cos(2.0 * np.pi * xn)], dim=-1)
        self.register_buffer("grid", grid.unsqueeze(0), persistent=False)
        self.fc0 = nn.Linear(4, WIDTH)
        self.convs = nn.ModuleList([SpectralConv1d(WIDTH, WIDTH, MODES_X) for _ in range(N_LAYERS)])
        self.ws = nn.ModuleList([nn.Conv1d(WIDTH, WIDTH, 1) for _ in range(N_LAYERS)])
        self.fc1 = nn.Linear(WIDTH, 128)
        self.fc2 = nn.Linear(128, window_k)

    def forward(self, u_start_norm):
        B, nx = u_start_norm.shape
        grid = self.grid.expand(B, -1, -1)
        z = self.fc0(torch.cat([u_start_norm.unsqueeze(-1), grid], dim=-1))
        z = z.permute(0, 2, 1).contiguous()
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            z = conv(z) + w(z)
            if i != len(self.convs) - 1:
                z = F.gelu(z)
        z = z.permute(0, 2, 1).contiguous()
        return self.fc2(F.gelu(self.fc1(z))).contiguous()


class OrderedResidualBlock1d(nn.Module):
    """
    Controlled local/Fourier ablation block.

    Pattern symbols:
        L: learned finite-range circular convolution
        F: learned truncated Fourier convolution
        I: exact identity/no learned substage

    Every learned substage uses the same residual update
        z <- z + GELU(op(z)).
    Therefore LFL, LLF, and FLL contain exactly the same learned operators and
    parameter count; only their ordering changes.
    """
    VALID_PATTERNS = {"LFL", "LIL", "IFI", "LLF", "FLL"}

    def __init__(self, width, modes_x, local_radius, pattern):
        super().__init__()
        pattern = str(pattern).upper()
        if pattern not in self.VALID_PATTERNS:
            raise ValueError(
                f"Unknown pattern {pattern!r}. "
                f"Expected one of {sorted(self.VALID_PATTERNS)}."
            )
        self.pattern = pattern
        ops = []
        for symbol in pattern:
            if symbol == "L":
                ops.append(LocalOperator1d(width, local_radius))
            elif symbol == "F":
                ops.append(FourierOperator1d(width, modes_x))
            else:
                ops.append(nn.Identity())
        self.ops = nn.ModuleList(ops)
        self.sigma = nn.GELU()

    def forward(self, z):
        for symbol, op in zip(self.pattern, self.ops):
            if symbol != "I":
                z = z + self.sigma(op(z))
        return z


class ReviewerAlternatingEncoder1d(nn.Module):
    def __init__(
        self,
        width,
        n_layers,
        modes_x,
        local_radius,
        pattern,
        x_grid,
    ):
        super().__init__()
        x = torch.tensor(x_grid, dtype=torch.float32)
        xn = (x - x.min()) / (x.max() - x.min() + 1.0e-12)
        grid = torch.stack(
            [xn, torch.sin(2.0 * np.pi * xn), torch.cos(2.0 * np.pi * xn)],
            dim=-1,
        )
        self.register_buffer("grid", grid.unsqueeze(0), persistent=False)
        self.input_proj = nn.Linear(4, width)
        self.blocks = nn.ModuleList(
            [
                OrderedResidualBlock1d(
                    width,
                    modes_x,
                    local_radius,
                    pattern,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, u_start_norm):
        B, _ = u_start_norm.shape
        grid = self.grid.expand(B, -1, -1)
        z = self.input_proj(
            torch.cat([u_start_norm.unsqueeze(-1), grid], dim=-1)
        )
        z = z.permute(0, 2, 1).contiguous()
        for block in self.blocks:
            z = block(z)
        return z.permute(0, 2, 1).contiguous()


class ReviewerAlternatingModel1d(nn.Module):
    def __init__(
        self,
        window_k,
        x_grid,
        pattern="LFL",
        width=32,
        n_layers=4,
        modes_x=16,
        local_radius=3,
    ):
        super().__init__()
        self.window_k = int(window_k)
        self.pattern = str(pattern).upper()
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.modes_x = int(modes_x)
        self.local_radius = int(local_radius)
        self.encoder = ReviewerAlternatingEncoder1d(
            self.width,
            self.n_layers,
            self.modes_x,
            self.local_radius,
            self.pattern,
            x_grid,
        )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.window_k)
        self.act = nn.GELU()

    def forward(self, u_start_norm):
        z = self.encoder(u_start_norm)
        return self.fc2(self.act(self.fc1(z))).contiguous()


class ReviewerWFNO1d(nn.Module):
    """Standard windowed FNO with explicit width/mode/depth controls."""

    def __init__(
        self,
        window_k,
        x_grid,
        width=32,
        n_layers=4,
        modes_x=16,
    ):
        super().__init__()
        self.window_k = int(window_k)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.modes_x = int(modes_x)
        x = torch.tensor(x_grid, dtype=torch.float32)
        xn = (x - x.min()) / (x.max() - x.min() + 1.0e-12)
        grid = torch.stack(
            [xn, torch.sin(2.0 * np.pi * xn), torch.cos(2.0 * np.pi * xn)],
            dim=-1,
        )
        self.register_buffer("grid", grid.unsqueeze(0), persistent=False)
        self.fc0 = nn.Linear(4, self.width)
        self.convs = nn.ModuleList(
            [
                SpectralConv1d(self.width, self.width, self.modes_x)
                for _ in range(self.n_layers)
            ]
        )
        self.ws = nn.ModuleList(
            [
                nn.Conv1d(self.width, self.width, 1)
                for _ in range(self.n_layers)
            ]
        )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.window_k)

    def forward(self, u_start_norm):
        B, _ = u_start_norm.shape
        grid = self.grid.expand(B, -1, -1)
        z = self.fc0(
            torch.cat([u_start_norm.unsqueeze(-1), grid], dim=-1)
        )
        z = z.permute(0, 2, 1).contiguous()
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            z = conv(z) + w(z)
            if i != len(self.convs) - 1:
                z = F.gelu(z)
        z = z.permute(0, 2, 1).contiguous()
        return self.fc2(F.gelu(self.fc1(z))).contiguous()


def count_real_scalar_params(model):
    """
    Count real scalar degrees of freedom.

    torch.numel() counts one complex element as one parameter. This auxiliary
    count reports a complex parameter as two real scalars.
    """
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        total += p.numel() * (2 if p.is_complex() else 1)
    return int(total)


class NeuralOperatorFNOFull1d(nn.Module):
    def __init__(self, nt, x_grid):
        super().__init__()
        if not USE_LIBRARY_FNO:
            raise RuntimeError("USE_LIBRARY_FNO=False, but NeuralOperatorFNOFull1d was requested.")
        try:
            from neuralop.models import FNO as NeuralOpFNO
        except Exception as exc:
            raise ImportError("Install NeuralOperator with: pip install neuraloperator") from exc

        self.nt = int(nt)
        x = torch.tensor(x_grid, dtype=torch.float32)
        xn = (x - x.min()) / (x.max() - x.min() + 1.0e-12)
        grid = torch.stack([xn, torch.sin(2.0 * np.pi * xn), torch.cos(2.0 * np.pi * xn)], dim=0)
        self.register_buffer("grid", grid.unsqueeze(0), persistent=False)

        kwargs = dict(
            n_modes=(min(MODES_X, len(x_grid) // 2 + 1),),
            in_channels=4,
            out_channels=self.nt - 1,
            hidden_channels=WIDTH,
            n_layers=N_LAYERS,
        )
        try:
            self.fno = NeuralOpFNO(**kwargs)
        except TypeError:
            kwargs.pop("n_layers", None)
            self.fno = NeuralOpFNO(**kwargs)

    def forward(self, u0_norm):
        B, nx = u0_norm.shape
        grid = self.grid.expand(B, -1, -1)
        x_in = torch.cat([u0_norm.unsqueeze(1), grid], dim=1)
        out = self.fno(x_in)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise RuntimeError(f"Expected neuralop FNO output with shape (B,C,Nx). Got {tuple(out.shape)}")
        if out.shape[-1] == nx:
            return out.permute(0, 2, 1).contiguous()
        if out.shape[1] == nx:
            return out.contiguous()
        raise RuntimeError(f"Cannot interpret neuralop FNO output shape {tuple(out.shape)} for nx={nx}")


def make_tb_awfno_predict(model, u_normer, basis_torch):
    def predict(u_start):
        coeff = model(u_normer.encode(u_start))
        Uf = reconstruct_window_torch(u_start, coeff, basis_torch, ENFORCE_MEAN)
        return coeff, Uf
    return predict


def make_direct_predict(model, u_normer, future_normer, enforce_mean=False):
    def predict(u_start):
        Uf = future_normer.decode(model(u_normer.encode(u_start)))
        if enforce_mean:
            Uf = enforce_mean_future(Uf, u_start)
        return Uf
    return predict


def make_full_fno_predict(model, u0_normer, full_future_normer):
    def predict(u0):
        Uf_norm = model(u0_normer.encode(u0))
        return full_future_normer.decode(Uf_norm)
    return predict


# ============================================================
# 9. LOSSES / TRAINING
# ============================================================

def data_loss_fn(pred, true):
    return F.mse_loss(pred, true)


def tb_awfno_loss(Uf_pred, Uf_true):
    return data_loss_fn(Uf_pred, Uf_true) + FINAL_LOSS_WEIGHT * data_loss_fn(Uf_pred[:, :, -1], Uf_true[:, :, -1])


def awfno_loss(Uf_pred, Uf_true):
    return data_loss_fn(Uf_pred, Uf_true) + FINAL_LOSS_WEIGHT * data_loss_fn(Uf_pred[:, :, -1], Uf_true[:, :, -1])


def wfno_loss(Uf_pred, Uf_true):
    return F.mse_loss(Uf_pred, Uf_true)


def train_epoch_tb_awfno(model, loader, opt, predict):
    model.train()
    total_loss = 0.0
    total_l2 = 0.0
    n = 0
    for batch in loader:
        u0 = to_device(batch["u_start"])
        Uf_true = to_device(batch["U_future"])
        opt.zero_grad(set_to_none=True)
        _, Uf_pred = predict(u0)
        loss = tb_awfno_loss(Uf_pred, Uf_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = u0.shape[0]
        total_loss += loss.item() * bs
        total_l2 += abs_l2_window_batch(Uf_pred.detach(), Uf_true).mean().item() * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "abs_l2": total_l2 / max(n, 1)}


@torch.no_grad()
def eval_epoch_tb_awfno(model, loader, predict):
    model.eval()
    total_loss = 0.0
    total_l2 = 0.0
    total_final = 0.0
    n = 0
    for batch in loader:
        u0 = to_device(batch["u_start"])
        Uf_true = to_device(batch["U_future"])
        _, Uf_pred = predict(u0)
        loss = tb_awfno_loss(Uf_pred, Uf_true)
        bs = u0.shape[0]
        total_loss += loss.item() * bs
        total_l2 += abs_l2_window_batch(Uf_pred, Uf_true).mean().item() * bs
        total_final += abs_l2_field_batch(Uf_pred[:, :, -1], Uf_true[:, :, -1]).mean().item() * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "abs_l2": total_l2 / max(n, 1), "final_l2": total_final / max(n, 1)}


def fit_tb_awfno_model(model, train_loader, val_loader, predict):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_val = float("inf")
    best_state = None
    hist = {"train_loss": [], "train_l2": [], "val_loss": [], "val_l2": [], "val_final_l2": []}
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tr = train_epoch_tb_awfno(model, train_loader, opt, predict)
        va = eval_epoch_tb_awfno(model, val_loader, predict)
        sched.step()
        hist["train_loss"].append(tr["loss"])
        hist["train_l2"].append(tr["abs_l2"])
        hist["val_loss"].append(va["loss"])
        hist["val_l2"].append(va["abs_l2"])
        hist["val_final_l2"].append(va["final_l2"])
        if va["abs_l2"] < best_val:
            best_val = va["abs_l2"]
            best_state = clean_state_dict(model.state_dict())
        if ep == 1 or ep % 10 == 0:
            print(f"[TB-AWFNO] {ep:04d}/{EPOCHS} | train loss {tr['loss']:.6e} | train L2 {tr['abs_l2']:.6e} | val loss {va['loss']:.6e} | val L2 {va['abs_l2']:.6e} | val final {va['final_l2']:.6e}")
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return {"model": model, "best_val_l2": best_val, "train_time_sec": time.time() - t0, "history": hist}


def train_epoch_direct(model, loader, opt, predict, kind):
    model.train()
    total_loss = 0.0
    total_l2 = 0.0
    n = 0
    for batch in loader:
        u0 = to_device(batch["u_start"])
        Uf_true = to_device(batch["U_future"])
        opt.zero_grad(set_to_none=True)
        Uf_pred = predict(u0)
        loss = awfno_loss(Uf_pred, Uf_true) if kind == "awfno" else wfno_loss(Uf_pred, Uf_true)
        loss.backward()
        if kind == "awfno":
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = u0.shape[0]
        total_loss += loss.item() * bs
        total_l2 += abs_l2_window_batch(Uf_pred.detach(), Uf_true).mean().item() * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "abs_l2": total_l2 / max(n, 1)}


@torch.no_grad()
def eval_epoch_direct(model, loader, predict, kind):
    model.eval()
    total_loss = 0.0
    total_l2 = 0.0
    total_final = 0.0
    n = 0
    for batch in loader:
        u0 = to_device(batch["u_start"])
        Uf_true = to_device(batch["U_future"])
        Uf_pred = predict(u0)
        loss = awfno_loss(Uf_pred, Uf_true) if kind == "awfno" else wfno_loss(Uf_pred, Uf_true)
        bs = u0.shape[0]
        total_loss += loss.item() * bs
        total_l2 += abs_l2_window_batch(Uf_pred, Uf_true).mean().item() * bs
        total_final += abs_l2_field_batch(Uf_pred[:, :, -1], Uf_true[:, :, -1]).mean().item() * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "abs_l2": total_l2 / max(n, 1), "final_l2": total_final / max(n, 1)}


def fit_direct_model(model, train_loader, val_loader, predict, name, kind):
    if kind == "awfno":
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(EPOCHS // 3, 1), gamma=0.5)
    best_val = float("inf")
    best_state = None
    hist = {"train_loss": [], "train_l2": [], "val_loss": [], "val_l2": [], "val_final_l2": []}
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tr = train_epoch_direct(model, train_loader, opt, predict, kind)
        va = eval_epoch_direct(model, val_loader, predict, kind)
        sched.step()
        hist["train_loss"].append(tr["loss"])
        hist["train_l2"].append(tr["abs_l2"])
        hist["val_loss"].append(va["loss"])
        hist["val_l2"].append(va["abs_l2"])
        hist["val_final_l2"].append(va["final_l2"])
        if va["abs_l2"] < best_val:
            best_val = va["abs_l2"]
            best_state = clean_state_dict(model.state_dict())
        if ep == 1 or ep % 10 == 0:
            print(f"[{name}] {ep:04d}/{EPOCHS} | train loss {tr['loss']:.6e} | train L2 {tr['abs_l2']:.6e} | val loss {va['loss']:.6e} | val L2 {va['abs_l2']:.6e} | val final {va['final_l2']:.6e}")
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return {"model": model, "best_val_l2": best_val, "train_time_sec": time.time() - t0, "history": hist}


def train_epoch_full_fno(model, loader, opt, predict):
    model.train()
    total_loss = 0.0
    total_l2 = 0.0
    n = 0
    for batch in loader:
        u0 = to_device(batch["u0"])
        Uf_true = to_device(batch["U_future"])
        opt.zero_grad(set_to_none=True)
        Uf_pred = predict(u0)
        loss = F.mse_loss(Uf_pred, Uf_true) + FINAL_LOSS_WEIGHT * F.mse_loss(Uf_pred[:, :, -1], Uf_true[:, :, -1])
        loss.backward()
        opt.step()
        bs = u0.shape[0]
        total_loss += loss.item() * bs
        total_l2 += abs_l2_window_batch(Uf_pred.detach(), Uf_true).mean().item() * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "abs_l2": total_l2 / max(n, 1)}


@torch.no_grad()
def eval_epoch_full_fno(model, loader, predict):
    model.eval()
    total_loss = 0.0
    total_l2 = 0.0
    total_final = 0.0
    n = 0
    for batch in loader:
        u0 = to_device(batch["u0"])
        Uf_true = to_device(batch["U_future"])
        Uf_pred = predict(u0)
        loss = F.mse_loss(Uf_pred, Uf_true) + FINAL_LOSS_WEIGHT * F.mse_loss(Uf_pred[:, :, -1], Uf_true[:, :, -1])
        bs = u0.shape[0]
        total_loss += loss.item() * bs
        total_l2 += abs_l2_window_batch(Uf_pred, Uf_true).mean().item() * bs
        total_final += abs_l2_field_batch(Uf_pred[:, :, -1], Uf_true[:, :, -1]).mean().item() * bs
        n += bs
    return {"loss": total_loss / max(n, 1), "abs_l2": total_l2 / max(n, 1), "final_l2": total_final / max(n, 1)}


def fit_full_fno_model(model, train_loader, val_loader, predict):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(EPOCHS // 3, 1), gamma=0.5)
    best_val = float("inf")
    best_state = None
    hist = {"train_loss": [], "train_l2": [], "val_loss": [], "val_l2": [], "val_final_l2": []}
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tr = train_epoch_full_fno(model, train_loader, opt, predict)
        va = eval_epoch_full_fno(model, val_loader, predict)
        sched.step()
        hist["train_loss"].append(tr["loss"])
        hist["train_l2"].append(tr["abs_l2"])
        hist["val_loss"].append(va["loss"])
        hist["val_l2"].append(va["abs_l2"])
        hist["val_final_l2"].append(va["final_l2"])
        if va["abs_l2"] < best_val:
            best_val = va["abs_l2"]
            best_state = clean_state_dict(model.state_dict())
        if ep == 1 or ep % 10 == 0:
            print(f"[FNO] {ep:04d}/{EPOCHS} | train loss {tr['loss']:.6e} | train L2 {tr['abs_l2']:.6e} | val loss {va['loss']:.6e} | val L2 {va['abs_l2']:.6e} | val final {va['final_l2']:.6e}")
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return {"model": model, "best_val_l2": best_val, "train_time_sec": time.time() - t0, "history": hist}


# ============================================================
# 10. ROLLOUT / DIRECT PREDICTION
# ============================================================

@torch.no_grad()
def autoregressive_rollout_tb_awfno(model, u0_test_np, nt, predict, print_diagnostics=True):
    model.eval()
    N, nx = u0_test_np.shape
    U_roll = np.zeros((N, nx, nt), dtype=np.float32)
    u_current = torch.tensor(u0_test_np, dtype=torch.float32, device=device)
    U_roll[:, :, 0] = u0_test_np
    cursor = 0
    block = 0
    print("\nAutoregressive rollout: TB-AWFNO")
    while cursor < nt - 1:
        _, Uf_pred = predict(u_current)
        use_k = min(WINDOW_K, nt - 1 - cursor)
        U_roll[:, :, cursor + 1:cursor + 1 + use_k] = Uf_pred[:, :, :use_k].detach().cpu().numpy()
        u_next = Uf_pred[:, :, use_k - 1].detach()
        cursor_next = cursor + use_k
        if print_diagnostics:
            step_norm = abs_l2_field_batch(u_next, u_current).mean().item()
            print(f"TB-AWFNO block {block:03d} | {cursor:04d}->{cursor_next:04d} | terminal increment L2 = {step_norm:.6e}")
        u_current = u_next
        cursor = cursor_next
        block += 1
    return U_roll


@torch.no_grad()
def autoregressive_rollout_direct(model, u0_test_np, nt, predict, name, print_diagnostics=True):
    model.eval()
    N, nx = u0_test_np.shape
    U_roll = np.zeros((N, nx, nt), dtype=np.float32)
    u_current = torch.tensor(u0_test_np, dtype=torch.float32, device=device)
    U_roll[:, :, 0] = u0_test_np
    cursor = 0
    block = 0
    print(f"\nAutoregressive rollout: {name}")
    while cursor < nt - 1:
        Uf_pred = predict(u_current)
        use_k = min(WINDOW_K, nt - 1 - cursor)
        U_roll[:, :, cursor + 1:cursor + 1 + use_k] = Uf_pred[:, :, :use_k].detach().cpu().numpy()
        u_next = Uf_pred[:, :, use_k - 1].detach()
        cursor_next = cursor + use_k
        if print_diagnostics:
            step_norm = abs_l2_field_batch(u_next, u_current).mean().item()
            print(f"{name} block {block:03d} | {cursor:04d}->{cursor_next:04d} | terminal increment L2 = {step_norm:.6e}")
        u_current = u_next
        cursor = cursor_next
        block += 1
    return U_roll


@torch.no_grad()
def direct_full_trajectory_prediction(model, u0_test_np, nt, predict, name="FNO", batch_size=None):
    model.eval()
    batch_size = batch_size or BATCH_SIZE
    N, nx = u0_test_np.shape
    U_pred = np.zeros((N, nx, nt), dtype=np.float32)
    U_pred[:, :, 0] = u0_test_np
    print(f"\nDirect full-trajectory prediction: {name}")
    for i0 in range(0, N, batch_size):
        i1 = min(i0 + batch_size, N)
        u0 = torch.tensor(u0_test_np[i0:i1], dtype=torch.float32, device=device)
        Uf_pred = predict(u0)
        U_pred[i0:i1, :, 1:] = Uf_pred.detach().cpu().numpy()
    return U_pred


def print_terminal_errors(U_roll, U_true, name):
    terminal_l2 = l2_per_timestep_np(U_roll, U_true)[:, -1]
    print(f"{name} final-time L2: mean={terminal_l2.mean():.6e}, median={np.median(terminal_l2):.6e}, max={terminal_l2.max():.6e}")


# ============================================================
# 11. PLOTS / PRINTING
# ============================================================

def get_plot_time_indices(nt):
    out = []
    for item in PLOT_TIME_INDICES_MANUAL:
        out.append(nt - 1 if item == "final" else int(item))
    clean = []
    for k in out:
        if 0 <= k < nt and k not in clean:
            clean.append(k)
    return clean or [0, nt // 2, nt - 1]


def plot_training_curves(hist, name):
    if not SAVE_PLOTS or not MAKE_TRAINING_CURVE_PLOTS:
        return
    plt.figure(figsize=(8, 5))
    plt.semilogy(hist["train_loss"], label="train loss")
    plt.semilogy(hist["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / f"{safe_name(name)}_loss_curves_{BASE_TAG}.png"
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)

    plt.figure(figsize=(8, 5))
    plt.semilogy(hist["train_l2"], label="train L2")
    plt.semilogy(hist["val_l2"], label="val L2")
    plt.semilogy(hist["val_final_l2"], label="val final L2")
    plt.xlabel("epoch")
    plt.ylabel("physical L2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PLOT_DIR / f"{safe_name(name)}_l2_curves_{BASE_TAG}.png"
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)


def plot_temporal_basis(basis_np, singular_values):
    if not SAVE_PLOTS:
        return
    out_dir = PLOT_DIR / "temporal_basis"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for m in range(basis_np.shape[1]):
        plt.plot(np.arange(1, WINDOW_K + 1), basis_np[:, m], label=f"mode {m + 1}")
    plt.xlabel("local step")
    plt.ylabel("basis value")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"temporal_basis_{BASE_TAG}.png"
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)

    plt.figure(figsize=(8, 5))
    plt.semilogy(singular_values, marker="o")
    plt.xlabel("index")
    plt.ylabel("singular value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"temporal_basis_singular_values_{BASE_TAG}.png"
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)


def plot_l2_per_time_compare_all(tb, aw, wf, fn):
    if not SAVE_PLOTS or not MAKE_L2_PER_TIME_PLOT:
        return
    out_dir = PLOT_DIR / "l2_error_per_timestep"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for stats, label in [(tb, "TB-AWFNO"), (aw, "AWFNO"), (wf, "WFNO"), (fn, "FNO")]:
        plt.plot(t_grid, stats["mean_t"], label=f"{label} mean", linewidth=2)
        plt.plot(t_grid, stats["median_t"], "--", label=f"{label} median")
    plt.xlabel("time")
    plt.ylabel(r"$E_{L^2}$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / f"all_tests_mean_median_L2_error_{BASE_TAG}.png"
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)


def plot_per_test_l2_curves(l2_dict):
    if not SAVE_PLOTS or not MAKE_PER_TEST_L2_PLOTS:
        return
    out_dir = PLOT_DIR / "l2_error_per_timestep"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_available = next(iter(l2_dict.values())).shape[0]
    n_plot = n_available if MAX_TEST_CASES_TO_PLOT is None else min(MAX_TEST_CASES_TO_PLOT, n_available)
    for i in range(n_plot):
        plt.figure(figsize=(8, 5))
        for name, l2_t in l2_dict.items():
            plt.plot(t_grid, l2_t[i], label=name, linewidth=2)
        plt.xlabel("time")
        plt.ylabel(r"$E_{L^2}$")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"test_{i:03d}_L2_error_per_timestep.png"
        plt.savefig(path, dpi=PLOT_DPI)
        plt.close()
        print("Saved:", path)


def plot_solution_curves(U_true, rollouts):
    if not SAVE_PLOTS or not MAKE_SOLUTION_PLOTS:
        return
    out_dir = PLOT_DIR / "selected_test_case_solutions"
    out_dir.mkdir(parents=True, exist_ok=True)
    time_ids = get_plot_time_indices(U_true.shape[-1])
    n_available = U_true.shape[0]
    n_plot = n_available if MAX_TEST_CASES_TO_PLOT is None else min(MAX_TEST_CASES_TO_PLOT, n_available)
    for i in range(n_plot):
        fig, axes = plt.subplots(len(time_ids), 1, figsize=(9, 3 * len(time_ids)), squeeze=False, constrained_layout=True)
        for row, tidx in enumerate(time_ids):
            ax = axes[row, 0]
            ax.plot(x_grid_np, U_true[i, :, tidx], label="true", linewidth=2.4)
            for name, arr in rollouts.items():
                ax.plot(x_grid_np, arr[i, :, tidx], label=name, linewidth=1.7)
            ax.set_title(f"test {i}, time index {tidx}")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.grid(True, alpha=0.3)
            ax.legend()
        path = out_dir / f"test_{i:03d}_solution_curves.png"
        plt.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)
        print("Saved:", path)


def plot_one_ranked_case(idx, rank_name, U_true, rollouts, l2_dict, full_l2_reference, out_dir):
    time_ids = get_plot_time_indices(U_true.shape[-1])
    n_rows = len(time_ids) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.0 * n_rows), squeeze=False, constrained_layout=True)
    for row, tidx in enumerate(time_ids):
        ax = axes[row, 0]
        ax.plot(x_grid_np, U_true[idx, :, tidx], label="true", linewidth=2.5)
        for name, arr in rollouts.items():
            ax.plot(x_grid_np, arr[idx, :, tidx], label=name, linewidth=1.7)
        ax.set_title(f"{rank_name}: test {idx}, time index {tidx}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(True, alpha=0.3)
        ax.legend()
    ax = axes[-1, 0]
    for name, l2_t in l2_dict.items():
        ax.plot(t_grid, l2_t[idx], label=name, linewidth=2)
    ax.set_title(f"Per-time L2 error | ranked by TB-AWFNO full L2 = {full_l2_reference[idx]:.6e}")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$E_{L^2}$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = out_dir / f"{safe_name(rank_name)}_test_{idx:03d}.png"
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    print("Saved:", path)
    return str(path)


def plot_best_worst_cases(U_true, rollouts, l2_dict, tb_stats, n_each=N_BEST_WORST):
    if not SAVE_PLOTS or not MAKE_BEST_WORST_PLOTS:
        return []
    out_dir = PLOT_DIR / "best_worst_test_cases"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_l2 = np.asarray(tb_stats["all_full_l2"])
    order = np.argsort(full_l2)
    best = order[:n_each]
    worst = order[-n_each:][::-1]
    saved = []
    ranking = {
        "ranking_model": "TB-AWFNO",
        "ranking_metric": "full_trajectory_absolute_L2",
        "best_indices": best.astype(int).tolist(),
        "best_values": full_l2[best].astype(float).tolist(),
        "worst_indices": worst.astype(int).tolist(),
        "worst_values": full_l2[worst].astype(float).tolist(),
    }
    for k, idx in enumerate(best, start=1):
        saved.append(plot_one_ranked_case(int(idx), f"best_{k}", U_true, rollouts, l2_dict, full_l2, out_dir))
    for k, idx in enumerate(worst, start=1):
        saved.append(plot_one_ranked_case(int(idx), f"worst_{k}", U_true, rollouts, l2_dict, full_l2, out_dir))
    ranking_path = out_dir / f"best_worst_ranking_{BASE_TAG}.json"
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2)
    print("Saved:", ranking_path)
    return saved


def print_l2_results(tb, aw, wf, fn, train_times=None):
    print("\n================ ABSOLUTE PHYSICAL L2 TEST RESULTS ================")
    print("Full trajectory L2: trapezoidal time rule over saved snapshots, including t=0")
    print("No relative normalization / no denominator is used.")
    train_times = train_times or {"tb": None, "aw": None, "wf": None, "fn": None}
    def mt(k):
        return "" if train_times.get(k) is None else f" | train time {train_times[k]:.2f}s"
    for label, stats, key in [("TB-AWFNO", tb, "tb"), ("AWFNO", aw, "aw"), ("WFNO", wf, "wf"), ("FNO", fn, "fn")]:
        print(f"{label:<8} | L2(full) mean {stats['full_l2_mean']:.6e} | std {stats['full_l2_std']:.6e} | median {stats['full_l2_median']:.6e} | L2(final) mean {stats['final_l2_mean']:.6e} | final median {stats['final_l2_median']:.6e}{mt(key)}")
    scores = {"TB-AWFNO": tb["full_l2_mean"], "AWFNO": aw["full_l2_mean"], "WFNO": wf["full_l2_mean"], "FNO": fn["full_l2_mean"]}
    winner = min(scores, key=scores.get)
    gain_aw = 100.0 * (aw["full_l2_mean"] - tb["full_l2_mean"]) / (aw["full_l2_mean"] + 1.0e-30)
    gain_wf = 100.0 * (wf["full_l2_mean"] - tb["full_l2_mean"]) / (wf["full_l2_mean"] + 1.0e-30)
    gain_fn = 100.0 * (fn["full_l2_mean"] - tb["full_l2_mean"]) / (fn["full_l2_mean"] + 1.0e-30)
    print(f"\nWinner by mean full-trajectory absolute physical L2: {winner}")
    print("\nImprovement of TB-AWFNO:")
    print(f"  vs AWFNO | full L2: {gain_aw:.2f}%")
    print(f"  vs WFNO  | full L2: {gain_wf:.2f}%")
    print(f"  vs FNO   | full L2: {gain_fn:.2f}%")
    print("===============================================================\n")
    return winner, gain_aw, gain_wf, gain_fn


# ============================================================
# 12. SAMPLE-SWEEP HELPERS
# ============================================================

def make_base_tag_for_sample_size(n_train):
    data_signature = {
        "data_path_name": DATA_PATH.name,
        "data_path_exists": DATA_PATH.exists(),
        "data_path_size": DATA_PATH.stat().st_size if DATA_PATH.exists() else None,
    }
    return short_hash({
        "seed": SEED,
        "split_mode": SPLIT_MODE,
        "split_seed": SPLIT_SEED,
        "data_signature": data_signature,
        "all_sample_sizes": ALL_SAMPLE_SIZES,
        "N_train": n_train,
        "N_val": N_VAL,
        "N_test": N_TEST,
        "Nx": Nx,
        "Nt": Nt,
        "WINDOW_K": WINDOW_K,
        "WINDOW_STRIDE": WINDOW_STRIDE,
        "N_TIME_MODES": N_TIME_MODES,
        "WIDTH": WIDTH,
        "N_LAYERS": N_LAYERS,
        "MODES_X": MODES_X,
        "LOCAL_RADIUS": LOCAL_RADIUS,
        "FINAL_LOSS_WEIGHT": FINAL_LOSS_WEIGHT,
        "ENFORCE_MEAN": ENFORCE_MEAN,
        "model_names": ["TB-AWFNO", "AWFNO", "WFNO", "FNO"],
    })


def build_train_val_objects_for_sample_size(n_train, U_train_pool, U_val):
    U_train = U_train_pool[:n_train]
    train_starts = make_window_start_indices(Nt, WINDOW_K, WINDOW_STRIDE)
    val_starts = make_window_start_indices(Nt, WINDOW_K, WINDOW_K)

    print("\nWindow starts per trajectory")
    print("----------------------------")
    print("train:", len(train_starts))
    print("val  :", len(val_starts))
    print("test : 0  <-- no test windows are constructed")

    u_train_win, Uf_train_win = build_window_tensors(U_train, train_starts)
    u_val_win, Uf_val_win = build_window_tensors(U_val, val_starts)

    print("\nWindow tensor shapes")
    print("--------------------")
    print("u_train_win :", u_train_win.shape)
    print("Uf_train_win:", Uf_train_win.shape)
    print("u_val_win   :", u_val_win.shape)
    print("Uf_val_win  :", Uf_val_win.shape)

    basis_np, basis_energy_ratio, singular_values = compute_temporal_basis_from_windows(u_train_win, Uf_train_win, N_TIME_MODES)
    basis_torch = torch.tensor(basis_np, dtype=torch.float32, device=device)

    print("\nTemporal-basis coefficient supervision")
    print("--------------------------------------")
    print("Not used. Coefficients are internal model outputs.")

    train_ds = WindowDataset1D(u_train_win, Uf_train_win)
    val_ds = WindowDataset1D(u_val_win, Uf_val_win)
    full_train_ds = FullTrajectoryDataset1D(U_train)
    full_val_ds = FullTrajectoryDataset1D(U_val)

    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(SEED + n_train)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=cpu_gen, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)
    full_train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=cpu_gen, pin_memory=PIN_MEMORY)
    full_val_loader = DataLoader(full_val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY)

    batch0 = next(iter(train_loader))
    full_batch0 = next(iter(full_train_loader))
    print("\nBatch shapes")
    print("------------")
    print("window u_start      :", tuple(batch0["u_start"].shape))
    print("window U_future     :", tuple(batch0["U_future"].shape))
    print("full FNO u0         :", tuple(full_batch0["u0"].shape))
    print("full FNO U_future   :", tuple(full_batch0["U_future"].shape))

    u_normer = GaussianNormalizer(torch.tensor(u_train_win, dtype=torch.float32)).to(device)
    future_normer = GaussianNormalizer(torch.tensor(Uf_train_win, dtype=torch.float32)).to(device)
    full_u0_normer = GaussianNormalizer(torch.tensor(U_train[:, :, 0], dtype=torch.float32)).to(device)
    full_future_normer = GaussianNormalizer(torch.tensor(U_train[:, :, 1:], dtype=torch.float32)).to(device)

    return {
        "U_train": U_train,
        "u_train_win": u_train_win,
        "Uf_train_win": Uf_train_win,
        "u_val_win": u_val_win,
        "Uf_val_win": Uf_val_win,
        "basis_np": basis_np,
        "basis_torch": basis_torch,
        "basis_energy_ratio": basis_energy_ratio,
        "singular_values": singular_values,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "full_train_loader": full_train_loader,
        "full_val_loader": full_val_loader,
        "u_normer": u_normer,
        "future_normer": future_normer,
        "full_u0_normer": full_u0_normer,
        "full_future_normer": full_future_normer,
    }


def train_models_for_sample_size(n_train, U_train_pool, U_val):
    global BASE_TAG
    BASE_TAG = make_base_tag_for_sample_size(n_train)

    print("\n" + "=" * 80)
    print(f"TRAINING SAMPLE SIZE: N_train = {n_train}")
    print("=" * 80)
    print("BASE_TAG:", BASE_TAG)

    objects = build_train_val_objects_for_sample_size(n_train, U_train_pool, U_val)
    basis_torch = objects["basis_torch"]
    train_loader = objects["train_loader"]
    val_loader = objects["val_loader"]
    full_train_loader = objects["full_train_loader"]
    full_val_loader = objects["full_val_loader"]
    u_normer = objects["u_normer"]
    future_normer = objects["future_normer"]
    full_u0_normer = objects["full_u0_normer"]
    full_future_normer = objects["full_future_normer"]

    tb_model = TBAWFNO1d(N_TIME_MODES, x_grid_np).to(device)
    aw_model = AWFNO1d(WINDOW_K, x_grid_np).to(device)
    wf_model = WFNO1d(WINDOW_K, x_grid_np).to(device)
    fn_model = NeuralOperatorFNOFull1d(Nt, x_grid_np).to(device)

    tb_predict = make_tb_awfno_predict(tb_model, u_normer, basis_torch)
    aw_predict = make_direct_predict(aw_model, u_normer, future_normer, ENFORCE_MEAN)
    wf_predict = make_direct_predict(wf_model, u_normer, future_normer, False)
    fn_predict = make_full_fno_predict(fn_model, full_u0_normer, full_future_normer)

    print("\nModels")
    print("------")
    print("Alternating block used by TB-AWFNO and AWFNO:")
    print("  v_l_1      = v_l   + sigma(L_l1(v_l))")
    print("  v_l_2      = v_l_1 + sigma(K_l(v_l_1))")
    print("  v_l_plus_1 = v_l_2 + sigma(L_l2(v_l_2))")
    print("  L_l1,L_l2  = local finite-range circular Conv1d")
    print("  K_l        = truncated Fourier convolution")
    print("WFNO uses a standard FNO-style backbone under the same window protocol.")
    print("FNO uses neuralop.models.FNO and predicts the full future trajectory directly.")
    print(f"TB-AWFNO params: {count_params(tb_model):,}")
    print(f"AWFNO params   : {count_params(aw_model):,}")
    print(f"WFNO params    : {count_params(wf_model):,}")
    print(f"FNO params     : {count_params(fn_model):,}")

    tb_path = MODEL_DIR / f"N{n_train}_TB_AWFNO_{BASE_TAG}.pt"
    aw_path = MODEL_DIR / f"N{n_train}_AWFNO_{BASE_TAG}.pt"
    wf_path = MODEL_DIR / f"N{n_train}_WFNO_{BASE_TAG}.pt"
    fn_path = MODEL_DIR / f"N{n_train}_FNO_neuralop_full_{BASE_TAG}.pt"

    print("\nCheckpoint paths")
    print("----------------")
    print("MODEL_DIR:", MODEL_DIR)
    print("TB path:", tb_path, "exists:", tb_path.exists())
    print("AW path:", aw_path, "exists:", aw_path.exists())
    print("WF path:", wf_path, "exists:", wf_path.exists())
    print("FN path:", fn_path, "exists:", fn_path.exists())
    print("FORCE_RETRAIN:", FORCE_RETRAIN, type(FORCE_RETRAIN))

    if tb_path.exists() and not FORCE_RETRAIN:
        print("\nLoading cached TB-AWFNO:", tb_path)
        meta = load_model_checkpoint(tb_path, tb_model)
        tb_fit = {"model": tb_model, "best_val_l2": meta.get("best_val_l2", float("nan")), "train_time_sec": meta.get("train_time_sec", 0.0), "history": meta.get("history", None)}
    else:
        print("\n================ Training TB-AWFNO ================")
        tb_fit = fit_tb_awfno_model(tb_model, train_loader, val_loader, tb_predict)
        save_model_checkpoint(tb_path, tb_fit["model"], {"best_val_l2": tb_fit["best_val_l2"], "train_time_sec": tb_fit["train_time_sec"], "history": tb_fit["history"], "n_train": n_train, "base_tag": BASE_TAG})
    if tb_fit["history"] is not None:
        plot_training_curves(tb_fit["history"], f"N{n_train}_TB-AWFNO")

    if aw_path.exists() and not FORCE_RETRAIN:
        print("\nLoading cached AWFNO:", aw_path)
        meta = load_model_checkpoint(aw_path, aw_model)
        aw_fit = {"model": aw_model, "best_val_l2": meta.get("best_val_l2", float("nan")), "train_time_sec": meta.get("train_time_sec", 0.0), "history": meta.get("history", None)}
    else:
        print("\n================ Training AWFNO ================")
        aw_fit = fit_direct_model(aw_model, train_loader, val_loader, aw_predict, "AWFNO", "awfno")
        save_model_checkpoint(aw_path, aw_fit["model"], {"best_val_l2": aw_fit["best_val_l2"], "train_time_sec": aw_fit["train_time_sec"], "history": aw_fit["history"], "n_train": n_train, "base_tag": BASE_TAG})
    if aw_fit["history"] is not None:
        plot_training_curves(aw_fit["history"], f"N{n_train}_AWFNO")

    if wf_path.exists() and not FORCE_RETRAIN:
        print("\nLoading cached WFNO:", wf_path)
        meta = load_model_checkpoint(wf_path, wf_model)
        wf_fit = {"model": wf_model, "best_val_l2": meta.get("best_val_l2", float("nan")), "train_time_sec": meta.get("train_time_sec", 0.0), "history": meta.get("history", None)}
    else:
        print("\n================ Training WFNO ================")
        wf_fit = fit_direct_model(wf_model, train_loader, val_loader, wf_predict, "WFNO", "wfno")
        save_model_checkpoint(wf_path, wf_fit["model"], {"best_val_l2": wf_fit["best_val_l2"], "train_time_sec": wf_fit["train_time_sec"], "history": wf_fit["history"], "n_train": n_train, "base_tag": BASE_TAG})
    if wf_fit["history"] is not None:
        plot_training_curves(wf_fit["history"], f"N{n_train}_WFNO")

    if fn_path.exists() and not FORCE_RETRAIN:
        print("\nLoading cached FNO:", fn_path)
        meta = load_model_checkpoint(fn_path, fn_model)
        fn_fit = {"model": fn_model, "best_val_l2": meta.get("best_val_l2", float("nan")), "train_time_sec": meta.get("train_time_sec", 0.0), "history": meta.get("history", None)}
    else:
        print("\n================ Training FNO ================")
        fn_fit = fit_full_fno_model(fn_model, full_train_loader, full_val_loader, fn_predict)
        save_model_checkpoint(fn_path, fn_fit["model"], {"best_val_l2": fn_fit["best_val_l2"], "train_time_sec": fn_fit["train_time_sec"], "history": fn_fit["history"], "n_train": n_train, "base_tag": BASE_TAG, "library": "neuralop.models.FNO"})
    if fn_fit["history"] is not None:
        plot_training_curves(fn_fit["history"], f"N{n_train}_FNO")

    plot_temporal_basis(objects["basis_np"], objects["singular_values"])

    return {
        "n_train": n_train,
        "base_tag": BASE_TAG,
        "objects": objects,
        "tb_fit": tb_fit,
        "aw_fit": aw_fit,
        "wf_fit": wf_fit,
        "fn_fit": fn_fit,
        "tb_predict": tb_predict,
        "aw_predict": aw_predict,
        "wf_predict": wf_predict,
        "fn_predict": fn_predict,
    }


def evaluate_on_common_test_set(bundle, U_test, test_ids):
    global BASE_TAG
    n_train = bundle["n_train"]
    BASE_TAG = bundle["base_tag"]

    print("\n" + "=" * 80)
    print(f"COMMON TEST SET EVALUATION | N_train = {n_train}")
    print("=" * 80)
    print("Common test shape:", U_test.shape)
    print("Test ids first/last:", test_ids[:5], test_ids[-5:])
    print("Protocol: no test windows, no test temporal-basis coefficients.")
    print("Windowed models use autoregressive rollout; FNO predicts the full future directly.")

    u0_test = U_test[:, :, 0]
    nt = U_test.shape[-1]

    U_roll_tb = autoregressive_rollout_tb_awfno(bundle["tb_fit"]["model"], u0_test, nt, bundle["tb_predict"], True)
    U_roll_aw = autoregressive_rollout_direct(bundle["aw_fit"]["model"], u0_test, nt, bundle["aw_predict"], "AWFNO", True)
    U_roll_wf = autoregressive_rollout_direct(bundle["wf_fit"]["model"], u0_test, nt, bundle["wf_predict"], "WFNO", True)
    U_roll_fn = direct_full_trajectory_prediction(bundle["fn_fit"]["model"], u0_test, nt, bundle["fn_predict"], "FNO", BATCH_SIZE)

    print("\nTerminal errors")
    print("---------------")
    print_terminal_errors(U_roll_tb, U_test, "TB-AWFNO")
    print_terminal_errors(U_roll_aw, U_test, "AWFNO")
    print_terminal_errors(U_roll_wf, U_test, "WFNO")
    print_terminal_errors(U_roll_fn, U_test, "FNO")

    tb_stats = rollout_l2_metrics(U_roll_tb, U_test)
    aw_stats = rollout_l2_metrics(U_roll_aw, U_test)
    wf_stats = rollout_l2_metrics(U_roll_wf, U_test)
    fn_stats = rollout_l2_metrics(U_roll_fn, U_test)

    train_times = {
        "tb": bundle["tb_fit"]["train_time_sec"],
        "aw": bundle["aw_fit"]["train_time_sec"],
        "wf": bundle["wf_fit"]["train_time_sec"],
        "fn": bundle["fn_fit"]["train_time_sec"],
    }
    winner, gain_vs_aw, gain_vs_wf, gain_vs_fn = print_l2_results(tb_stats, aw_stats, wf_stats, fn_stats, train_times)

    rollouts = {"TB-AWFNO": U_roll_tb, "AWFNO": U_roll_aw, "WFNO": U_roll_wf, "FNO": U_roll_fn}
    l2_dict = {"TB-AWFNO": tb_stats["all_l2_t"], "AWFNO": aw_stats["all_l2_t"], "WFNO": wf_stats["all_l2_t"], "FNO": fn_stats["all_l2_t"]}

    plot_l2_per_time_compare_all(tb_stats, aw_stats, wf_stats, fn_stats)
    plot_per_test_l2_curves(l2_dict)
    plot_solution_curves(U_test, rollouts)
    best_worst_paths = plot_best_worst_cases(U_test, rollouts, l2_dict, tb_stats, n_each=N_BEST_WORST)

    rollout_path = RESULTS_DIR / f"N{n_train}_common_test_rollouts_{BASE_TAG}.npz"
    np.savez_compressed(
        rollout_path,
        U_roll_TB_AWFNO=U_roll_tb,
        U_roll_AWFNO=U_roll_aw,
        U_roll_WFNO=U_roll_wf,
        U_roll_FNO=U_roll_fn,
        U_true=U_test,
        x_grid=x_grid_np,
        t=t_grid,
        test_ids=test_ids,
        TB_AWFNO_l2_t=tb_stats["all_l2_t"],
        AWFNO_l2_t=aw_stats["all_l2_t"],
        WFNO_l2_t=wf_stats["all_l2_t"],
        FNO_l2_t=fn_stats["all_l2_t"],
        TB_AWFNO_full_l2=tb_stats["all_full_l2"],
        AWFNO_full_l2=aw_stats["all_full_l2"],
        WFNO_full_l2=wf_stats["all_full_l2"],
        FNO_full_l2=fn_stats["all_full_l2"],
        best_worst_plot_paths=np.array(best_worst_paths, dtype=object),
    )
    print("Saved rollout arrays:", rollout_path)

    row = {
        "N_train": n_train,
        "winner": winner,
        "TB_AWFNO_full_l2_mean": tb_stats["full_l2_mean"],
        "AWFNO_full_l2_mean": aw_stats["full_l2_mean"],
        "WFNO_full_l2_mean": wf_stats["full_l2_mean"],
        "FNO_full_l2_mean": fn_stats["full_l2_mean"],
        "TB_AWFNO_final_l2_mean": tb_stats["final_l2_mean"],
        "AWFNO_final_l2_mean": aw_stats["final_l2_mean"],
        "WFNO_final_l2_mean": wf_stats["final_l2_mean"],
        "FNO_final_l2_mean": fn_stats["final_l2_mean"],
        "gain_TB_AWFNO_vs_AWFNO_full_percent": gain_vs_aw,
        "gain_TB_AWFNO_vs_WFNO_full_percent": gain_vs_wf,
        "gain_TB_AWFNO_vs_FNO_full_percent": gain_vs_fn,
        "TB_AWFNO_best_val_l2": float(bundle["tb_fit"]["best_val_l2"]),
        "AWFNO_best_val_l2": float(bundle["aw_fit"]["best_val_l2"]),
        "WFNO_best_val_l2": float(bundle["wf_fit"]["best_val_l2"]),
        "FNO_best_val_l2": float(bundle["fn_fit"]["best_val_l2"]),
        "TB_AWFNO_train_time_sec": float(bundle["tb_fit"]["train_time_sec"]),
        "AWFNO_train_time_sec": float(bundle["aw_fit"]["train_time_sec"]),
        "WFNO_train_time_sec": float(bundle["wf_fit"]["train_time_sec"]),
        "FNO_train_time_sec": float(bundle["fn_fit"]["train_time_sec"]),
        "temporal_basis_energy_ratio_train_only": float(bundle["objects"]["basis_energy_ratio"]),
        "base_tag": BASE_TAG,
        "rollout_path": str(rollout_path),
        "best_worst_plot_paths": best_worst_paths,
    }

    summary_path = RESULTS_DIR / f"N{n_train}_summary_common_test_{BASE_TAG}.json"
    summary = {
        "task": EXPERIMENT_NAME,
        "data_path": str(DATA_PATH),
        "N_train": n_train,
        "N_val": N_VAL,
        "N_test": N_TEST,
        "all_sample_sizes": ALL_SAMPLE_SIZES,
        "run_sample_sizes": SAMPLE_SIZES,
        "split_mode": SPLIT_MODE,
        "split_seed": SPLIT_SEED,
        "protocol": {
            "temporal_basis_from_training_windows_only": True,
            "test_windows_constructed": False,
            "test_temporal_basis_coefficients_projected": False,
            "windowed_models": ["TB-AWFNO", "AWFNO", "WFNO"],
            "windowed_rollout_initial_condition": "test initial state only",
            "windowed_later_rollout_inputs": "previous predictions",
            "non_windowed_fno": "neuralop.models.FNO maps initial condition to full future trajectory",
        },
        "architecture": {
            "models": ["TB-AWFNO", "AWFNO", "WFNO", "FNO"],
            "alternating_update": "local-Fourier-local residual alternation for TB-AWFNO and AWFNO",
            "wfno": "standard FNO-style backbone under the same window protocol",
            "fno": "neuralop.models.FNO, non-windowed full-trajectory output",
            "window_k": WINDOW_K,
            "n_time_modes": N_TIME_MODES,
            "width": WIDTH,
            "n_layers": N_LAYERS,
            "modes_x": MODES_X,
            "local_radius": LOCAL_RADIUS,
        },
        "metrics": row,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary:", summary_path)

    return row


def plot_sample_size_sweep(rows):
    if not SAVE_PLOTS:
        return
    rows_sorted = sorted(rows, key=lambda r: r["N_train"])
    Ns = [r["N_train"] for r in rows_sorted]
    plt.figure(figsize=(8, 5))
    for key, label in [("TB_AWFNO_full_l2_mean", "TB-AWFNO"), ("AWFNO_full_l2_mean", "AWFNO"), ("WFNO_full_l2_mean", "WFNO"), ("FNO_full_l2_mean", "FNO")]:
        plt.plot(Ns, [r[key] for r in rows_sorted], marker="o", label=label)
    plt.xlabel("number of training trajectories")
    plt.ylabel("mean absolute full-trajectory physical L2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = PLOT_DIR / "sample_size_sweep_abs_full_L2_common_test.png"
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)

    plt.figure(figsize=(8, 5))
    for key, label in [("TB_AWFNO_final_l2_mean", "TB-AWFNO"), ("AWFNO_final_l2_mean", "AWFNO"), ("WFNO_final_l2_mean", "WFNO"), ("FNO_final_l2_mean", "FNO")]:
        plt.plot(Ns, [r[key] for r in rows_sorted], marker="o", label=label)
    plt.xlabel("number of training trajectories")
    plt.ylabel("mean absolute final-time physical L2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = PLOT_DIR / "sample_size_sweep_abs_final_L2_common_test.png"
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)


def print_sample_size_summary(rows):
    rows_sorted = sorted(rows, key=lambda r: r["N_train"])
    print("\n================ SAMPLE-SIZE SWEEP SUMMARY: ABSOLUTE FULL L2 ================")
    for r in rows_sorted:
        print(f"N={r['N_train']:4d} | Abs full L2: TB-AWFNO {r['TB_AWFNO_full_l2_mean']:.6e} | AWFNO {r['AWFNO_full_l2_mean']:.6e} | WFNO {r['WFNO_full_l2_mean']:.6e} | FNO {r['FNO_full_l2_mean']:.6e} | winner {r['winner']}")
    print("============================================================================\n")
    print("\n================ SAMPLE-SIZE SWEEP SUMMARY: ABSOLUTE FINAL L2 ================")
    for r in rows_sorted:
        final_scores = {"TB-AWFNO": r["TB_AWFNO_final_l2_mean"], "AWFNO": r["AWFNO_final_l2_mean"], "WFNO": r["WFNO_final_l2_mean"], "FNO": r["FNO_final_l2_mean"]}
        final_winner = min(final_scores, key=final_scores.get)
        print(f"N={r['N_train']:4d} | Abs final L2: TB-AWFNO {r['TB_AWFNO_final_l2_mean']:.6e} | AWFNO {r['AWFNO_final_l2_mean']:.6e} | WFNO {r['WFNO_final_l2_mean']:.6e} | FNO {r['FNO_final_l2_mean']:.6e} | winner {final_winner}")
    print("=============================================================================\n")


# ============================================================
# 13. CONTROLLED REVIEWER EXPERIMENTS
# ============================================================

def set_reviewer_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_alt_spec(
    name,
    pattern="LFL",
    window_k=5,
    width=32,
    n_layers=4,
    modes_x=16,
    local_radius=3,
):
    return {
        "name": str(name),
        "family": "alternating",
        "pattern": str(pattern).upper(),
        "window_k": int(window_k),
        "window_stride": int(window_k),
        "width": int(width),
        "n_layers": int(n_layers),
        "modes_x": int(modes_x),
        "local_radius": int(local_radius),
    }


def make_wfno_spec(
    name,
    window_k=5,
    width=32,
    n_layers=4,
    modes_x=16,
):
    return {
        "name": str(name),
        "family": "wfno",
        "pattern": "standard_fno",
        "window_k": int(window_k),
        "window_stride": int(window_k),
        "width": int(width),
        "n_layers": int(n_layers),
        "modes_x": int(modes_x),
        "local_radius": 0,
    }


def reviewer_spec_key(spec):
    return (
        spec["family"],
        spec["pattern"],
        spec["window_k"],
        spec["window_stride"],
        spec["width"],
        spec["n_layers"],
        spec["modes_x"],
        spec["local_radius"],
    )


def make_reviewer_specs():
    valid_suites = {"core", "ordering", "sensitivity", "all"}
    if REVIEWER_SUITE not in valid_suites:
        raise ValueError(
            f"REVIEWER_SUITE={REVIEWER_SUITE!r}; expected one of "
            f"{sorted(valid_suites)}"
        )

    base = make_alt_spec(
        "AWFNO_LFL",
        pattern="LFL",
        window_k=WINDOW_K,
        width=WIDTH,
        n_layers=N_LAYERS,
        modes_x=MODES_X,
        local_radius=LOCAL_RADIUS,
    )
    specs = []

    if REVIEWER_SUITE in {"core", "all"}:
        specs.extend(
            [
                base,
                make_alt_spec(
                    "LOCAL_ONLY_LIL",
                    pattern="LIL",
                    window_k=WINDOW_K,
                    width=WIDTH,
                    n_layers=N_LAYERS,
                    modes_x=MODES_X,
                    local_radius=LOCAL_RADIUS,
                ),
                make_alt_spec(
                    "FOURIER_ONLY_IFI",
                    pattern="IFI",
                    window_k=WINDOW_K,
                    width=WIDTH,
                    n_layers=N_LAYERS,
                    modes_x=MODES_X,
                    local_radius=LOCAL_RADIUS,
                ),
                make_wfno_spec(
                    f"WFNO_W{WIDTH}",
                    window_k=WINDOW_K,
                    width=WIDTH,
                    n_layers=N_LAYERS,
                    modes_x=MODES_X,
                ),
                make_wfno_spec(
                    f"WFNO_PM_W{WFNO_PM_WIDTH}",
                    window_k=WINDOW_K,
                    width=WFNO_PM_WIDTH,
                    n_layers=N_LAYERS,
                    modes_x=MODES_X,
                ),
            ]
        )

    if REVIEWER_SUITE in {"ordering", "all"}:
        specs.extend(
            [
                base,
                make_alt_spec(
                    "AWFNO_LLF",
                    pattern="LLF",
                    window_k=WINDOW_K,
                    width=WIDTH,
                    n_layers=N_LAYERS,
                    modes_x=MODES_X,
                    local_radius=LOCAL_RADIUS,
                ),
                make_alt_spec(
                    "AWFNO_FLL",
                    pattern="FLL",
                    window_k=WINDOW_K,
                    width=WIDTH,
                    n_layers=N_LAYERS,
                    modes_x=MODES_X,
                    local_radius=LOCAL_RADIUS,
                ),
            ]
        )

    if REVIEWER_SUITE in {"sensitivity", "all"}:
        specs.append(base)
        for radius in R_SWEEP:
            if int(radius) == int(LOCAL_RADIUS):
                continue
            specs.append(
                make_alt_spec(
                    f"AWFNO_R{radius}",
                    pattern="LFL",
                    window_k=WINDOW_K,
                    width=WIDTH,
                    n_layers=N_LAYERS,
                    modes_x=MODES_X,
                    local_radius=radius,
                )
            )
        for modes in MODES_SWEEP:
            if int(modes) == int(MODES_X):
                continue
            specs.append(
                make_alt_spec(
                    f"AWFNO_MODES{modes}",
                    pattern="LFL",
                    window_k=WINDOW_K,
                    width=WIDTH,
                    n_layers=N_LAYERS,
                    modes_x=modes,
                    local_radius=LOCAL_RADIUS,
                )
            )
        for window_k in K_SWEEP:
            if int(window_k) == int(WINDOW_K):
                continue
            specs.append(
                make_alt_spec(
                    f"AWFNO_K{window_k}",
                    pattern="LFL",
                    window_k=window_k,
                    width=WIDTH,
                    n_layers=N_LAYERS,
                    modes_x=MODES_X,
                    local_radius=LOCAL_RADIUS,
                )
            )

    # Preserve the first descriptive name for an identical configuration.
    deduplicated = []
    seen = set()
    for spec in specs:
        key = reviewer_spec_key(spec)
        if key not in seen:
            seen.add(key)
            deduplicated.append(spec)

    if REVIEWER_MODEL_FILTER:
        requested = set(REVIEWER_MODEL_FILTER)
        known = {spec["name"] for spec in deduplicated}
        missing = sorted(requested - known)
        if missing:
            raise ValueError(
                f"Unknown REVIEWER_MODELS entries: {missing}. "
                f"Available for suite {REVIEWER_SUITE!r}: {sorted(known)}"
            )
        deduplicated = [
            spec for spec in deduplicated if spec["name"] in requested
        ]

    if not deduplicated:
        raise RuntimeError("No reviewer experiment specifications were selected.")
    return deduplicated


def make_reviewer_model(spec):
    if spec["family"] == "alternating":
        return ReviewerAlternatingModel1d(
            window_k=spec["window_k"],
            x_grid=x_grid_np,
            pattern=spec["pattern"],
            width=spec["width"],
            n_layers=spec["n_layers"],
            modes_x=spec["modes_x"],
            local_radius=spec["local_radius"],
        )
    if spec["family"] == "wfno":
        return ReviewerWFNO1d(
            window_k=spec["window_k"],
            x_grid=x_grid_np,
            width=spec["width"],
            n_layers=spec["n_layers"],
            modes_x=spec["modes_x"],
        )
    raise ValueError(f"Unknown model family: {spec['family']!r}")


def build_windows_explicit(U, window_k, stride):
    starts = make_window_start_indices(U.shape[-1], window_k, stride)
    n, nx, _ = U.shape
    u_start = np.empty((n * len(starts), nx), dtype=np.float32)
    U_future = np.empty(
        (n * len(starts), nx, window_k),
        dtype=np.float32,
    )
    q = 0
    for i in range(n):
        for s in starts:
            u_start[q] = U[i, :, s]
            U_future[q] = U[i, :, s + 1:s + window_k + 1]
            q += 1
    return u_start, U_future, starts


def prepare_reviewer_window_data(
    U_train_pool,
    U_val,
    n_train,
    window_k,
    window_stride,
):
    U_train = U_train_pool[:n_train]
    u_train, Uf_train, train_starts = build_windows_explicit(
        U_train,
        window_k,
        window_stride,
    )
    u_val, Uf_val, val_starts = build_windows_explicit(
        U_val,
        window_k,
        window_k,
    )
    train_ds = WindowDataset1D(u_train, Uf_train)
    val_ds = WindowDataset1D(u_val, Uf_val)
    u_normer = GaussianNormalizer(
        torch.tensor(u_train, dtype=torch.float32)
    ).to(device)
    future_normer = GaussianNormalizer(
        torch.tensor(Uf_train, dtype=torch.float32)
    ).to(device)
    print(
        f"Prepared K={window_k}, stride={window_stride}: "
        f"{len(train_starts)} train windows/trajectory, "
        f"{len(val_starts)} validation windows/trajectory, "
        f"train tensor={Uf_train.shape}"
    )
    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "u_normer": u_normer,
        "future_normer": future_normer,
        "n_train_windows": int(len(train_ds)),
        "n_val_windows": int(len(val_ds)),
        "train_starts": train_starts,
        "val_starts": val_starts,
    }


def make_reviewer_loaders(data_obj, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 1009 * REVIEWER_N_TRAIN)
    train_loader = DataLoader(
        data_obj["train_ds"],
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        data_obj["val_ds"],
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader


def make_reviewer_predict(model, u_normer, future_normer):
    def predict(u_start):
        return future_normer.decode(model(u_normer.encode(u_start)))
    return predict


def reviewer_window_loss(pred, true):
    return (
        F.mse_loss(pred, true)
        + COMMON_FINAL_LOSS_WEIGHT
        * F.mse_loss(pred[:, :, -1], true[:, :, -1])
    )


def reviewer_train_epoch(model, loader, optimizer, predict):
    model.train()
    loss_sum = l2_sum = 0.0
    n_seen = 0
    for batch in loader:
        u_start = to_device(batch["u_start"])
        U_true = to_device(batch["U_future"])
        optimizer.zero_grad(set_to_none=True)
        U_pred = predict(u_start)
        loss = reviewer_window_loss(U_pred, U_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = u_start.shape[0]
        loss_sum += float(loss.item()) * bs
        l2_sum += (
            float(abs_l2_window_batch(U_pred.detach(), U_true).mean().item())
            * bs
        )
        n_seen += bs
    return {
        "loss": loss_sum / max(n_seen, 1),
        "abs_l2": l2_sum / max(n_seen, 1),
    }


@torch.no_grad()
def reviewer_eval_epoch(model, loader, predict):
    model.eval()
    loss_sum = l2_sum = final_sum = 0.0
    n_seen = 0
    for batch in loader:
        u_start = to_device(batch["u_start"])
        U_true = to_device(batch["U_future"])
        U_pred = predict(u_start)
        loss = reviewer_window_loss(U_pred, U_true)
        bs = u_start.shape[0]
        loss_sum += float(loss.item()) * bs
        l2_sum += float(abs_l2_window_batch(U_pred, U_true).mean().item()) * bs
        final_sum += (
            float(
                abs_l2_field_batch(
                    U_pred[:, :, -1],
                    U_true[:, :, -1],
                ).mean().item()
            )
            * bs
        )
        n_seen += bs
    return {
        "loss": loss_sum / max(n_seen, 1),
        "abs_l2": l2_sum / max(n_seen, 1),
        "final_l2": final_sum / max(n_seen, 1),
    }


def fit_reviewer_model(model, train_loader, val_loader, predict, model_name):
    # Identical optimization for every architecture in the controlled study.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )
    best_val = float("inf")
    best_state = None
    history = {
        "train_loss": [],
        "train_l2": [],
        "val_loss": [],
        "val_l2": [],
        "val_final_l2": [],
    }
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_stats = reviewer_train_epoch(
            model,
            train_loader,
            optimizer,
            predict,
        )
        val_stats = reviewer_eval_epoch(model, val_loader, predict)
        scheduler.step()
        history["train_loss"].append(train_stats["loss"])
        history["train_l2"].append(train_stats["abs_l2"])
        history["val_loss"].append(val_stats["loss"])
        history["val_l2"].append(val_stats["abs_l2"])
        history["val_final_l2"].append(val_stats["final_l2"])
        if val_stats["abs_l2"] < best_val:
            best_val = val_stats["abs_l2"]
            best_state = clean_state_dict(model.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"[{model_name}] {epoch:04d}/{EPOCHS} | "
                f"train L2 {train_stats['abs_l2']:.6e} | "
                f"val L2 {val_stats['abs_l2']:.6e} | "
                f"val final {val_stats['final_l2']:.6e}"
            )
    train_time = time.perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return {
        "best_val_l2": float(best_val),
        "train_time_sec": float(train_time),
        "history": history,
    }


def torch_load_checkpoint_safe(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def reviewer_rollout(model, predict, U_test, window_k):
    model.eval()
    n, nx, nt = U_test.shape
    U_roll = np.empty((n, nx, nt), dtype=np.float32)
    U_roll[:, :, 0] = U_test[:, :, 0]
    for i0 in range(0, n, BATCH_SIZE):
        i1 = min(i0 + BATCH_SIZE, n)
        u_current = torch.tensor(
            U_test[i0:i1, :, 0],
            dtype=torch.float32,
            device=device,
        )
        cursor = 0
        block = 0
        while cursor < nt - 1:
            U_future = predict(u_current)
            use_k = min(window_k, nt - 1 - cursor)
            U_roll[
                i0:i1,
                :,
                cursor + 1:cursor + 1 + use_k,
            ] = U_future[:, :, :use_k].detach().cpu().numpy()
            u_current = U_future[:, :, use_k - 1].detach()
            if PRINT_ROLLOUT_BLOCKS and i0 == 0:
                print(
                    f"rollout block {block:03d}: "
                    f"{cursor:04d}->{cursor + use_k:04d}"
                )
            cursor += use_k
            block += 1
    return U_roll


def rfft_parseval_weights(nx):
    n_freq = nx // 2 + 1
    weights = np.ones(n_freq, dtype=np.float64)
    if nx % 2 == 0:
        if n_freq > 2:
            weights[1:-1] = 2.0
    elif n_freq > 1:
        weights[1:] = 2.0
    return weights


def spectral_error_metrics(pred, true, cutoff):
    if pred.shape != true.shape:
        raise ValueError(f"Spectral metric shape mismatch: {pred.shape} vs {true.shape}")
    _, nx, nt = pred.shape
    cutoff = max(1, min(int(cutoff), nx // 2 + 1))
    freq_idx = np.arange(nx // 2 + 1)
    low_mask = freq_idx < cutoff
    high_mask = ~low_mask
    fft_weights = rfft_parseval_weights(nx).reshape(1, -1, 1)
    time_weights = _time_weights_np(nt).reshape(1, 1, -1)

    err_hat = np.fft.rfft(
        pred.astype(np.float64) - true.astype(np.float64),
        axis=1,
        norm="ortho",
    )
    true_hat = np.fft.rfft(
        true.astype(np.float64),
        axis=1,
        norm="ortho",
    )
    err_energy = np.abs(err_hat) ** 2 * fft_weights
    true_energy = np.abs(true_hat) ** 2 * fft_weights

    def summarize(mask):
        err_band_t = np.sum(err_energy[:, mask, :], axis=1)
        true_band_t = np.sum(true_energy[:, mask, :], axis=1)
        err_integrated = (
            DX
            * DT_SNAPSHOT
            * np.sum(err_band_t * time_weights[:, 0, :], axis=1)
        )
        true_integrated = (
            DX
            * DT_SNAPSHOT
            * np.sum(true_band_t * time_weights[:, 0, :], axis=1)
        )
        abs_l2 = np.sqrt(err_integrated + 1.0e-30)
        global_relative = np.sqrt(
            np.sum(err_integrated)
            / (np.sum(true_integrated) + 1.0e-30)
        )
        return {
            "abs_l2_mean": float(np.mean(abs_l2)),
            "abs_l2_std": float(np.std(abs_l2)),
            "relative_l2_global": float(global_relative),
        }

    low = summarize(low_mask)
    high = summarize(high_mask)
    return {
        "cutoff_definition": f"low modes 0,...,{cutoff - 1}; high modes >= {cutoff}",
        "cutoff": int(cutoff),
        "low_abs_l2_mean": low["abs_l2_mean"],
        "low_abs_l2_std": low["abs_l2_std"],
        "low_relative_l2_global": low["relative_l2_global"],
        "high_abs_l2_mean": high["abs_l2_mean"],
        "high_abs_l2_std": high["abs_l2_std"],
        "high_relative_l2_global": high["relative_l2_global"],
    }


def sharp_gradient_error_metrics(pred, true, quantile):
    if pred.shape != true.shape:
        raise ValueError(f"Sharp metric shape mismatch: {pred.shape} vs {true.shape}")
    quantile = float(quantile)
    if not 0.0 < quantile < 1.0:
        raise ValueError("SHARP_GRADIENT_QUANTILE must be strictly between 0 and 1.")
    grad = np.abs(
        np.roll(true, -1, axis=1) - np.roll(true, 1, axis=1)
    ) / (2.0 * DX)
    threshold = np.quantile(grad, quantile, axis=1, keepdims=True)
    sharp_mask = grad >= threshold
    # t=0 has exactly zero prediction error and would dilute both regions.
    sharp_mask[:, :, 0] = False
    valid_mask = np.ones_like(sharp_mask, dtype=bool)
    valid_mask[:, :, 0] = False
    smooth_mask = valid_mask & (~sharp_mask)
    error2 = (
        pred.astype(np.float64) - true.astype(np.float64)
    ) ** 2
    sharp_energy = float(np.sum(error2[sharp_mask]))
    smooth_energy = float(np.sum(error2[smooth_mask]))
    sharp_count = int(np.sum(sharp_mask))
    smooth_count = int(np.sum(smooth_mask))
    actual_sharp_fraction = sharp_count / max(sharp_count + smooth_count, 1)
    sharp_rmse = np.sqrt(sharp_energy / max(sharp_count, 1))
    smooth_rmse = np.sqrt(smooth_energy / max(smooth_count, 1))
    total_energy = sharp_energy + smooth_energy
    return {
        "gradient_quantile": quantile,
        "sharp_spatial_fraction": float(actual_sharp_fraction),
        "sharp_rmse": float(sharp_rmse),
        "smooth_rmse": float(smooth_rmse),
        "sharp_to_smooth_rmse_ratio": float(
            sharp_rmse / (smooth_rmse + 1.0e-30)
        ),
        "sharp_fraction_of_error_energy": float(
            sharp_energy / (total_energy + 1.0e-30)
        ),
    }


def synchronize_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def profile_reviewer_inference(model, predict, u0_np, nt, window_k):
    if not PROFILE_INFERENCE:
        return {}
    model.eval()
    u0 = torch.tensor(
        u0_np[:1],
        dtype=torch.float32,
        device=device,
    )

    for _ in range(PROFILE_WARMUP):
        _ = predict(u0)
    synchronize_if_needed()

    forward_times = []
    for _ in range(PROFILE_FORWARD_REPEATS):
        synchronize_if_needed()
        start = time.perf_counter()
        _ = predict(u0)
        synchronize_if_needed()
        forward_times.append((time.perf_counter() - start) * 1000.0)

    def one_rollout():
        u_current = u0
        cursor = 0
        n_calls = 0
        while cursor < nt - 1:
            U_future = predict(u_current)
            use_k = min(window_k, nt - 1 - cursor)
            u_current = U_future[:, :, use_k - 1]
            cursor += use_k
            n_calls += 1
        return n_calls

    for _ in range(PROFILE_WARMUP):
        one_rollout()
    synchronize_if_needed()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rollout_times = []
    n_calls = None
    for _ in range(PROFILE_ROLLOUT_REPEATS):
        synchronize_if_needed()
        start = time.perf_counter()
        n_calls = one_rollout()
        synchronize_if_needed()
        rollout_times.append((time.perf_counter() - start) * 1000.0)
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated() / 1024**2)
        if torch.cuda.is_available()
        else float("nan")
    )
    return {
        "forward_batch1_ms_median": float(np.median(forward_times)),
        "forward_batch1_ms_mean": float(np.mean(forward_times)),
        "full_rollout_batch1_ms_median": float(np.median(rollout_times)),
        "full_rollout_batch1_ms_mean": float(np.mean(rollout_times)),
        "rollout_network_calls": int(n_calls),
        "peak_allocated_memory_mb": peak_memory_mb,
    }


def flatten_dict(obj, prefix=""):
    out = {}
    for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, full_key))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[full_key] = value
    return out


def reviewer_checkpoint_tag(spec, seed):
    config = {
        "spec": spec,
        "seed": int(seed),
        "n_train": REVIEWER_N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "split_mode": SPLIT_MODE,
        "split_seed": SPLIT_SEED,
        "target_nx": TARGET_NX,
        "time_subsample": TIME_SUBSAMPLE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "final_loss_weight": COMMON_FINAL_LOSS_WEIGHT,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "gradient_clip": 1.0,
    }
    return short_hash(config)


def run_one_reviewer_experiment(
    spec,
    seed,
    data_obj,
    U_test,
    test_ids,
    reviewer_root,
):
    set_reviewer_seed(seed)
    train_loader, val_loader = make_reviewer_loaders(data_obj, seed)
    model = make_reviewer_model(spec).to(device)
    predict = make_reviewer_predict(
        model,
        data_obj["u_normer"],
        data_obj["future_normer"],
    )

    # Shape smoke test before a potentially expensive training run.
    batch = next(iter(val_loader))
    with torch.no_grad():
        smoke_output = predict(to_device(batch["u_start"][:2]))
    expected_shape = (
        min(2, batch["u_start"].shape[0]),
        Nx,
        spec["window_k"],
    )
    if tuple(smoke_output.shape) != expected_shape:
        raise RuntimeError(
            f"{spec['name']} produced {tuple(smoke_output.shape)}, "
            f"expected {expected_shape}."
        )

    params = count_params(model)
    real_params = count_real_scalar_params(model)
    tag = reviewer_checkpoint_tag(spec, seed)
    run_dir = reviewer_root / safe_name(spec["name"]) / f"seed_{seed}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "model.pt"

    print("\n" + "=" * 88)
    print(f"MODEL: {spec['name']} | seed={seed}")
    print("=" * 88)
    print("spec:", json.dumps(spec, sort_keys=True))
    print(f"trainable tensor elements : {params:,}")
    print(f"trainable real scalars    : {real_params:,}")
    print("checkpoint:", checkpoint_path)

    if checkpoint_path.exists() and not FORCE_RETRAIN:
        payload = torch_load_checkpoint_safe(checkpoint_path)
        model.load_state_dict(
            clean_state_dict(payload["state_dict"]),
            strict=True,
        )
        fit = payload.get("meta", {})
        print("Loaded cached controlled checkpoint.")
    else:
        fit = fit_reviewer_model(
            model,
            train_loader,
            val_loader,
            predict,
            spec["name"],
        )
        save_model_checkpoint(
            checkpoint_path,
            model,
            {
                **fit,
                "spec": spec,
                "seed": int(seed),
                "params": int(params),
                "real_scalar_params": int(real_params),
            },
        )

    U_roll = reviewer_rollout(
        model,
        predict,
        U_test,
        spec["window_k"],
    )
    global_metrics = rollout_l2_metrics(U_roll, U_test)
    spectral = spectral_error_metrics(
        U_roll,
        U_test,
        FREQUENCY_CUTOFF,
    )
    sharp = sharp_gradient_error_metrics(
        U_roll,
        U_test,
        SHARP_GRADIENT_QUANTILE,
    )
    timing = profile_reviewer_inference(
        model,
        predict,
        U_test[:, :, 0],
        U_test.shape[-1],
        spec["window_k"],
    )

    result = {
        "model": spec["name"],
        "seed": int(seed),
        "spec": spec,
        "params": int(params),
        "real_scalar_params": int(real_params),
        "best_val_l2": float(fit.get("best_val_l2", float("nan"))),
        "train_time_sec": float(fit.get("train_time_sec", float("nan"))),
        "train_time_sec_per_epoch": float(
            fit.get("train_time_sec", float("nan")) / max(EPOCHS, 1)
        ),
        "full_l2_mean": global_metrics["full_l2_mean"],
        "full_l2_std_across_test": global_metrics["full_l2_std"],
        "final_l2_mean": global_metrics["final_l2_mean"],
        "final_l2_std_across_test": global_metrics["final_l2_std"],
        "spectral": spectral,
        "sharp_region": sharp,
        "timing": timing,
        "n_train_windows": int(data_obj["n_train_windows"]),
        "n_val_windows": int(data_obj["n_val_windows"]),
        "checkpoint_path": str(checkpoint_path),
    }

    if SAVE_REVIEWER_ROLLOUTS:
        rollout_path = run_dir / "test_rollout.npz"
        np.savez_compressed(
            rollout_path,
            U_pred=U_roll,
            test_ids=np.asarray(test_ids, dtype=np.int64),
            model_name=np.asarray(spec["name"]),
            seed=np.asarray(seed, dtype=np.int64),
            x_grid=x_grid_np,
            t_grid=t_grid,
        )
        result["rollout_path"] = str(rollout_path)

    result_path = run_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"RESULT {spec['name']}: full L2={result['full_l2_mean']:.6e}, "
        f"final L2={result['final_l2_mean']:.6e}, "
        f"low={spectral['low_abs_l2_mean']:.6e}, "
        f"high={spectral['high_abs_l2_mean']:.6e}, "
        f"sharp RMSE={sharp['sharp_rmse']:.6e}"
    )
    return result


def aggregate_reviewer_results(results):
    flat_results = [flatten_dict(result) for result in results]
    by_model = {}
    for row in flat_results:
        by_model.setdefault(row["model"], []).append(row)

    aggregates = []
    for model_name, rows in sorted(by_model.items()):
        aggregate = {
            "model": model_name,
            "n_seeds": len(rows),
            "seeds": [int(row["seed"]) for row in rows],
        }
        all_keys = sorted(set().union(*(row.keys() for row in rows)))
        for key in all_keys:
            if key in {"model", "seed"}:
                continue
            values = [row.get(key) for row in rows]
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in values
            ):
                array = np.asarray(values, dtype=np.float64)
                if np.all(np.isfinite(array)):
                    aggregate[f"{key}.mean"] = float(np.mean(array))
                    aggregate[f"{key}.std_across_seeds"] = float(np.std(array))
        aggregates.append(aggregate)
    return flat_results, aggregates


def save_reviewer_tables(results, reviewer_root):
    flat_results, aggregates = aggregate_reviewer_results(results)
    payload = {
        "experiment": EXPERIMENT_NAME,
        "suite": REVIEWER_SUITE,
        "n_train": REVIEWER_N_TRAIN,
        "seeds": REVIEWER_SEEDS,
        "controlled_training_protocol": {
            "loss": "window MSE + final-window MSE",
            "final_loss_weight": COMMON_FINAL_LOSS_WEIGHT,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "gradient_clip": 1.0,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "raw_results": results,
        "aggregates": aggregates,
    }
    summary_json = reviewer_root / f"summary_{REVIEWER_SUITE}.json"
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2)

    all_fields = sorted(set().union(*(row.keys() for row in flat_results)))
    summary_csv = reviewer_root / f"summary_{REVIEWER_SUITE}_raw.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(flat_results)

    aggregate_fields = sorted(
        set().union(*(row.keys() for row in aggregates))
    )
    aggregate_csv = reviewer_root / f"summary_{REVIEWER_SUITE}_aggregated.csv"
    with open(aggregate_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregates)

    print("\nSaved reviewer summaries")
    print("------------------------")
    print(summary_json)
    print(summary_csv)
    print(aggregate_csv)
    return summary_json, summary_csv, aggregate_csv


def verify_parameter_matching(specs):
    counts = {}
    for spec in specs:
        model = make_reviewer_model(spec)
        counts[spec["name"]] = {
            "params": count_params(model),
            "real_scalar_params": count_real_scalar_params(model),
        }
        del model

    ordering_names = [
        name for name in ["AWFNO_LFL", "AWFNO_LLF", "AWFNO_FLL"]
        if name in counts
    ]
    if len(ordering_names) >= 2:
        ordering_counts = {counts[name]["params"] for name in ordering_names}
        if len(ordering_counts) != 1:
            raise RuntimeError(
                f"Ordering variants are not parameter matched: {counts}"
            )

    if "AWFNO_LFL" in counts:
        aw_params = counts["AWFNO_LFL"]["params"]
        pm_names = [name for name in counts if name.startswith("WFNO_PM_")]
        for name in pm_names:
            gap = (counts[name]["params"] - aw_params) / aw_params
            counts[name]["relative_gap_vs_AWFNO"] = float(gap)
            if abs(gap) > 0.05:
                raise RuntimeError(
                    f"{name} is not within 5% of AWFNO parameters: "
                    f"{counts[name]['params']} vs {aw_params}."
                )

    print("\nParameter preflight")
    print("-------------------")
    for name, item in counts.items():
        extra = ""
        if "relative_gap_vs_AWFNO" in item:
            extra = (
                f" | gap vs AWFNO="
                f"{100.0 * item['relative_gap_vs_AWFNO']:+.2f}%"
            )
        print(
            f"{name:22s}: params={item['params']:,} | "
            f"real scalars={item['real_scalar_params']:,}{extra}"
        )
    return counts


# ============================================================
# 14. MAIN
# ============================================================

def main():
    if REVIEWER_N_TRAIN > N_TRAIN_POOL:
        raise ValueError(
            f"REVIEWER_N_TRAIN={REVIEWER_N_TRAIN} exceeds "
            f"N_TRAIN_POOL={N_TRAIN_POOL}."
        )

    print("\nControlled reviewer experiment")
    print("------------------------------")
    print("name                    :", EXPERIMENT_NAME)
    print("suite                   :", REVIEWER_SUITE)
    print("data path               :", DATA_PATH)
    print("output root             :", OUTPUT_ROOT)
    print("device                  :", device)
    print("N_train                 :", REVIEWER_N_TRAIN)
    print("N_val / N_test          :", N_VAL, N_TEST)
    print("seeds                   :", REVIEWER_SEEDS)
    print("epochs                  :", EPOCHS)
    print("common optimizer        : AdamW")
    print("common scheduler        : CosineAnnealingLR")
    print("common loss             : window MSE + final-window MSE")
    print("final loss weight       :", COMMON_FINAL_LOSS_WEIGHT)
    print("common grad clipping    : 1.0")
    print("force retrain           :", FORCE_RETRAIN)
    print("save rollouts           :", SAVE_REVIEWER_ROLLOUTS)

    U_all, x, t = load_hdf5_1d_dataset(DATA_PATH)
    U_all, x = downsample_space_stride(U_all, x, TARGET_NX)
    set_geometry_from_arrays(U_all.shape, x, t)
    (
        U_train_pool,
        train_ids,
        U_val,
        val_ids,
        U_test,
        test_ids,
    ) = make_train_val_test_splits(U_all)

    if RUN_EXACT_HASH_OVERLAP_CHECK:
        check_exact_hash_overlap(U_train_pool, U_val, "train_pool", "val")
        check_exact_hash_overlap(U_train_pool, U_test, "train_pool", "test")
        check_exact_hash_overlap(U_val, U_test, "val", "test")
    if RUN_DISTANCE_OVERLAP_CHECK:
        check_dataset_overlap(
            U_train_pool,
            U_test,
            "train_pool",
            "test",
            tol=OVERLAP_TOL,
        )

    reviewer_root = OUTPUT_ROOT / (
        f"{EXPERIMENT_NAME}_N{REVIEWER_N_TRAIN}_"
        f"Nx{Nx}_Nt{Nt}_split{SPLIT_SEED}"
    )
    reviewer_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        reviewer_root / "split_ids_and_grids.npz",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        x_grid=x_grid_np,
        t_grid=t_grid,
    )

    specs = make_reviewer_specs()
    print("\nSelected models")
    print("---------------")
    for spec in specs:
        print(json.dumps(spec, sort_keys=True))
    verify_parameter_matching(specs)

    data_cache = {}
    results = []
    for seed in REVIEWER_SEEDS:
        for spec in specs:
            data_key = (
                spec["window_k"],
                spec["window_stride"],
            )
            if data_key not in data_cache:
                data_cache[data_key] = prepare_reviewer_window_data(
                    U_train_pool,
                    U_val,
                    REVIEWER_N_TRAIN,
                    spec["window_k"],
                    spec["window_stride"],
                )
            result = run_one_reviewer_experiment(
                spec,
                seed,
                data_cache[data_key],
                U_test,
                test_ids,
                reviewer_root,
            )
            results.append(result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    save_reviewer_tables(results, reviewer_root)
    print("\nControlled reviewer suite completed.")


if __name__ == "__main__":
    main()