#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annulus 2D scalar-dynamics benchmark for finite-window and full-trajectory neural operators.

The script compares four models:
    1. TB-AWFNO: temporal-basis alternating windowed Fourier neural operator
    2. AWFNO: alternating windowed Fourier neural operator
    3. WFNO: windowed Fourier neural operator baseline
    4. FNO: non-windowed full-trajectory Fourier neural operator baseline

Expected dataset keys:
    c_grid or Y_output : array of shape (N, T, Nr, Ntheta, 1)
    r_grid             : radial grid
    theta_grid         : angular grid

Evaluation protocol:
    - train, validation, and test trajectories are disjoint;
    - the temporal basis is computed from training windows only;
    - test trajectories are evaluated without test-window fitting or basis projection;
    - windowed models are initialized only from the true initial condition;
    - subsequent window inputs are previous predictions;
    - the full-trajectory FNO maps the initial condition directly to the future trajectory.

Primary metric:
    absolute physical L2 error on the annulus using radial quadrature and periodic angular quadrature.
"""

import os
import gc
import json
import copy
import time
import random
import hashlib
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. CONFIG
# ============================================================

SEED = int(os.environ.get("SEED", "1234"))

# Copy the dataset to a fast local path when possible, then point DATA_PATH here.
DATA_PATH = Path(os.environ.get(
    "DATA_PATH",
    "/content/gdrive/MyDrive/anus/hyteg_annulus_c_grid_64x128_samples500.pt",
))

OUTPUT_ROOT = Path(os.environ.get(
    "OUTPUT_ROOT",
    "/content/gdrive/MyDrive/anus/output",
))

# Spatial/time subsampling. Keep None for the original grid.
TARGET_NR = None
TARGET_NTHETA = None
TIME_SUBSAMPLE = int(os.environ.get("TIME_SUBSAMPLE", "1"))

# Keep this fixed so that separate runs use the same split.
ALL_SAMPLE_SIZES = [25,50,100,200,400]

_run_sizes = os.environ.get("RUN_SAMPLE_SIZES", "").strip()
if _run_sizes:
    SAMPLE_SIZES = [int(x.strip()) for x in _run_sizes.split(",") if x.strip()]
else:
    SAMPLE_SIZES = ALL_SAMPLE_SIZES

SPLIT_MODE = os.environ.get("SPLIT_MODE", "contiguous")  # "contiguous" or "random"
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", "2026"))

# Important: use the full pool size even when running a single sample size.
N_TRAIN_POOL = max(ALL_SAMPLE_SIZES)
N_VAL = int(os.environ.get("N_VAL", "30"))
N_TEST = int(os.environ.get("N_TEST", "70"))

# Window / temporal basis / model
WINDOW_K = int(os.environ.get("WINDOW_K", "30"))
WINDOW_STRIDE = int(os.environ.get("WINDOW_STRIDE", "30"))
N_TIME_MODES = int(os.environ.get("N_TIME_MODES", "18"))

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("EPOCHS", "150"))
LR = float(os.environ.get("LR", "1.0e-3"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1.0e-6"))

WIDTH = int(os.environ.get("WIDTH", "32"))
N_LAYERS = int(os.environ.get("N_LAYERS", "4"))
MODES_R = int(os.environ.get("MODES_R", "16"))
MODES_THETA = int(os.environ.get("MODES_THETA", "24"))
LOCAL_RADIUS = int(os.environ.get("LOCAL_RADIUS", "2"))

FINAL_LOSS_WEIGHT = float(os.environ.get("FINAL_LOSS_WEIGHT", "0.0"))
ENFORCE_MEAN = os.environ.get("ENFORCE_MEAN", "0") == "1"

USE_LIBRARY_FNO = os.environ.get("USE_LIBRARY_FNO", "1") == "1"
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"

RUN_EXACT_HASH_OVERLAP_CHECK = os.environ.get("RUN_EXACT_HASH_OVERLAP_CHECK", "1") == "1"
RUN_DISTANCE_OVERLAP_CHECK = os.environ.get("RUN_DISTANCE_OVERLAP_CHECK", "0") == "1"
OVERLAP_TOL = float(os.environ.get("OVERLAP_TOL", "1.0e-10"))

SAVE_PLOTS = os.environ.get("SAVE_PLOTS", "1") == "1"
SHOW_PLOTS = False
MAKE_TRAINING_CURVE_PLOTS = os.environ.get("MAKE_TRAINING_CURVE_PLOTS", "1") == "1"
MAKE_L2_PER_TIME_PLOT = os.environ.get("MAKE_L2_PER_TIME_PLOT", "1") == "1"
MAKE_PER_TEST_L2_PLOTS = os.environ.get("MAKE_PER_TEST_L2_PLOTS", "0") == "1"
MAKE_ANNULUS_PLOTS = os.environ.get("MAKE_ANNULUS_PLOTS", "1") == "1"
MAKE_BEST_WORST_PLOTS = os.environ.get("MAKE_BEST_WORST_PLOTS", "1") == "1"
N_BEST_WORST = int(os.environ.get("N_BEST_WORST", "2"))
MAX_TEST_CASES_TO_PLOT = int(os.environ.get("MAX_TEST_CASES_TO_PLOT", "6"))
PLOT_DPI = int(os.environ.get("PLOT_DPI", "140"))
L2_CHUNK_SIZE = int(os.environ.get("L2_CHUNK_SIZE", "50"))

USE_TRAPEZOIDAL_TIME_L2 = os.environ.get("USE_TRAPEZOIDAL_TIME_L2", "1") == "1"
INCLUDE_T0_IN_FULL_L2 = os.environ.get("INCLUDE_T0_IN_FULL_L2", "1") == "1"

# Manual plot times.
PLOT_TIME_INDICES_MANUAL = [20,35,60, "final"]

EXPERIMENT_NAME = "annulus2d_awfno_sample_sweep"

CACHE_ROOT = OUTPUT_ROOT / (
    f"cache_{EXPERIMENT_NAME}_"
    f"sizes{'-'.join(map(str, ALL_SAMPLE_SIZES))}_"
    f"K{WINDOW_K}_M{N_TIME_MODES}_W{WIDTH}_mr{MODES_R}_mt{MODES_THETA}_R{LOCAL_RADIUS}"
)
MODEL_DIR = os.path.join(CACHE_ROOT, "models")
PLOT_DIR = os.path.join(CACHE_ROOT, "plots")
DATA_DIR = os.path.join(CACHE_ROOT, "data")
RESULTS_DIR = os.path.join(CACHE_ROOT, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcdefaults()
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "legend.fontsize": 11,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

pin_memory = device.type == "cuda"


# ============================================================
# 2. GLOBAL GEOMETRY
# ============================================================

r_grid_np = None
theta_grid_np = None
t_grid = None

Nr = None
Ntheta = None
Nt = None

DR = None
DTHETA = None
DT_SNAPSHOT = 1.0

WR_L2 = None
BASE_TAG = "uninitialized"


def make_annulus_quadrature_weights_from_grid(r_grid, ntheta):
    """Trapezoidal in r and periodic rectangle in theta for integral f r dr dtheta."""
    nr = len(r_grid)
    if nr > 1:
        dr = float(np.mean(np.diff(r_grid.astype(np.float64))))
    else:
        dr = 1.0
    dtheta = float(2.0 * np.pi / ntheta)

    wr = r_grid.astype(np.float64).copy() * dr
    if nr > 1:
        wr[0] *= 0.5
        wr[-1] *= 0.5
    return wr.astype(np.float32), dtheta, dr


def set_geometry_from_arrays(U_shape, r_grid, theta_grid, t=None):
    global r_grid_np, theta_grid_np, t_grid
    global Nr, Ntheta, Nt, DR, DTHETA, DT_SNAPSHOT, WR_L2

    if len(U_shape) != 4:
        raise ValueError(f"Expected U shape (N,Nr,Ntheta,Nt). Got {U_shape}")

    _, Nr, Ntheta, Nt = U_shape

    r_grid_np = np.asarray(r_grid, dtype=np.float32).reshape(-1)
    theta_grid_np = np.asarray(theta_grid, dtype=np.float32).reshape(-1)

    if len(r_grid_np) != Nr:
        raise ValueError(f"r_grid length {len(r_grid_np)} does not match Nr={Nr}")
    if len(theta_grid_np) != Ntheta:
        raise ValueError(f"theta_grid length {len(theta_grid_np)} does not match Ntheta={Ntheta}")

    WR_L2, DTHETA, DR = make_annulus_quadrature_weights_from_grid(r_grid_np, Ntheta)

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
    print("N, Nr, Ntheta, Nt:", U_shape[0], Nr, Ntheta, Nt)
    print("r range          :", float(r_grid_np.min()), float(r_grid_np.max()))
    print("theta range      :", float(theta_grid_np.min()), float(theta_grid_np.max()))
    print("DR               :", DR)
    print("DTHETA           :", DTHETA)
    print("DT_SNAPSHOT      :", DT_SNAPSHOT)


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
    import re
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return text.strip("_") or "item"


def print_array_memory(name, arr):
    gb = arr.nbytes / 1024**3
    print(f"{name:24s}: shape={arr.shape}, dtype={arr.dtype}, memory={gb:.3f} GB")


# ============================================================
# 4. DATA LOADING / SPLIT
# ============================================================

def load_annulus_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Annulus dataset file not found: {path}\n"
            "Copy it locally first, or "
            "set os.environ['DATA_PATH'] to the correct path."
        )

    data = torch.load(path, map_location="cpu")

    if "c_grid" in data:
        arr = data["c_grid"]
    elif "Y_output" in data:
        arr = data["Y_output"]
    else:
        raise KeyError("Dataset must contain either 'c_grid' or 'Y_output'.")

    arr = arr.float().numpy() if torch.is_tensor(arr) else np.asarray(arr, dtype=np.float32)
    if arr.ndim != 5 or arr.shape[-1] != 1:
        raise ValueError(f"Expected c_grid/Y_output shape (N,T,Nr,Ntheta,1). Got {arr.shape}")

    if "r_grid" not in data:
        raise KeyError("Dataset must contain 'r_grid'.")
    if "theta_grid" not in data:
        raise KeyError("Dataset must contain 'theta_grid'.")

    r = data["r_grid"]
    theta = data["theta_grid"]
    r = r.float().numpy() if torch.is_tensor(r) else np.asarray(r, dtype=np.float32)
    theta = theta.float().numpy() if torch.is_tensor(theta) else np.asarray(theta, dtype=np.float32)

    if "t_grid" in data:
        t = data["t_grid"]
        t = t.float().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    elif "t" in data:
        t = data["t"]
        t = t.float().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    elif "timestep_indices" in data:
        t = data["timestep_indices"]
        t = t.float().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)
    else:
        t = np.arange(arr.shape[1], dtype=np.float32)

    sample_ids = data.get("sample_ids", torch.arange(arr.shape[0]))
    sample_ids = sample_ids.long().numpy() if torch.is_tensor(sample_ids) else np.asarray(sample_ids)

    if TIME_SUBSAMPLE > 1:
        arr = arr[:, ::TIME_SUBSAMPLE]
        t = t[::TIME_SUBSAMPLE]

    U = np.transpose(arr[..., 0], (0, 2, 3, 1)).astype(np.float32)
    if not np.all(np.isfinite(U)):
        raise ValueError("Loaded annulus solution array contains NaN or Inf values.")

    print("\nLoaded annulus data")
    print("-------------------")
    print("file:", path)
    print_array_memory("U", U)
    print("r shape:", r.shape)
    print("theta shape:", theta.shape)
    print("t shape:", t.shape)
    print("U min/max:", float(U.min()), float(U.max()))
    return U, r.astype(np.float32), theta.astype(np.float32), t.astype(np.float32), sample_ids


def downsample_polar_grid(U, r, theta, target_nr=None, target_ntheta=None):
    if target_nr is None and target_ntheta is None:
        return U, r, theta

    U = np.asarray(U, dtype=np.float32)
    r = np.asarray(r, dtype=np.float32).reshape(-1)
    theta = np.asarray(theta, dtype=np.float32).reshape(-1)
    _, nr0, nth0, _ = U.shape

    fr = 1
    if target_nr is not None:
        if target_nr > nr0 or nr0 % target_nr != 0:
            raise ValueError(f"Need Nr={nr0} divisible by TARGET_NR={target_nr}")
        fr = nr0 // target_nr

    fth = 1
    if target_ntheta is not None:
        if target_ntheta > nth0 or nth0 % target_ntheta != 0:
            raise ValueError(f"Need Ntheta={nth0} divisible by TARGET_NTHETA={target_ntheta}")
        fth = nth0 // target_ntheta

    U_down = U[:, ::fr, ::fth, :]
    r_down = r[::fr]
    theta_down = theta[::fth]

    print("\nPolar-grid downsampling")
    print("-----------------------")
    print("original U shape:", U.shape)
    print("stride r/theta  :", fr, fth)
    print("new U shape     :", U_down.shape)
    return U_down.astype(np.float32), r_down.astype(np.float32), theta_down.astype(np.float32)


def make_train_val_test_splits(U_all):
    needed = N_TRAIN_POOL + N_VAL + N_TEST
    if U_all.shape[0] < needed:
        raise ValueError(
            f"Need at least {needed} trajectories for split, but dataset has {U_all.shape[0]}. "
            f"N_TRAIN_POOL={N_TRAIN_POOL}, N_VAL={N_VAL}, N_TEST={N_TEST}."
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


def save_split_dataset_npz(path, U_train_pool, train_ids, U_val, val_ids, U_test, test_ids, r, theta, t):
    np.savez_compressed(
        path,
        U_train_pool=U_train_pool,
        train_ids=train_ids,
        U_val=U_val,
        val_ids=val_ids,
        U_test=U_test,
        test_ids=test_ids,
        r_grid=r,
        theta_grid=theta,
        t_grid=t,
        sample_sizes=np.array(SAMPLE_SIZES, dtype=np.int64),
        all_sample_sizes=np.array(ALL_SAMPLE_SIZES, dtype=np.int64),
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
# 5. PHYSICAL ANNULUS L2 METRICS
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


def _wr_torch(device_, dtype, with_time=False):
    if with_time:
        return torch.tensor(WR_L2, device=device_, dtype=dtype).view(1, Nr, 1, 1)
    return torch.tensor(WR_L2, device=device_, dtype=dtype).view(1, Nr, 1)


def abs_l2_window_batch(pred, true):
    wtime = _time_weights_torch(pred.shape[-1], pred.device, pred.dtype).view(1, 1, 1, -1)
    wr = _wr_torch(pred.device, pred.dtype, with_time=True)
    err2 = (pred - true) ** 2
    integral = torch.sum(err2 * wr * wtime, dim=(1, 2, 3)) * DTHETA * DT_SNAPSHOT
    return torch.sqrt(integral + 1.0e-30)


def abs_l2_field_batch(pred, true):
    wr = _wr_torch(pred.device, pred.dtype, with_time=False)
    integral = torch.sum((pred - true) ** 2 * wr, dim=(1, 2)) * DTHETA
    return torch.sqrt(integral + 1.0e-30)


def l2_per_timestep_np(pred, true, chunk_size=L2_CHUNK_SIZE):
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")
    n, nr, nth, nt = pred.shape
    if nr != Nr or nth != Ntheta:
        raise ValueError(f"Geometry mismatch: array has {(nr,nth)}, globals {(Nr,Ntheta)}")

    out = np.zeros((n, nt), dtype=np.float64)
    wr = WR_L2.astype(np.float32).reshape(1, Nr, 1, 1)
    for i0 in range(0, n, chunk_size):
        i1 = min(i0 + chunk_size, n)
        err = pred[i0:i1].astype(np.float32, copy=False) - true[i0:i1].astype(np.float32, copy=False)
        err2 = err * err
        integral_t = np.sum(err2 * wr, axis=(1, 2), dtype=np.float64) * DTHETA
        out[i0:i1] = np.sqrt(integral_t + 1.0e-30)
        del err, err2, integral_t
    gc.collect()
    return out


def abs_l2_full_traj_np(pred, true):
    pred_use = pred if INCLUDE_T0_IN_FULL_L2 else pred[:, :, :, 1:]
    true_use = true if INCLUDE_T0_IN_FULL_L2 else true[:, :, :, 1:]
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
    N, nr, nth, _ = U.shape
    u0 = np.zeros((N * len(starts), nr, nth), dtype=np.float32)
    Uf = np.zeros((N * len(starts), nr, nth, WINDOW_K), dtype=np.float32)
    q = 0
    for i in range(N):
        for s in starts:
            u0[q] = U[i, :, :, s]
            Uf[q] = U[i, :, :, s + 1:s + WINDOW_K + 1]
            q += 1
    return u0, Uf


def compute_temporal_basis_from_windows(u_start, U_future, n_modes):
    R = U_future - u_start[:, :, :, None]
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
    R = torch.einsum("bxym,km->bxyk", coeff, basis)
    Uf = u_start.unsqueeze(-1) + R
    if enforce_mean:
        mean0 = u_start.mean(dim=(1, 2), keepdim=True).unsqueeze(-1)
        Uf = Uf + mean0 - Uf.mean(dim=(1, 2), keepdim=True)
    return Uf


def enforce_mean_future(U_future, u_start):
    mean0 = u_start.mean(dim=(1, 2), keepdim=True).unsqueeze(-1)
    return U_future + mean0 - U_future.mean(dim=(1, 2), keepdim=True)


# ============================================================
# 7. DATASET / NORMALIZER
# ============================================================

class WindowDataset2D(Dataset):
    def __init__(self, u_start, U_future):
        self.u_start = torch.tensor(u_start, dtype=torch.float32)
        self.U_future = torch.tensor(U_future, dtype=torch.float32)

    def __len__(self):
        return self.u_start.shape[0]

    def __getitem__(self, i):
        return {"u_start": self.u_start[i], "U_future": self.U_future[i]}


class FullTrajectoryDataset2D(Dataset):
    def __init__(self, U):
        U = np.asarray(U, dtype=np.float32)
        if U.ndim != 4:
            raise ValueError(f"Expected U with shape (N,Nr,Ntheta,Nt). Got {U.shape}")
        if U.shape[-1] < 2:
            raise ValueError("Need at least two saved time snapshots for full-trajectory prediction.")
        self.u0 = torch.tensor(U[:, :, :, 0], dtype=torch.float32)
        self.U_future = torch.tensor(U[:, :, :, 1:], dtype=torch.float32)

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

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_r, modes_theta):
        super().__init__()
        self.modes_r = modes_r
        self.modes_theta = modes_theta
        scale = 1.0 / np.sqrt(in_channels * out_channels)
        self.weights_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_r, modes_theta, dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_r, modes_theta, dtype=torch.cfloat)
        )

    @staticmethod
    def compl_mul2d(a, b):
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        B, _, nr, nth = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            B,
            self.weights_pos.shape[1],
            nr,
            nth // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        mr = min(self.modes_r, nr)
        mt = min(self.modes_theta, nth // 2 + 1)
        if mr > 0 and mt > 0:
            out_ft[:, :, :mr, :mt] = self.compl_mul2d(x_ft[:, :, :mr, :mt], self.weights_pos[:, :, :mr, :mt])
            out_ft[:, :, -mr:, :mt] = self.compl_mul2d(x_ft[:, :, -mr:, :mt], self.weights_neg[:, :, :mr, :mt])
        return torch.fft.irfft2(out_ft, s=(nr, nth), norm="ortho")


class LocalOperator2d(nn.Module):
    def __init__(self, width, radius):
        super().__init__()
        self.radius = int(radius)
        k = 2 * self.radius + 1
        self.op = nn.Conv2d(width, width, kernel_size=k, padding=0, bias=True)

    def polar_pad(self, z):
        p = self.radius
        if p == 0:
            return z
        z = F.pad(z, (p, p, 0, 0), mode="circular")
        z = F.pad(z, (0, 0, p, p), mode="replicate")
        return z

    def forward(self, z):
        return self.op(self.polar_pad(z))


class FourierOperator2d(nn.Module):
    def __init__(self, width, modes_r, modes_theta):
        super().__init__()
        self.op = SpectralConv2d(width, width, modes_r, modes_theta)

    def forward(self, z):
        return self.op(z)


class AlternatingLayer2d(nn.Module):
    r"""One L-K-L residual alternating layer."""
    def __init__(self, width, modes_r, modes_theta, local_radius):
        super().__init__()
        self.L_l1 = LocalOperator2d(width, local_radius)
        self.K_l = FourierOperator2d(width, modes_r, modes_theta)
        self.L_l2 = LocalOperator2d(width, local_radius)
        self.sigma = nn.GELU()

    def forward(self, v_l):
        v_l_1 = v_l + self.sigma(self.L_l1(v_l))
        v_l_2 = v_l_1 + self.sigma(torch.tanh(self.K_l(v_l_1)))
        return v_l_2 + self.sigma(self.L_l2(v_l_2))


class AlternatingEncoder2d(nn.Module):
    def __init__(self, width, n_layers, modes_r, modes_theta, local_radius, r_grid, theta_grid):
        super().__init__()
        r = torch.tensor(r_grid, dtype=torch.float32)
        theta = torch.tensor(theta_grid, dtype=torch.float32)
        rn = (r - r.min()) / (r.max() - r.min() + 1.0e-12)
        R, TH = torch.meshgrid(rn, theta, indexing="ij")
        grid = torch.stack([R, torch.sin(TH), torch.cos(TH)], dim=-1)
        self.register_buffer("grid", grid.unsqueeze(0), persistent=False)
        self.input_proj = nn.Linear(4, width)
        self.blocks = nn.ModuleList([
            AlternatingLayer2d(width, modes_r, modes_theta, local_radius)
            for _ in range(n_layers)
        ])

    def forward(self, u_start_norm):
        B, _, _ = u_start_norm.shape
        grid = self.grid.expand(B, -1, -1, -1)
        z = self.input_proj(torch.cat([u_start_norm.unsqueeze(-1), grid], dim=-1))
        z = z.permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            z = block(z)
        return z.permute(0, 2, 3, 1).contiguous()


class TBAWFNO2d(nn.Module):
    def __init__(self, n_modes, r_grid, theta_grid):
        super().__init__()
        self.encoder = AlternatingEncoder2d(WIDTH, N_LAYERS, MODES_R, MODES_THETA, LOCAL_RADIUS, r_grid, theta_grid)
        self.fc1 = nn.Linear(WIDTH, 128)
        self.fc2 = nn.Linear(128, n_modes)
        self.act = nn.GELU()

    def forward(self, u_start_norm):
        z = self.encoder(u_start_norm)
        return self.fc2(self.act(self.fc1(z))).contiguous()


class AWFNO2d(nn.Module):
    def __init__(self, window_k, r_grid, theta_grid):
        super().__init__()
        self.encoder = AlternatingEncoder2d(WIDTH, N_LAYERS, MODES_R, MODES_THETA, LOCAL_RADIUS, r_grid, theta_grid)
        self.fc1 = nn.Linear(WIDTH, 128)
        self.fc2 = nn.Linear(128, window_k)
        self.act = nn.GELU()

    def forward(self, u_start_norm):
        z = self.encoder(u_start_norm)
        return self.fc2(self.act(self.fc1(z))).contiguous()


class WFNO2d(nn.Module):
    def __init__(self, window_k, r_grid, theta_grid):
        super().__init__()
        r = torch.tensor(r_grid, dtype=torch.float32)
        theta = torch.tensor(theta_grid, dtype=torch.float32)
        rn = (r - r.min()) / (r.max() - r.min() + 1.0e-12)
        R, TH = torch.meshgrid(rn, theta, indexing="ij")
        grid = torch.stack([R, torch.sin(TH), torch.cos(TH)], dim=-1)
        self.register_buffer("grid", grid.unsqueeze(0), persistent=False)
        self.fc0 = nn.Linear(4, WIDTH)
        self.convs = nn.ModuleList([SpectralConv2d(WIDTH, WIDTH, MODES_R, MODES_THETA) for _ in range(N_LAYERS)])
        self.ws = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 1) for _ in range(N_LAYERS)])
        self.fc1 = nn.Linear(WIDTH, 128)
        self.fc2 = nn.Linear(128, window_k)

    def forward(self, u_start_norm):
        B, _, _ = u_start_norm.shape
        grid = self.grid.expand(B, -1, -1, -1)
        z = self.fc0(torch.cat([u_start_norm.unsqueeze(-1), grid], dim=-1))
        z = z.permute(0, 3, 1, 2).contiguous()
        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            z = conv(z) + w(z)
            if i != len(self.convs) - 1:
                z = F.gelu(z)
        z = z.permute(0, 2, 3, 1).contiguous()
        return self.fc2(F.gelu(self.fc1(z))).contiguous()


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


class NeuralOperatorFNOFull2d(nn.Module):
    def __init__(self, nt, r_grid, theta_grid):
        super().__init__()
        if not USE_LIBRARY_FNO:
            raise RuntimeError("USE_LIBRARY_FNO=False, but NeuralOperatorFNOFull2d was requested.")
        try:
            from neuralop.models import FNO as NeuralOpFNO
        except Exception as exc:
            raise ImportError("Install neuraloperator first: pip install neuraloperator") from exc

        self.nt = int(nt)
        if self.nt < 2:
            raise ValueError(f"Need nt >= 2. Got nt={nt}")

        r = torch.tensor(r_grid, dtype=torch.float32)
        theta = torch.tensor(theta_grid, dtype=torch.float32)
        rn = (r - r.min()) / (r.max() - r.min() + 1.0e-12)
        R, TH = torch.meshgrid(rn, theta, indexing="ij")
        grid = torch.stack([R, torch.sin(TH), torch.cos(TH)], dim=0)
        self.register_buffer("grid", grid.unsqueeze(0), persistent=False)

        kwargs = dict(
            n_modes=(min(MODES_R, len(r_grid)), min(MODES_THETA, len(theta_grid) // 2 + 1)),
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
        B, nr, nth = u0_norm.shape
        grid = self.grid.expand(B, -1, -1, -1)
        x_in = torch.cat([u0_norm.unsqueeze(1), grid], dim=1)
        out = self.fno(x_in)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 4:
            raise RuntimeError(f"Expected neuralop FNO output with shape (B,C,Nr,Ntheta). Got {tuple(out.shape)}")
        if out.shape[2] == nr and out.shape[3] == nth:
            return out.permute(0, 2, 3, 1).contiguous()
        if out.shape[1] == nr and out.shape[2] == nth:
            return out.contiguous()
        raise RuntimeError(f"Cannot interpret neuralop FNO output shape {tuple(out.shape)} for nr/nth={(nr,nth)}")


def make_full_fno_predict(model, u0_normer, full_future_normer):
    def predict(u0):
        Uf_norm = model(u0_normer.encode(u0))
        return full_future_normer.decode(Uf_norm)
    return predict


# ============================================================
# 9. LOSSES / TRAINING
# ============================================================

def tb_awfno_loss(Uf_pred, Uf_true):
    loss_recon = F.mse_loss(Uf_pred, Uf_true)
    loss_final = F.mse_loss(Uf_pred[:, :, :, -1], Uf_true[:, :, :, -1])
    return loss_recon + FINAL_LOSS_WEIGHT * loss_final


def awfno_loss(Uf_pred, Uf_true):
    loss_recon = F.mse_loss(Uf_pred, Uf_true)
    loss_final = F.mse_loss(Uf_pred[:, :, :, -1], Uf_true[:, :, :, -1])
    return loss_recon + FINAL_LOSS_WEIGHT * loss_final


def wfno_loss(Uf_pred, Uf_true):
    return F.mse_loss(Uf_pred, Uf_true)


def train_epoch_tb_awfno(model, loader, opt, predict):
    model.train()
    total_loss = 0.0
    total_l2 = 0.0
    n = 0
    for batch in loader:
        u0 = batch["u_start"].to(device, non_blocking=True)
        Uf_true = batch["U_future"].to(device, non_blocking=True)
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
        u0 = batch["u_start"].to(device, non_blocking=True)
        Uf_true = batch["U_future"].to(device, non_blocking=True)
        _, Uf_pred = predict(u0)
        loss = tb_awfno_loss(Uf_pred, Uf_true)
        bs = u0.shape[0]
        total_loss += loss.item() * bs
        total_l2 += abs_l2_window_batch(Uf_pred, Uf_true).mean().item() * bs
        total_final += abs_l2_field_batch(Uf_pred[:, :, :, -1], Uf_true[:, :, :, -1]).mean().item() * bs
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
        u0 = batch["u_start"].to(device, non_blocking=True)
        Uf_true = batch["U_future"].to(device, non_blocking=True)
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
        u0 = batch["u_start"].to(device, non_blocking=True)
        Uf_true = batch["U_future"].to(device, non_blocking=True)
        Uf_pred = predict(u0)
        loss = awfno_loss(Uf_pred, Uf_true) if kind == "awfno" else wfno_loss(Uf_pred, Uf_true)
        bs = u0.shape[0]
        total_loss += loss.item() * bs
        total_l2 += abs_l2_window_batch(Uf_pred, Uf_true).mean().item() * bs
        total_final += abs_l2_field_batch(Uf_pred[:, :, :, -1], Uf_true[:, :, :, -1]).mean().item() * bs
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
        u0 = batch["u0"].to(device, non_blocking=True)
        Uf_true = batch["U_future"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        Uf_pred = predict(u0)
        loss_recon = F.mse_loss(Uf_pred, Uf_true)
        loss = loss_recon
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
        u0 = batch["u0"].to(device, non_blocking=True)
        Uf_true = batch["U_future"].to(device, non_blocking=True)
        Uf_pred = predict(u0)
        loss_recon = F.mse_loss(Uf_pred, Uf_true)
        loss_final = F.mse_loss(Uf_pred[:, :, :, -1], Uf_true[:, :, :, -1])
        loss = loss_recon + FINAL_LOSS_WEIGHT * loss_final
        bs = u0.shape[0]
        total_loss += loss.item() * bs
        total_l2 += abs_l2_window_batch(Uf_pred, Uf_true).mean().item() * bs
        total_final += abs_l2_field_batch(Uf_pred[:, :, :, -1], Uf_true[:, :, :, -1]).mean().item() * bs
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
# 10. ROLLOUT / DIRECT FULL-TRAJECTORY PREDICTION
# ============================================================

@torch.no_grad()
def autoregressive_rollout_tb_awfno(model, u0_test_np, nt, predict, print_diagnostics=True):
    model.eval()
    N, nr, nth = u0_test_np.shape
    U_roll = np.zeros((N, nr, nth, nt), dtype=np.float32)
    u_current = torch.tensor(u0_test_np, dtype=torch.float32, device=device)
    U_roll[:, :, :, 0] = u0_test_np
    cursor = 0
    block = 0
    print("\nAutoregressive rollout: TB-AWFNO")
    while cursor < nt - 1:
        _, Uf_pred = predict(u_current)
        use_k = min(WINDOW_K, nt - 1 - cursor)
        U_roll[:, :, :, cursor + 1:cursor + 1 + use_k] = Uf_pred[:, :, :, :use_k].detach().cpu().numpy()
        u_next = Uf_pred[:, :, :, use_k - 1].detach()
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
    N, nr, nth = u0_test_np.shape
    U_roll = np.zeros((N, nr, nth, nt), dtype=np.float32)
    u_current = torch.tensor(u0_test_np, dtype=torch.float32, device=device)
    U_roll[:, :, :, 0] = u0_test_np
    cursor = 0
    block = 0
    print(f"\nAutoregressive rollout: {name}")
    while cursor < nt - 1:
        Uf_pred = predict(u_current)
        use_k = min(WINDOW_K, nt - 1 - cursor)
        U_roll[:, :, :, cursor + 1:cursor + 1 + use_k] = Uf_pred[:, :, :, :use_k].detach().cpu().numpy()
        u_next = Uf_pred[:, :, :, use_k - 1].detach()
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
    N, nr, nth = u0_test_np.shape
    U_pred = np.zeros((N, nr, nth, nt), dtype=np.float32)
    U_pred[:, :, :, 0] = u0_test_np
    print(f"\nDirect full-trajectory prediction: {name}")
    for i0 in range(0, N, batch_size):
        i1 = min(i0 + batch_size, N)
        u0 = torch.tensor(u0_test_np[i0:i1], dtype=torch.float32, device=device)
        Uf_pred = predict(u0)
        U_pred[i0:i1, :, :, 1:] = Uf_pred.detach().cpu().numpy()
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
        if item == "final":
            out.append(nt - 1)
        else:
            out.append(int(item))
    clean = []
    for k in out:
        if 0 <= k < nt and k not in clean:
            clean.append(k)
    if not clean:
        clean = [0, nt // 2, nt - 1]
    return clean


def polar_field_to_closed_annulus(field_2d):
    theta_closed = np.concatenate([theta_grid_np, np.array([theta_grid_np[0] + 2.0 * np.pi], dtype=theta_grid_np.dtype)])
    field_closed = np.concatenate([field_2d, field_2d[:, :1]], axis=1)
    R, TH = np.meshgrid(r_grid_np, theta_closed, indexing="ij")
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    return X, Y, field_closed


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
    path = os.path.join(PLOT_DIR, f"{safe_name(name)}_loss_curves_{BASE_TAG}.png")
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
    path = os.path.join(PLOT_DIR, f"{safe_name(name)}_l2_curves_{BASE_TAG}.png")
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)


def plot_temporal_basis(basis_np, singular_values):
    if not SAVE_PLOTS:
        return
    out_dir = os.path.join(PLOT_DIR, "temporal_basis")
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for m in range(basis_np.shape[1]):
        plt.plot(np.arange(1, WINDOW_K + 1), basis_np[:, m], label=f"mode {m + 1}")
    plt.xlabel("local step")
    plt.ylabel("basis value")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"temporal_basis_{BASE_TAG}.png")
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)

    plt.figure(figsize=(8, 5))
    plt.semilogy(singular_values, marker="o")
    plt.xlabel("index")
    plt.ylabel("singular value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"temporal_basis_singular_values_{BASE_TAG}.png")
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)


def plot_l2_per_time_compare_all(tb, aw, wf, fn):
    if not SAVE_PLOTS or not MAKE_L2_PER_TIME_PLOT:
        return
    out_dir = os.path.join(PLOT_DIR, "l2_error_per_timestep")
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(t_grid, tb["mean_t"], label="TB-AWFNO mean", linewidth=2)
    plt.plot(t_grid, aw["mean_t"], label="AWFNO mean", linewidth=2)
    plt.plot(t_grid, wf["mean_t"], label="WFNO mean", linewidth=2)
    plt.plot(t_grid, fn["mean_t"], label="FNO mean", linewidth=2)
    plt.plot(t_grid, tb["median_t"], "--", label="TB-AWFNO median")
    plt.plot(t_grid, aw["median_t"], "--", label="AWFNO median")
    plt.plot(t_grid, wf["median_t"], "--", label="WFNO median")
    plt.plot(t_grid, fn["median_t"], "--", label="FNO median")
    plt.xlabel("time index")
    plt.ylabel(r"$E_{L^2(\Omega)}$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"all_tests_mean_median_L2_error_{BASE_TAG}.png")
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)


def plot_per_test_l2_curves(l2_dict):
    if not SAVE_PLOTS or not MAKE_PER_TEST_L2_PLOTS:
        return
    out_dir = os.path.join(PLOT_DIR, "l2_error_per_timestep")
    os.makedirs(out_dir, exist_ok=True)
    n_available = next(iter(l2_dict.values())).shape[0]
    n_plot = n_available if MAX_TEST_CASES_TO_PLOT is None else min(MAX_TEST_CASES_TO_PLOT, n_available)
    for i in range(n_plot):
        plt.figure(figsize=(8, 5))
        for name, l2_t in l2_dict.items():
            plt.plot(t_grid, l2_t[i], label=name, linewidth=2)
        plt.xlabel("time")
        plt.ylabel(r"$E_{L^2(\Omega)}$")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        path = os.path.join(out_dir, f"test_{i:03d}_L2_error_per_timestep.png")
        plt.savefig(path, dpi=PLOT_DPI)
        plt.close()
        print("Saved:", path)


def plot_annulus_cases(U_true, rollouts):
    if not SAVE_PLOTS or not MAKE_ANNULUS_PLOTS:
        return

    out_dir = os.path.join(PLOT_DIR, "selected_test_case_annulus")
    os.makedirs(out_dir, exist_ok=True)

    time_ids = get_plot_time_indices(U_true.shape[-1])
    n_available = U_true.shape[0]
    n_plot = min(MAX_TEST_CASES_TO_PLOT, n_available)

    model_names = list(rollouts.keys())

    for i in range(n_plot):
        nrows = len(time_ids)

        # Only true + model predictions. No error columns.
        ncols = 1 + len(model_names)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.0 * ncols, 3.0 * nrows),
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(nrows, ncols)

        # Shared color scale for true and all predictions.
        state_fields = []
        for tidx in time_ids:
            true_field = U_true[i, :, :, tidx]
            state_fields.append(true_field)

            for name in model_names:
                pred = rollouts[name][i, :, :, tidx]
                state_fields.append(pred)

        vmin = min(float(f.min()) for f in state_fields)
        vmax = max(float(f.max()) for f in state_fields)

        im_state = None

        for row, tidx in enumerate(time_ids):
            true_field = U_true[i, :, :, tidx]

            col = 0

            # True field
            X, Y, Fclosed = polar_field_to_closed_annulus(true_field)
            im_state = axes[row, col].pcolormesh(
                X,
                Y,
                Fclosed,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
            )
            axes[row, col].set_aspect("equal")
            axes[row, col].set_title("true" if row == 0 else "")
            axes[row, col].set_ylabel(f"t={tidx}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            col += 1

            # Model predictions
            for name in model_names:
                pred = rollouts[name][i, :, :, tidx]
                X, Y, Fclosed = polar_field_to_closed_annulus(pred)

                axes[row, col].pcolormesh(
                    X,
                    Y,
                    Fclosed,
                    shading="auto",
                    vmin=vmin,
                    vmax=vmax,
                )
                axes[row, col].set_aspect("equal")
                axes[row, col].set_title(name if row == 0 else "")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                col += 1

        fig.colorbar(
            im_state,
            ax=axes.ravel().tolist(),
            shrink=0.8,
            label="c",
        )

        path = os.path.join(out_dir, f"test_{i:03d}_annulus_fields.png")
        plt.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

        print("Saved:", path)


def plot_one_ranked_case(idx, rank_name, U_true, rollouts, l2_dict, full_l2_reference, out_dir):
    time_ids = get_plot_time_indices(U_true.shape[-1])
    model_names = list(rollouts.keys())
    nrows = len(time_ids) + 1
    ncols = 1 + len(model_names)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), constrained_layout=True)
    axes = np.asarray(axes).reshape(nrows, ncols)
    for row, tidx in enumerate(time_ids):
        true_field = U_true[idx, :, :, tidx]
        state_fields = [true_field] + [rollouts[name][idx, :, :, tidx] for name in model_names]
        vmin = min(float(f.min()) for f in state_fields)
        vmax = max(float(f.max()) for f in state_fields)
        col = 0
        X, Y, Fclosed = polar_field_to_closed_annulus(true_field)
        axes[row, col].pcolormesh(X, Y, Fclosed, shading="auto", vmin=vmin, vmax=vmax)
        axes[row, col].set_aspect("equal")
        axes[row, col].set_title("true" if row == 0 else "")
        axes[row, col].set_ylabel(f"t={tidx}")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        col += 1
        for name in model_names:
            pred = rollouts[name][idx, :, :, tidx]
            X, Y, Fclosed = polar_field_to_closed_annulus(pred)
            axes[row, col].pcolormesh(X, Y, Fclosed, shading="auto", vmin=vmin, vmax=vmax)
            axes[row, col].set_aspect("equal")
            axes[row, col].set_title(name if row == 0 else "")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            col += 1
    ax = axes[-1, 0]
    for name, l2_t in l2_dict.items():
        ax.plot(t_grid, l2_t[idx], label=name, linewidth=2)
    ax.set_title(f"{rank_name}: test {idx}, TB-AWFNO full L2 = {full_l2_reference[idx]:.6e}")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$E_{L^2(\Omega)}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    for c in range(1, ncols):
        axes[-1, c].axis("off")
    path = os.path.join(out_dir, f"{safe_name(rank_name)}_test_{idx:03d}.png")
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close(fig)
    print("Saved:", path)
    return path


def plot_best_worst_cases(U_true, rollouts, l2_dict, tb_stats, n_each=N_BEST_WORST):
    if not SAVE_PLOTS or not MAKE_BEST_WORST_PLOTS:
        return []
    out_dir = os.path.join(PLOT_DIR, "best_worst_test_cases")
    os.makedirs(out_dir, exist_ok=True)
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
    ranking_path = os.path.join(out_dir, f"best_worst_ranking_{BASE_TAG}.json")
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

    for label, stats, key in [
        ("TB-AWFNO", tb, "tb"),
        ("AWFNO", aw, "aw"),
        ("WFNO", wf, "wf"),
        ("FNO", fn, "fn"),
    ]:
        print(
            f"{label:<8} | L2(full) mean {stats['full_l2_mean']:.6e} | "
            f"std {stats['full_l2_std']:.6e} | median {stats['full_l2_median']:.6e} | "
            f"L2(final) mean {stats['final_l2_mean']:.6e} | final median {stats['final_l2_median']:.6e}{mt(key)}"
        )

    scores = {
        "TB-AWFNO": tb["full_l2_mean"],
        "AWFNO": aw["full_l2_mean"],
        "WFNO": wf["full_l2_mean"],
        "FNO": fn["full_l2_mean"],
    }
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
        "run_sample_sizes": SAMPLE_SIZES,
        "N_train": n_train,
        "N_train_pool": N_TRAIN_POOL,
        "N_val": N_VAL,
        "N_test": N_TEST,
        "Nr": Nr,
        "Ntheta": Ntheta,
        "Nt": Nt,
        "WINDOW_K": WINDOW_K,
        "WINDOW_STRIDE": WINDOW_STRIDE,
        "N_TIME_MODES": N_TIME_MODES,
        "WIDTH": WIDTH,
        "N_LAYERS": N_LAYERS,
        "MODES_R": MODES_R,
        "MODES_THETA": MODES_THETA,
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
    print_array_memory("u_train_win", u_train_win)
    print_array_memory("Uf_train_win", Uf_train_win)
    print_array_memory("u_val_win", u_val_win)
    print_array_memory("Uf_val_win", Uf_val_win)

    basis_np, basis_energy_ratio, singular_values = compute_temporal_basis_from_windows(
        u_train_win, Uf_train_win, N_TIME_MODES
    )
    basis_torch = torch.tensor(basis_np, dtype=torch.float32, device=device)
    print("\nTemporal-basis coefficient supervision")
    print("--------------------------------------")
    print("Not used. Coefficients are internal model outputs.")

    train_ds = WindowDataset2D(u_train_win, Uf_train_win)
    val_ds = WindowDataset2D(u_val_win, Uf_val_win)
    full_train_ds = FullTrajectoryDataset2D(U_train)
    full_val_ds = FullTrajectoryDataset2D(U_val)

    cpu_gen = torch.Generator()
    cpu_gen.manual_seed(SEED + n_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=cpu_gen, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)
    full_train_loader = DataLoader(full_train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=cpu_gen, pin_memory=pin_memory)
    full_val_loader = DataLoader(full_val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

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
    full_u0_normer = GaussianNormalizer(torch.tensor(U_train[:, :, :, 0], dtype=torch.float32)).to(device)
    full_future_normer = GaussianNormalizer(torch.tensor(U_train[:, :, :, 1:], dtype=torch.float32)).to(device)

    return {
        "U_train": U_train,
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

    tb_model = TBAWFNO2d(N_TIME_MODES, r_grid_np, theta_grid_np).to(device)
    aw_model = AWFNO2d(WINDOW_K, r_grid_np, theta_grid_np).to(device)
    wf_model = WFNO2d(WINDOW_K, r_grid_np, theta_grid_np).to(device)
    fn_model = NeuralOperatorFNOFull2d(Nt, r_grid_np, theta_grid_np).to(device)

    tb_predict = make_tb_awfno_predict(tb_model, u_normer, basis_torch)
    aw_predict = make_direct_predict(aw_model, u_normer, future_normer, ENFORCE_MEAN)
    wf_predict = make_direct_predict(wf_model, u_normer, future_normer, False)
    fn_predict = make_full_fno_predict(fn_model, full_u0_normer, full_future_normer)

    print("\nModels")
    print("------")
    print("Alternating block used by TB-AWFNO and AWFNO: L-K-L residual block")
    print(f"TB-AWFNO params: {count_params(tb_model):,}")
    print(f"AWFNO params   : {count_params(aw_model):,}")
    print(f"WFNO params    : {count_params(wf_model):,}")
    print(f"FNO params     : {count_params(fn_model):,}")

    tb_path = os.path.join(MODEL_DIR, f"N{n_train}_TB_AWFNO_{BASE_TAG}.pt")
    aw_path = os.path.join(MODEL_DIR, f"N{n_train}_AWFNO_{BASE_TAG}.pt")
    wf_path = os.path.join(MODEL_DIR, f"N{n_train}_WFNO_{BASE_TAG}.pt")
    fn_path = os.path.join(MODEL_DIR, f"N{n_train}_FNO_neuralop_full_{BASE_TAG}.pt")

    if os.path.exists(tb_path) and not FORCE_RETRAIN:
        print("\nLoading cached TB-AWFNO:", tb_path)
        meta = load_model_checkpoint(tb_path, tb_model)
        tb_fit = {"model": tb_model, "best_val_l2": meta.get("best_val_l2", float("nan")), "train_time_sec": meta.get("train_time_sec", 0.0), "history": meta.get("history", None)}
    else:
        print("\n================ Training TB-AWFNO ================")
        tb_fit = fit_tb_awfno_model(tb_model, train_loader, val_loader, tb_predict)
        save_model_checkpoint(tb_path, tb_fit["model"], {"best_val_l2": tb_fit["best_val_l2"], "train_time_sec": tb_fit["train_time_sec"], "history": tb_fit["history"], "n_train": n_train, "base_tag": BASE_TAG})
    if tb_fit["history"] is not None:
        plot_training_curves(tb_fit["history"], f"N{n_train}_TB-AWFNO")

    if os.path.exists(aw_path) and not FORCE_RETRAIN:
        print("\nLoading cached AWFNO:", aw_path)
        meta = load_model_checkpoint(aw_path, aw_model)
        aw_fit = {"model": aw_model, "best_val_l2": meta.get("best_val_l2", float("nan")), "train_time_sec": meta.get("train_time_sec", 0.0), "history": meta.get("history", None)}
    else:
        print("\n================ Training AWFNO ================")
        aw_fit = fit_direct_model(aw_model, train_loader, val_loader, aw_predict, "AWFNO", "awfno")
        save_model_checkpoint(aw_path, aw_fit["model"], {"best_val_l2": aw_fit["best_val_l2"], "train_time_sec": aw_fit["train_time_sec"], "history": aw_fit["history"], "n_train": n_train, "base_tag": BASE_TAG})
    if aw_fit["history"] is not None:
        plot_training_curves(aw_fit["history"], f"N{n_train}_AWFNO")

    if os.path.exists(wf_path) and not FORCE_RETRAIN:
        print("\nLoading cached WFNO:", wf_path)
        meta = load_model_checkpoint(wf_path, wf_model)
        wf_fit = {"model": wf_model, "best_val_l2": meta.get("best_val_l2", float("nan")), "train_time_sec": meta.get("train_time_sec", 0.0), "history": meta.get("history", None)}
    else:
        print("\n================ Training WFNO ================")
        wf_fit = fit_direct_model(wf_model, train_loader, val_loader, wf_predict, "WFNO", "wfno")
        save_model_checkpoint(wf_path, wf_fit["model"], {"best_val_l2": wf_fit["best_val_l2"], "train_time_sec": wf_fit["train_time_sec"], "history": wf_fit["history"], "n_train": n_train, "base_tag": BASE_TAG})
    if wf_fit["history"] is not None:
        plot_training_curves(wf_fit["history"], f"N{n_train}_WFNO")

    if os.path.exists(fn_path) and not FORCE_RETRAIN:
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

    u0_test = U_test[:, :, :, 0]
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
    plot_annulus_cases(U_test, rollouts)
    best_worst_paths = plot_best_worst_cases(U_test, rollouts, l2_dict, tb_stats, n_each=N_BEST_WORST)

    rollout_path = os.path.join(RESULTS_DIR, f"N{n_train}_common_test_rollouts_{BASE_TAG}.npz")
    np.savez_compressed(
        rollout_path,
        U_roll_TB_AWFNO=U_roll_tb,
        U_roll_AWFNO=U_roll_aw,
        U_roll_WFNO=U_roll_wf,
        U_roll_FNO=U_roll_fn,
        U_true=U_test,
        r_grid=r_grid_np,
        theta_grid=theta_grid_np,
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
        "rollout_path": rollout_path,
        "best_worst_plot_paths": best_worst_paths,
    }

    summary_path = os.path.join(RESULTS_DIR, f"N{n_train}_summary_common_test_{BASE_TAG}.json")
    summary = {
        "task": EXPERIMENT_NAME,
        "data_path": str(DATA_PATH),
        "N_train": n_train,
        "N_train_pool": N_TRAIN_POOL,
        "N_val": N_VAL,
        "N_test": N_TEST,
        "sample_sizes_run": SAMPLE_SIZES,
        "all_sample_sizes": ALL_SAMPLE_SIZES,
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
            "modes_r": MODES_R,
            "modes_theta": MODES_THETA,
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
    for key, label in [
        ("TB_AWFNO_full_l2_mean", "TB-AWFNO"),
        ("AWFNO_full_l2_mean", "AWFNO"),
        ("WFNO_full_l2_mean", "WFNO"),
        ("FNO_full_l2_mean", "FNO"),
    ]:
        plt.plot(Ns, [r[key] for r in rows_sorted], marker="o", label=label)
    plt.xlabel("number of training trajectories")
    plt.ylabel(r"mean absolute full-trajectory $L^2(0,T;L^2(\Omega))$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"sample_size_sweep_abs_full_L2_common_test_run_{'-'.join(map(str, SAMPLE_SIZES))}.png")
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)

    plt.figure(figsize=(8, 5))
    for key, label in [
        ("TB_AWFNO_final_l2_mean", "TB-AWFNO"),
        ("AWFNO_final_l2_mean", "AWFNO"),
        ("WFNO_final_l2_mean", "WFNO"),
        ("FNO_final_l2_mean", "FNO"),
    ]:
        plt.plot(Ns, [r[key] for r in rows_sorted], marker="o", label=label)
    plt.xlabel("number of training trajectories")
    plt.ylabel(r"mean absolute final-time $L^2(\Omega)$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"sample_size_sweep_abs_final_L2_common_test_run_{'-'.join(map(str, SAMPLE_SIZES))}.png")
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()
    print("Saved:", path)


def print_sample_size_summary(rows):
    rows_sorted = sorted(rows, key=lambda r: r["N_train"])
    print("\n================ SAMPLE-SIZE SWEEP SUMMARY: ABSOLUTE FULL L2 ================")
    for r in rows_sorted:
        print(
            f"N={r['N_train']:4d} | Abs full L2: "
            f"TB-AWFNO {r['TB_AWFNO_full_l2_mean']:.6e} | "
            f"AWFNO {r['AWFNO_full_l2_mean']:.6e} | "
            f"WFNO {r['WFNO_full_l2_mean']:.6e} | "
            f"FNO {r['FNO_full_l2_mean']:.6e} | winner {r['winner']}"
        )
    print("============================================================================\n")

    print("\n================ SAMPLE-SIZE SWEEP SUMMARY: ABSOLUTE FINAL L2 ================")
    for r in rows_sorted:
        final_scores = {
            "TB-AWFNO": r["TB_AWFNO_final_l2_mean"],
            "AWFNO": r["AWFNO_final_l2_mean"],
            "WFNO": r["WFNO_final_l2_mean"],
            "FNO": r["FNO_final_l2_mean"],
        }
        final_winner = min(final_scores, key=final_scores.get)
        print(
            f"N={r['N_train']:4d} | Abs final L2: "
            f"TB-AWFNO {r['TB_AWFNO_final_l2_mean']:.6e} | "
            f"AWFNO {r['AWFNO_final_l2_mean']:.6e} | "
            f"WFNO {r['WFNO_final_l2_mean']:.6e} | "
            f"FNO {r['FNO_final_l2_mean']:.6e} | winner {final_winner}"
        )
    print("=============================================================================\n")


# ============================================================
# 13. MAIN
# ============================================================

def main():
    print("\nExperiment")
    print("----------")
    print("name             :", EXPERIMENT_NAME)
    print("data path        :", DATA_PATH)
    print("output root      :", OUTPUT_ROOT)
    print("cache root       :", CACHE_ROOT)
    print("device           :", device)
    print("ALL_SAMPLE_SIZES :", ALL_SAMPLE_SIZES)
    print("SAMPLE_SIZES run :", SAMPLE_SIZES)
    print("N_TRAIN_POOL     :", N_TRAIN_POOL)
    print("N_VAL/N_TEST     :", N_VAL, N_TEST)
    print("EPOCHS           :", EPOCHS)

    U_all, r, theta, t, sample_ids = load_annulus_dataset(DATA_PATH)
    U_all, r, theta = downsample_polar_grid(U_all, r, theta, TARGET_NR, TARGET_NTHETA)
    set_geometry_from_arrays(U_all.shape, r, theta, t)

    U_train_pool, train_ids, U_val, val_ids, U_test, test_ids = make_train_val_test_splits(U_all)
    train_sample_ids = sample_ids[train_ids]
    val_sample_ids = sample_ids[val_ids]
    test_sample_ids = sample_ids[test_ids]

    split_path = os.path.join(DATA_DIR, "train_val_test_split.npz")
    save_split_dataset_npz(split_path, U_train_pool, train_sample_ids, U_val, val_sample_ids, U_test, test_sample_ids, r, theta, t)

    if RUN_EXACT_HASH_OVERLAP_CHECK:
        check_exact_hash_overlap(U_train_pool, U_val, "train_pool", "val")
        check_exact_hash_overlap(U_train_pool, U_test, "train_pool", "test")
        check_exact_hash_overlap(U_val, U_test, "val", "test")

    if RUN_DISTANCE_OVERLAP_CHECK:
        check_dataset_overlap(U_train_pool, U_test, "train_pool", "test", tol=OVERLAP_TOL)
        check_dataset_overlap(U_val, U_test, "val", "test", tol=OVERLAP_TOL)

    bundles = []
    for n_train in SAMPLE_SIZES:
        bundle = train_models_for_sample_size(n_train, U_train_pool, U_val)
        bundles.append(bundle)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    rows = []
    for bundle in bundles:
        row = evaluate_on_common_test_set(bundle, U_test, test_sample_ids)
        rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    rows_sorted = sorted(rows, key=lambda r: r["N_train"])
    summary_table_path = os.path.join(RESULTS_DIR, f"sample_size_sweep_summary_run_{'-'.join(map(str, SAMPLE_SIZES))}.json")
    with open(summary_table_path, "w") as f:
        json.dump(rows_sorted, f, indent=2)
    print("Saved sample-size summary:", summary_table_path)

    plot_sample_size_sweep(rows_sorted)
    print_sample_size_summary(rows_sorted)


if __name__ == "__main__":
    main()