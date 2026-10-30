#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controlled reviewer ablations for 2D Cahn--Hilliard AWFNO.

Expected dataset
----------------
PyTorch file containing
    X_input : (N, Nx, Ny, 1)
    Y_output: (N, T, Nx, Ny, 1)

The script is designed for Google Colab and stores checkpoints/results in
Google Drive.  Configure it with environment variables before `%run`.

Suites
------
core:
    AWFNO L-F-L, local-only L-I-L, Fourier-only I-F-I, standard WFNO,
    and a parameter-matched WFNO.
ordering:
    L-F-L, L-L-F, and F-L-L.  These contain exactly the same learned
    operators and differ only in their order.
sensitivity:
    local radius R, retained Fourier modes, and window length K.
tb:
    The paper configuration of TB-AWFNO and the direct parameter-matched
    WFNO control.  The TB-AWFNO basis is computed from training windows only.

All learned models use the same split, training windows, normalizers, loss,
AdamW optimizer, cosine scheduler, gradient clipping, early-model selection,
and Cahn--Hilliard mean/mass enforcement.  Test rollout observes only the
initial condition and then recursively feeds predictions.

Primary outputs
---------------
summary_<suite>_raw.csv
summary_<suite>_aggregated.csv
summary_<suite>.json
parameter_table_<suite>.csv
reviewer_<suite>_comparison.png

The diagnostics include full/final physical L2, fixed low/high 2D Fourier-band
errors, high-gradient interface and bulk RMSE, interface error-energy fraction,
and mass drift.
"""

import copy
import gc
import hashlib
import json
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================================================
# 0. ENVIRONMENT / CONFIG
# ============================================================

def env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else int(default)


def env_float(name, default):
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else float(default)


def env_str(name, default):
    return str(os.environ.get(name, default))


def env_int_list(name, default):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return list(default)
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def env_str_list(name, default=()):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return list(default)
    return [x.strip() for x in str(raw).split(",") if x.strip()]


DATA_FILE = Path(env_str(
    "DATA_FILE",
    "/content/drive/MyDrive/anus/cahn_hilliard_train_grid_32x32.pt",
))
OUTPUT_ROOT = Path(env_str(
    "REVIEWER_OUTPUT_ROOT",
    "/content/drive/MyDrive/anus/ch2d_reviewer_ablations",
))
EXPERIMENT_NAME = env_str("EXPERIMENT_NAME", "controlled_v1")
REVIEWER_SUITE = env_str("REVIEWER_SUITE", "core").strip().lower()
REVIEWER_MODEL_FILTER = env_str_list("REVIEWER_MODELS")
REVIEWER_SEEDS = env_int_list("REVIEWER_SEEDS", [1234, 2026, 2027])

N_TRAIN = env_int("REVIEWER_N_TRAIN", 400)
N_VAL = env_int("REVIEWER_N_VAL", 30)
N_TEST = env_int("REVIEWER_N_TEST", 70)
SPLIT_SEED = env_int("SPLIT_SEED", 2026)

EPOCHS = env_int("REVIEWER_EPOCHS", 400)
BATCH_SIZE = env_int("REVIEWER_BATCH_SIZE", 32)
LR = env_float("REVIEWER_LR", 1.0e-3)
WEIGHT_DECAY = env_float("REVIEWER_WEIGHT_DECAY", 1.0e-6)
GRAD_CLIP = env_float("REVIEWER_GRAD_CLIP", 1.0)
FINAL_LOSS_WEIGHT = env_float("REVIEWER_FINAL_LOSS_WEIGHT", 1.0)
NUM_WORKERS = env_int("REVIEWER_NUM_WORKERS", 0)

WIDTH = env_int("REVIEWER_WIDTH", 32)
N_LAYERS = env_int("REVIEWER_LAYERS", 4)
MODES_X = env_int("REVIEWER_MODES_X", 16)
MODES_Y = env_int("REVIEWER_MODES_Y", 16)
LOCAL_RADIUS = env_int("REVIEWER_LOCAL_RADIUS", 2)
WINDOW_K = env_int("REVIEWER_WINDOW_K", 5)
N_TIME_MODES = env_int("REVIEWER_TIME_MODES", 4)
HEAD_WIDTH = env_int("REVIEWER_HEAD_WIDTH", 128)

# W=33 is a close capacity control for W=32, R=2, modes=16 AWFNO in 2D.
WFNO_PM_WIDTH = env_int("REVIEWER_WFNO_PM_WIDTH", 33)
PARAM_MATCH_TOLERANCE = env_float("REVIEWER_PARAM_MATCH_TOLERANCE", 0.05)

R_SWEEP = env_int_list("REVIEWER_R_SWEEP", [1, 2, 3])
MODES_SWEEP = env_int_list("REVIEWER_MODES_SWEEP", [8, 12, 16])
K_SWEEP = env_int_list("REVIEWER_K_SWEEP", [1, 5, 10])

DT_PHYSICAL = env_float("DT_PHYSICAL", 0.01)
X_LEFT = env_float("X_LEFT", 0.0)
X_RIGHT = env_float("X_RIGHT", 1.0)
Y_LEFT = env_float("Y_LEFT", 0.0)
Y_RIGHT = env_float("Y_RIGHT", 1.0)

ENFORCE_MEAN = env_bool("REVIEWER_ENFORCE_MEAN", True)
FREQUENCY_CUTOFF = env_float("REVIEWER_FREQUENCY_CUTOFF", 8.0)
INTERFACE_GRAD_QUANTILE = env_float(
    "REVIEWER_INTERFACE_GRAD_QUANTILE", 0.90
)

FORCE_RETRAIN = env_bool("FORCE_RETRAIN", False)
REQUIRE_PM_CACHE = env_bool("REVIEWER_REQUIRE_PM_CACHE", False)
PROFILE_COMPUTE = env_bool("REVIEWER_PROFILE_COMPUTE", True)
PROFILE_WARMUP = env_int("REVIEWER_PROFILE_WARMUP", 10)
PROFILE_REPEATS = env_int("REVIEWER_PROFILE_REPEATS", 50)
SAVE_ROLLOUTS = env_bool("SAVE_REVIEWER_ROLLOUTS", False)
SAVE_PLOTS = env_bool("SAVE_REVIEWER_PLOTS", True)
DETERMINISTIC = env_bool("REVIEWER_DETERMINISTIC", False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPORAL_BASIS_CACHE = {}


# Geometry is initialized after loading the data.
NX = NY = NT = None
DX = DY = None
X_GRID = Y_GRID = None


# ============================================================
# 1. GENERAL HELPERS
# ============================================================

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def short_hash(obj):
    text = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def torch_load_safe(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def clean_state_dict(state):
    state = copy.deepcopy(state)
    if isinstance(state, dict) and "_metadata" in state:
        del state["_metadata"]
    if hasattr(state, "_metadata"):
        try:
            del state._metadata
        except Exception:
            pass
    return OrderedDict((k, v) for k, v in state.items() if k != "_metadata")


def count_tensor_params(model):
    """Conventional PyTorch parameter count; one complex element counts as one."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_real_scalar_params(model):
    """Complex parameters count as two real scalar degrees of freedom."""
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel() * (2 if p.is_complex() else 1)
    return int(total)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_csv(rows, path):
    import csv
    rows = list(rows)
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def time_weights(nt):
    w = np.ones(int(nt), dtype=np.float64)
    if nt > 1:
        w[0] = w[-1] = 0.5
    return w


# ============================================================
# 2. DATA / SPLIT / WINDOWS
# ============================================================

def load_cahn_hilliard(path):
    global NX, NY, NT, DX, DY, X_GRID, Y_GRID
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Set it in Colab with %env DATA_FILE=/content/drive/.../file.pt"
        )
    data = torch_load_safe(path, "cpu")
    if "X_input" not in data or "Y_output" not in data:
        raise KeyError("Dataset must contain X_input and Y_output.")

    X = torch.as_tensor(data["X_input"]).float()
    Y = torch.as_tensor(data["Y_output"]).float()
    if X.ndim != 4 or X.shape[-1] != 1:
        raise ValueError(f"Expected X_input (N,Nx,Ny,1), got {tuple(X.shape)}")
    if Y.ndim != 5 or Y.shape[-1] != 1:
        raise ValueError(f"Expected Y_output (N,T,Nx,Ny,1), got {tuple(Y.shape)}")
    if X.shape[0] != Y.shape[0] or X.shape[1:3] != Y.shape[2:4]:
        raise ValueError("X_input and Y_output sample/spatial shapes do not match.")

    x0 = X[..., 0].numpy().astype(np.float32)
    yf = Y[..., 0].permute(0, 2, 3, 1).contiguous().numpy().astype(np.float32)
    U = np.concatenate([x0[..., None], yf], axis=-1).astype(np.float32)
    if not np.isfinite(U).all():
        raise ValueError("Dataset contains NaN or Inf.")

    NX, NY, NT = U.shape[1:]
    if "x_grid" in data:
        X_GRID = torch.as_tensor(
            data["x_grid"]
        ).float().reshape(-1).numpy()
    elif "xg" in data:
        X_GRID = torch.as_tensor(
            data["xg"]
        ).float().reshape(-1).numpy()
    else:
        X_GRID = np.linspace(
            X_LEFT, X_RIGHT, NX, endpoint=False, dtype=np.float32
        )
    if "y_grid" in data:
        Y_GRID = torch.as_tensor(
            data["y_grid"]
        ).float().reshape(-1).numpy()
    elif "yg" in data:
        Y_GRID = torch.as_tensor(
            data["yg"]
        ).float().reshape(-1).numpy()
    else:
        Y_GRID = np.linspace(
            Y_LEFT, Y_RIGHT, NY, endpoint=False, dtype=np.float32
        )
    if len(X_GRID) != NX or len(Y_GRID) != NY:
        raise ValueError("Grid lengths do not match the field dimensions.")
    DX = float(np.mean(np.diff(X_GRID))) if NX > 1 else 1.0
    DY = float(np.mean(np.diff(Y_GRID))) if NY > 1 else 1.0

    print("\nData")
    print("----")
    print("file       :", path)
    print("U shape    :", U.shape, "(N,Nx,Ny,Nt)")
    print("dx, dy, dt :", DX, DY, DT_PHYSICAL)
    return U


def split_data(U):
    need = N_TRAIN + N_VAL + N_TEST
    if need > len(U):
        raise ValueError(
            f"Need {need} trajectories (train={N_TRAIN}, val={N_VAL}, "
            f"test={N_TEST}), but the dataset has {len(U)}."
        )
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(U))[:need]
    train_idx = perm[:N_TRAIN]
    val_idx = perm[N_TRAIN:N_TRAIN + N_VAL]
    test_idx = perm[N_TRAIN + N_VAL:]
    return (
        U[train_idx],
        U[val_idx],
        U[test_idx],
        train_idx,
        val_idx,
        test_idx,
    )


def window_starts(nt, window_k):
    window_k = int(window_k)
    if window_k < 1 or window_k >= nt:
        raise ValueError(f"Need 1 <= K < Nt; got K={window_k}, Nt={nt}.")
    starts = list(range(0, nt - window_k, window_k))
    if not starts:
        raise ValueError(f"No complete windows for K={window_k}, Nt={nt}.")
    return starts


def build_windows(U, window_k):
    starts = window_starts(U.shape[-1], window_k)
    n, nx, ny, _ = U.shape
    u0 = np.empty((n * len(starts), nx, ny), dtype=np.float32)
    uf = np.empty(
        (n * len(starts), nx, ny, int(window_k)), dtype=np.float32
    )
    q = 0
    for i in range(n):
        for s in starts:
            u0[q] = U[i, :, :, s]
            uf[q] = U[i, :, :, s + 1:s + 1 + int(window_k)]
            q += 1
    return u0, uf


def compute_time_pod_basis_from_windows(
    u_start,
    u_future,
    n_modes,
):
    """
    Compute the temporal residual basis from TRAINING windows only.

    This is the temporal-basis construction used by the paper model:
    each future window is represented as the start field plus a linear
    combination of temporal POD vectors.
    """
    window_k = int(u_future.shape[-1])
    n_modes = int(n_modes)
    if not 1 <= n_modes <= window_k:
        raise ValueError(
            f"Need 1 <= temporal modes <= K; got M={n_modes}, K={window_k}."
        )
    residual = u_future - u_start[..., None]
    matrix = residual.reshape(-1, window_k).astype(np.float64)
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    basis = vt[:n_modes].T

    # The sign of each singular vector is mathematically arbitrary. Fix it
    # deterministically so recomputing the training-only basis after a Colab
    # restart produces the same coefficient convention.
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0

    energy = singular_values ** 2
    energy_ratio = float(
        energy[:n_modes].sum() / (energy.sum() + 1.0e-30)
    )
    basis = basis.astype(np.float32)
    singular_values = singular_values.astype(np.float32)
    print("\nTemporal POD basis (TRAIN windows only)")
    print("----------------------------------------")
    print("matrix shape       :", matrix.shape)
    print("window length K    :", window_k)
    print("temporal modes M   :", n_modes)
    print("energy captured    :", f"{energy_ratio:.10f}")
    print("singular values    :", singular_values.tolist())
    return basis, energy_ratio, singular_values


class WindowDataset(Dataset):
    def __init__(self, u0, uf):
        self.u0 = torch.from_numpy(np.asarray(u0, dtype=np.float32))
        self.uf = torch.from_numpy(np.asarray(uf, dtype=np.float32))

    def __len__(self):
        return len(self.u0)

    def __getitem__(self, index):
        return self.u0[index], self.uf[index]


class ScalarNormalizer:
    def __init__(self, values, eps=1.0e-6):
        values = torch.as_tensor(values, dtype=torch.float32)
        self.mean = values.mean()
        self.std = values.std()
        self.eps = float(eps)

    def encode(self, values):
        return (values - self.mean) / (self.std + self.eps)

    def decode(self, values):
        return values * (self.std + self.eps) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def make_loaders(U_train, U_val, window_k, seed):
    tr0, trf = build_windows(U_train, window_k)
    va0, vaf = build_windows(U_val, window_k)
    input_norm = ScalarNormalizer(tr0).to(DEVICE)
    future_norm = ScalarNormalizer(trf).to(DEVICE)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        WindowDataset(tr0, trf),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        WindowDataset(va0, vaf),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, input_norm, future_norm


def enforce_mean_future(pred, u0):
    target_mean = u0.mean(dim=(1, 2), keepdim=True).unsqueeze(-1)
    return pred + target_mean - pred.mean(dim=(1, 2), keepdim=True)


# ============================================================
# 3. MODELS
# ============================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        scale = 1.0 / np.sqrt(self.in_channels * self.out_channels)
        shape = (
            self.in_channels,
            self.out_channels,
            self.modes_x,
            self.modes_y,
        )
        self.weights_pos = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(*shape, dtype=torch.cfloat)
        )

    @staticmethod
    def multiply(a, b):
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        batch, _, nx, ny = x.shape
        if self.modes_x > nx // 2:
            raise ValueError(
                f"modes_x={self.modes_x} exceeds nx//2={nx // 2}; "
                "positive/negative bands would overlap."
            )
        if self.modes_y > ny // 2 + 1:
            raise ValueError(
                f"modes_y={self.modes_y} exceeds rFFT size {ny // 2 + 1}."
            )
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            nx,
            ny // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        mx, my = self.modes_x, self.modes_y
        out_ft[:, :, :mx, :my] = self.multiply(
            x_ft[:, :, :mx, :my], self.weights_pos
        )
        out_ft[:, :, -mx:, :my] = self.multiply(
            x_ft[:, :, -mx:, :my], self.weights_neg
        )
        return torch.fft.irfft2(out_ft, s=(nx, ny), norm="ortho")


class LocalOperator2d(nn.Module):
    def __init__(self, width, radius):
        super().__init__()
        radius = int(radius)
        self.op = nn.Conv2d(
            int(width),
            int(width),
            kernel_size=2 * radius + 1,
            padding=radius,
            padding_mode="circular",
            bias=True,
        )

    def forward(self, z):
        return self.op(z)


class FourierOperator2d(nn.Module):
    def __init__(self, width, modes_x, modes_y):
        super().__init__()
        self.op = SpectralConv2d(width, width, modes_x, modes_y)

    def forward(self, z):
        return self.op(z)


class OrderedResidualBlock2d(nn.Module):
    """
    L = learned finite-range circular convolution
    F = learned truncated Fourier convolution
    I = exact identity (no learned substage)

    Local stages use z <- z + GELU(L(z)); Fourier stages use the paper's
    z <- z + GELU(tanh(F(z))).  Thus LFL, LLF, and FLL have identical learned
    operators and parameter counts; only order changes.
    """
    VALID_PATTERNS = {"LFL", "LIL", "IFI", "LLF", "FLL"}

    def __init__(self, width, modes_x, modes_y, local_radius, pattern):
        super().__init__()
        pattern = str(pattern).upper()
        if pattern not in self.VALID_PATTERNS:
            raise ValueError(
                f"Unknown pattern {pattern!r}; choose "
                f"{sorted(self.VALID_PATTERNS)}."
            )
        self.pattern = pattern
        ops = []
        for symbol in pattern:
            if symbol == "L":
                ops.append(LocalOperator2d(width, local_radius))
            elif symbol == "F":
                ops.append(FourierOperator2d(width, modes_x, modes_y))
            else:
                ops.append(nn.Identity())
        self.ops = nn.ModuleList(ops)
        self.activation = nn.GELU()

    def forward(self, z):
        for symbol, op in zip(self.pattern, self.ops):
            if symbol == "L":
                z = z + self.activation(op(z))
            elif symbol == "F":
                z = z + self.activation(torch.tanh(op(z)))
        return z


def grid_features(x_grid, y_grid):
    x = torch.as_tensor(x_grid, dtype=torch.float32)
    y = torch.as_tensor(y_grid, dtype=torch.float32)
    xn = (x - x.min()) / (x.max() - x.min() + 1.0e-12)
    yn = (y - y.min()) / (y.max() - y.min() + 1.0e-12)
    xx, yy = torch.meshgrid(xn, yn, indexing="ij")
    return torch.stack(
        [
            xx,
            yy,
            torch.sin(2 * np.pi * xx),
            torch.cos(2 * np.pi * xx),
            torch.sin(2 * np.pi * yy),
            torch.cos(2 * np.pi * yy),
        ],
        dim=-1,
    ).unsqueeze(0)


class AlternatingModel2d(nn.Module):
    def __init__(
        self,
        window_k,
        x_grid,
        y_grid,
        pattern="LFL",
        width=32,
        n_layers=4,
        modes_x=16,
        modes_y=16,
        local_radius=2,
        head_width=128,
        output_channels=None,
    ):
        super().__init__()
        self.window_k = int(window_k)
        self.output_channels = int(
            self.window_k if output_channels is None else output_channels
        )
        self.pattern = str(pattern).upper()
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        self.local_radius = int(local_radius)
        self.register_buffer(
            "grid", grid_features(x_grid, y_grid), persistent=False
        )
        self.input_proj = nn.Linear(7, self.width)
        self.blocks = nn.ModuleList([
            OrderedResidualBlock2d(
                self.width,
                self.modes_x,
                self.modes_y,
                self.local_radius,
                self.pattern,
            )
            for _ in range(self.n_layers)
        ])
        self.fc1 = nn.Linear(self.width, int(head_width))
        self.fc2 = nn.Linear(int(head_width), self.output_channels)

    def forward(self, u0_norm):
        batch = u0_norm.shape[0]
        grid = self.grid.expand(batch, -1, -1, -1)
        z = torch.cat([u0_norm.unsqueeze(-1), grid], dim=-1)
        z = self.input_proj(z).permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            z = block(z)
        z = z.permute(0, 2, 3, 1).contiguous()
        return self.fc2(F.gelu(self.fc1(z))).contiguous()


class WFNO2d(nn.Module):
    def __init__(
        self,
        window_k,
        x_grid,
        y_grid,
        width=32,
        n_layers=4,
        modes_x=16,
        modes_y=16,
        head_width=128,
    ):
        super().__init__()
        self.window_k = int(window_k)
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.modes_x = int(modes_x)
        self.modes_y = int(modes_y)
        self.register_buffer(
            "grid", grid_features(x_grid, y_grid), persistent=False
        )
        self.fc0 = nn.Linear(7, self.width)
        self.convs = nn.ModuleList([
            SpectralConv2d(
                self.width, self.width, self.modes_x, self.modes_y
            )
            for _ in range(self.n_layers)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.n_layers)
        ])
        self.fc1 = nn.Linear(self.width, int(head_width))
        self.fc2 = nn.Linear(int(head_width), self.window_k)

    def forward(self, u0_norm):
        batch = u0_norm.shape[0]
        grid = self.grid.expand(batch, -1, -1, -1)
        z = torch.cat([u0_norm.unsqueeze(-1), grid], dim=-1)
        z = self.fc0(z).permute(0, 3, 1, 2).contiguous()
        for i, (conv, pointwise) in enumerate(zip(self.convs, self.ws)):
            z = conv(z) + pointwise(z)
            if i != self.n_layers - 1:
                z = F.gelu(z)
        z = z.permute(0, 2, 3, 1).contiguous()
        return self.fc2(F.gelu(self.fc1(z))).contiguous()


# ============================================================
# 4. MODEL SPECIFICATIONS
# ============================================================

def alt_spec(
    name,
    pattern="LFL",
    window_k=WINDOW_K,
    width=WIDTH,
    n_layers=N_LAYERS,
    modes_x=MODES_X,
    modes_y=MODES_Y,
    local_radius=LOCAL_RADIUS,
):
    return {
        "name": str(name),
        "family": "alternating",
        "pattern": str(pattern).upper(),
        "window_k": int(window_k),
        "width": int(width),
        "n_layers": int(n_layers),
        "modes_x": int(modes_x),
        "modes_y": int(modes_y),
        "local_radius": int(local_radius),
    }


def wfno_spec(
    name,
    window_k=WINDOW_K,
    width=WIDTH,
    n_layers=N_LAYERS,
    modes_x=MODES_X,
    modes_y=MODES_Y,
):
    return {
        "name": str(name),
        "family": "wfno",
        "pattern": "standard_fno",
        "window_k": int(window_k),
        "width": int(width),
        "n_layers": int(n_layers),
        "modes_x": int(modes_x),
        "modes_y": int(modes_y),
        "local_radius": 0,
    }


def tb_alt_spec(
    name,
    n_time_modes=N_TIME_MODES,
    pattern="LFL",
    window_k=WINDOW_K,
    width=WIDTH,
    n_layers=N_LAYERS,
    modes_x=MODES_X,
    modes_y=MODES_Y,
    local_radius=LOCAL_RADIUS,
):
    spec = alt_spec(
        name,
        pattern=pattern,
        window_k=window_k,
        width=width,
        n_layers=n_layers,
        modes_x=modes_x,
        modes_y=modes_y,
        local_radius=local_radius,
    )
    spec["family"] = "tb_alternating"
    spec["n_time_modes"] = int(n_time_modes)
    return spec


def spec_key(spec):
    return tuple(spec[k] for k in (
        "family",
        "pattern",
        "window_k",
        "width",
        "n_layers",
        "modes_x",
        "modes_y",
        "local_radius",
    )) + (int(spec.get("n_time_modes", 0)),)


def make_specs():
    if REVIEWER_SUITE not in {
        "core", "ordering", "sensitivity", "tb", "all"
    }:
        raise ValueError(
            "REVIEWER_SUITE must be core, ordering, sensitivity, tb, or all."
        )
    base = alt_spec("AWFNO_LFL")
    specs = []
    if REVIEWER_SUITE in {"core", "all"}:
        specs.extend([
            base,
            alt_spec("LOCAL_ONLY_LIL", pattern="LIL"),
            alt_spec("FOURIER_ONLY_IFI", pattern="IFI"),
            wfno_spec(f"WFNO_W{WIDTH}"),
            wfno_spec(f"WFNO_PM_W{WFNO_PM_WIDTH}", width=WFNO_PM_WIDTH),
        ])
    if REVIEWER_SUITE in {"ordering", "all"}:
        specs.extend([
            base,
            alt_spec("AWFNO_LLF", pattern="LLF"),
            alt_spec("AWFNO_FLL", pattern="FLL"),
        ])
    if REVIEWER_SUITE in {"sensitivity", "all"}:
        specs.append(base)
        for radius in R_SWEEP:
            if int(radius) != LOCAL_RADIUS:
                specs.append(alt_spec(
                    f"AWFNO_R{radius}", local_radius=int(radius)
                ))
        for modes in MODES_SWEEP:
            if int(modes) != MODES_X or int(modes) != MODES_Y:
                specs.append(alt_spec(
                    f"AWFNO_MODES{modes}",
                    modes_x=int(modes),
                    modes_y=int(modes),
                ))
        for window_k in K_SWEEP:
            if int(window_k) != WINDOW_K:
                specs.append(alt_spec(
                    f"AWFNO_K{window_k}", window_k=int(window_k)
                ))
    if REVIEWER_SUITE in {"tb", "all"}:
        specs.extend([
            tb_alt_spec("TB_AWFNO_LFL"),
            wfno_spec(
                f"WFNO_PM_W{WFNO_PM_WIDTH}",
                width=WFNO_PM_WIDTH,
            ),
        ])

    unique, seen = [], set()
    for spec in specs:
        key = spec_key(spec)
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    if REVIEWER_MODEL_FILTER:
        requested = set(REVIEWER_MODEL_FILTER)
        available = {s["name"] for s in unique}
        missing = requested - available
        if missing:
            raise ValueError(
                f"Unknown REVIEWER_MODELS {sorted(missing)}. "
                f"Available: {sorted(available)}"
            )
        unique = [s for s in unique if s["name"] in requested]
    return unique


def validate_spec(spec):
    if spec["width"] < 1 or spec["n_layers"] < 1:
        raise ValueError(f"{spec['name']}: width/layers must be positive.")
    if spec["modes_x"] < 1 or spec["modes_y"] < 1:
        raise ValueError(f"{spec['name']}: retained modes must be positive.")
    if (
        spec["family"] in {"alternating", "tb_alternating"}
        and spec["local_radius"] < 1
    ):
        raise ValueError(f"{spec['name']}: local radius must be >= 1.")
    if spec["family"].startswith("tb_"):
        if not 1 <= spec["n_time_modes"] <= spec["window_k"]:
            raise ValueError(
                f"{spec['name']}: need 1 <= temporal modes <= K; got "
                f"M={spec['n_time_modes']}, K={spec['window_k']}."
            )
    elif int(spec.get("n_time_modes", 0)) != 0:
        raise ValueError(
            f"{spec['name']}: direct models must have n_time_modes=0."
        )
    if spec["modes_x"] > NX // 2:
        raise ValueError(
            f"{spec['name']}: modes_x={spec['modes_x']} > Nx//2={NX // 2}."
        )
    if spec["modes_y"] > NY // 2 + 1:
        raise ValueError(
            f"{spec['name']}: modes_y={spec['modes_y']} > Ny//2+1="
            f"{NY // 2 + 1}."
        )
    window_starts(NT, spec["window_k"])


def build_model(spec):
    validate_spec(spec)
    if spec["family"] in {"alternating", "tb_alternating"}:
        return AlternatingModel2d(
            spec["window_k"],
            X_GRID,
            Y_GRID,
            pattern=spec["pattern"],
            width=spec["width"],
            n_layers=spec["n_layers"],
            modes_x=spec["modes_x"],
            modes_y=spec["modes_y"],
            local_radius=spec["local_radius"],
            head_width=HEAD_WIDTH,
            output_channels=(
                spec["n_time_modes"]
                if spec["family"] == "tb_alternating" else None
            ),
        )
    if spec["family"] == "wfno":
        return WFNO2d(
            spec["window_k"],
            X_GRID,
            Y_GRID,
            width=spec["width"],
            n_layers=spec["n_layers"],
            modes_x=spec["modes_x"],
            modes_y=spec["modes_y"],
            head_width=HEAD_WIDTH,
        )
    raise ValueError(spec["family"])


# ============================================================
# 5. CONTROLLED TRAINING / ROLLOUT
# ============================================================

def make_predict(model, input_norm, future_norm):
    def predict(u0):
        pred = future_norm.decode(model(input_norm.encode(u0)))
        if ENFORCE_MEAN:
            pred = enforce_mean_future(pred, u0)
        return pred
    return predict


def make_tb_predict(model, input_norm, basis):
    basis = torch.as_tensor(
        basis, dtype=torch.float32, device=DEVICE
    )

    def predict(u0):
        coefficients = model(input_norm.encode(u0))
        residual = torch.einsum(
            "bxym,km->bxyk", coefficients, basis
        )
        pred = u0.unsqueeze(-1) + residual
        if ENFORCE_MEAN:
            pred = enforce_mean_future(pred, u0)
        return pred

    return predict


def normalized_mse(pred, true, normalizer):
    return F.mse_loss(normalizer.encode(pred), normalizer.encode(true))


def window_loss(pred, true, normalizer):
    all_steps = normalized_mse(pred, true, normalizer)
    final_step = normalized_mse(
        pred[..., -1], true[..., -1], normalizer
    )
    return all_steps + FINAL_LOSS_WEIGHT * final_step


def spatial_l2_torch(pred, true):
    return torch.sqrt(
        torch.sum((pred - true) ** 2, dim=(1, 2)) * DX * DY + 1.0e-30
    )


def run_epoch(
    model,
    loader,
    predict,
    future_norm,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = total_l2 = total_final = 0.0
    n_seen = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for u0, true in loader:
            u0 = u0.to(DEVICE, non_blocking=True)
            true = true.to(DEVICE, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = predict(u0)
            loss = window_loss(pred, true, future_norm)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), GRAD_CLIP
                )
                optimizer.step()

            batch = u0.shape[0]
            l2_steps = spatial_l2_torch(pred.detach(), true)
            total_loss += float(loss.detach()) * batch
            total_l2 += float(l2_steps.mean()) * batch
            total_final += float(l2_steps[..., -1].mean()) * batch
            n_seen += batch
    return {
        "loss": total_loss / max(n_seen, 1),
        "window_l2": total_l2 / max(n_seen, 1),
        "window_final_l2": total_final / max(n_seen, 1),
    }


def fit_model(
    model,
    train_loader,
    val_loader,
    predict,
    future_norm,
    model_name,
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    best_value = float("inf")
    best_state = None
    history = {
        "train_loss": [],
        "train_window_l2": [],
        "val_loss": [],
        "val_window_l2": [],
        "val_final_l2": [],
    }
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_stats = run_epoch(
            model, train_loader, predict, future_norm, optimizer
        )
        val_stats = run_epoch(
            model, val_loader, predict, future_norm, optimizer=None
        )
        scheduler.step()
        history["train_loss"].append(train_stats["loss"])
        history["train_window_l2"].append(train_stats["window_l2"])
        history["val_loss"].append(val_stats["loss"])
        history["val_window_l2"].append(val_stats["window_l2"])
        history["val_final_l2"].append(val_stats["window_final_l2"])
        if val_stats["window_l2"] < best_value:
            best_value = val_stats["window_l2"]
            best_state = clean_state_dict(model.state_dict())

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"[{model_name}] {epoch:04d}/{EPOCHS} | "
                f"train {train_stats['loss']:.5e} | "
                f"val L2 {val_stats['window_l2']:.5e} | "
                f"val final {val_stats['window_final_l2']:.5e}"
            )
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return {
        "best_val_window_l2": float(best_value),
        "train_time_sec": float(time.perf_counter() - started),
        "history": history,
    }


@torch.no_grad()
def recursive_rollout(model, predict, U, window_k, batch_size=None):
    model.eval()
    U = np.asarray(U, dtype=np.float32)
    n, nx, ny, nt = U.shape
    batch_size = int(batch_size or BATCH_SIZE)
    output = np.empty_like(U)
    output[..., 0] = U[..., 0]
    for i0 in range(0, n, batch_size):
        i1 = min(i0 + batch_size, n)
        current = torch.from_numpy(U[i0:i1, :, :, 0]).to(DEVICE)
        cursor = 1
        while cursor < nt:
            block = predict(current)
            take = min(int(window_k), nt - cursor)
            output[i0:i1, :, :, cursor:cursor + take] = (
                block[..., :take].detach().cpu().numpy()
            )
            current = block[..., take - 1]
            cursor += take
    return output


def checkpoint_signature(spec, seed):
    data_stat = DATA_FILE.stat()
    signature = {
        "spec": spec,
        "seed": int(seed),
        "data_name": DATA_FILE.name,
        "data_size": int(data_stat.st_size),
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "split_seed": SPLIT_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "final_loss_weight": FINAL_LOSS_WEIGHT,
        "enforce_mean": ENFORCE_MEAN,
        "head_width": HEAD_WIDTH,
    }
    if spec["family"].startswith("tb_"):
        signature["temporal_basis_version"] = "train_residual_pod_v1"
    return signature


def checkpoint_path(run_dir, spec, seed):
    signature = checkpoint_signature(spec, seed)
    return run_dir / (
        f"{spec['name']}_seed{seed}_{short_hash(signature)}.pt"
    )


def load_or_train(
    spec,
    seed,
    U_train,
    U_val,
    run_dir,
):
    set_seed(seed)
    train_loader, val_loader, input_norm, future_norm = make_loaders(
        U_train, U_val, spec["window_k"], seed
    )
    model = build_model(spec).to(DEVICE)
    basis_np = None
    basis_energy = float("nan")
    basis_singular_values = []
    if spec["family"].startswith("tb_"):
        basis_key = (
            int(spec["window_k"]),
            int(spec["n_time_modes"]),
        )
        if basis_key not in TEMPORAL_BASIS_CACHE:
            train_u0, train_future = build_windows(
                U_train, spec["window_k"]
            )
            TEMPORAL_BASIS_CACHE[basis_key] = (
                compute_time_pod_basis_from_windows(
                    train_u0,
                    train_future,
                    spec["n_time_modes"],
                )
            )
            del train_u0, train_future
        (
            basis_np,
            basis_energy,
            basis_singular_values,
        ) = TEMPORAL_BASIS_CACHE[basis_key]
        predict = make_tb_predict(model, input_norm, basis_np)
    else:
        predict = make_predict(model, input_norm, future_norm)
    path = checkpoint_path(run_dir, spec, seed)
    cached = False
    if (
        REQUIRE_PM_CACHE
        and REVIEWER_SUITE == "tb"
        and spec["name"] == f"WFNO_PM_W{WFNO_PM_WIDTH}"
        and not path.exists()
    ):
        raise FileNotFoundError(
            "The existing parameter-matched WFNO checkpoint was not "
            f"found at {path}. Check EXPERIMENT_NAME, DATA_FILE, the "
            "training configuration, and the output directory. Set "
            "REVIEWER_REQUIRE_PM_CACHE=0 only if you intentionally want "
            "to train WFNO again."
        )

    if path.exists() and not FORCE_RETRAIN:
        payload = torch_load_safe(path, DEVICE)
        model.load_state_dict(
            clean_state_dict(payload["state_dict"]), strict=True
        )
        fit_info = dict(payload["fit_info"])
        saved_basis = payload.get("temporal_basis")
        if basis_np is not None and saved_basis is not None:
            saved_basis = np.asarray(saved_basis, dtype=np.float32)
            if not np.allclose(saved_basis, basis_np, atol=1.0e-6):
                raise RuntimeError(
                    f"{spec['name']}: cached temporal basis does not "
                    "match the basis recomputed from training windows."
                )
        cached = True
        print(f"Loaded checkpoint: {path.name}")
    else:
        fit_info = fit_model(
            model,
            train_loader,
            val_loader,
            predict,
            future_norm,
            spec["name"],
        )
        torch.save(
            {
                "state_dict": clean_state_dict(model.state_dict()),
                "fit_info": fit_info,
                "signature": checkpoint_signature(spec, seed),
                "temporal_basis": basis_np,
            },
            path,
        )
        print(f"Saved checkpoint: {path.name}")
    if basis_np is not None:
        fit_info["temporal_basis_energy"] = float(basis_energy)
        fit_info["temporal_basis_singular_values"] = (
            np.asarray(basis_singular_values).tolist()
        )
    return {
        "model": model,
        "predict": predict,
        "input_norm": input_norm,
        "future_norm": future_norm,
        "fit_info": fit_info,
        "cached": cached,
        "checkpoint": str(path),
        "temporal_basis": basis_np,
    }


# ============================================================
# 6. REVIEWER DIAGNOSTICS
# ============================================================

def spatial_l2_per_time_np(pred, true):
    err = (
        np.asarray(pred, dtype=np.float64)
        - np.asarray(true, dtype=np.float64)
    )
    return np.sqrt(
        np.sum(err * err, axis=(1, 2)) * DX * DY + 1.0e-30
    )


def physical_l2_metrics(pred, true):
    l2_t = spatial_l2_per_time_np(pred, true)
    weights = time_weights(l2_t.shape[-1])
    full = np.sqrt(
        np.sum(l2_t * l2_t * weights[None, :], axis=-1)
        * DT_PHYSICAL
        + 1.0e-30
    )
    final = l2_t[:, -1]
    return {
        "full_l2_mean": float(full.mean()),
        "full_l2_std_test": float(full.std()),
        "full_l2_median": float(np.median(full)),
        "final_l2_mean": float(final.mean()),
        "final_l2_std_test": float(final.std()),
        "final_l2_median": float(np.median(final)),
        "l2_t_mean": l2_t.mean(axis=0),
        "all_full_l2": full,
        "all_final_l2": final,
    }


def frequency_band_metrics(pred, true, cutoff):
    """
    Absolute trajectory L2 after an orthogonal 2D radial Fourier split.

    The fixed cutoff is used for every model, including the modes sweep.
    This makes the diagnostic comparable across architectures.
    """
    # Keep t0 (whose rollout error is exactly zero) so the temporal
    # quadrature is identical to the reported full-trajectory physical L2.
    err = (
        np.asarray(pred, dtype=np.float64)
        - np.asarray(true, dtype=np.float64)
    )
    err_ft = np.fft.fft2(err, axes=(1, 2), norm="ortho")
    kx = np.fft.fftfreq(err.shape[1]) * err.shape[1]
    ky = np.fft.fftfreq(err.shape[2]) * err.shape[2]
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    low_mask = radius < float(cutoff)
    high_mask = ~low_mask
    energy = np.abs(err_ft) ** 2
    weights = time_weights(err.shape[-1])

    def band(mask):
        per_time = np.sum(energy[:, mask, :], axis=1)
        per_sample = np.sqrt(
            np.sum(per_time * weights[None, :], axis=-1)
            * DX * DY * DT_PHYSICAL
            + 1.0e-30
        )
        return per_sample

    low = band(low_mask)
    high = band(high_mask)
    total_energy = energy.sum(axis=(1, 2, 3)) + 1.0e-30
    high_fraction = (
        energy[:, high_mask, :].sum(axis=(1, 2)) / total_energy
    )
    return {
        "low_frequency_l2_mean": float(low.mean()),
        "low_frequency_l2_std_test": float(low.std()),
        "high_frequency_l2_mean": float(high.mean()),
        "high_frequency_l2_std_test": float(high.std()),
        "high_frequency_error_fraction_mean": float(
            high_fraction.mean()
        ),
        "frequency_cutoff": float(cutoff),
    }


def interface_metrics(pred, true, quantile):
    """
    Localized errors on interfaces defined from ground-truth high gradients.

    For each test trajectory and future time, the interface mask contains
    exactly the top (1-quantile) fraction of ground-truth |grad u| pixels.
    It is independent of the prediction and adapts to coarsening interfaces
    without choosing an arbitrary phase-value threshold.
    """
    pred = np.asarray(pred[..., 1:], dtype=np.float64)
    true = np.asarray(true[..., 1:], dtype=np.float64)
    gx = (
        np.roll(true, -1, axis=1) - np.roll(true, 1, axis=1)
    ) / (2.0 * DX)
    gy = (
        np.roll(true, -1, axis=2) - np.roll(true, 1, axis=2)
    ) / (2.0 * DY)
    grad = np.sqrt(gx * gx + gy * gy)
    n, nx, ny, nt = grad.shape
    flat_grad = grad.reshape(n, nx * ny, nt)
    n_interface = max(
        1, int(np.ceil((1.0 - float(quantile)) * nx * ny))
    )
    top_indices = np.argpartition(
        flat_grad,
        kth=nx * ny - n_interface,
        axis=1,
    )[:, -n_interface:, :]
    flat_interface = np.zeros_like(flat_grad, dtype=bool)
    np.put_along_axis(
        flat_interface, top_indices, True, axis=1
    )
    interface = flat_interface.reshape(n, nx, ny, nt)
    bulk = ~interface
    sq = (pred - true) ** 2

    def masked_rmse(mask):
        numerator = np.sum(sq * mask, axis=(1, 2, 3))
        denominator = np.sum(mask, axis=(1, 2, 3))
        return np.sqrt(numerator / np.maximum(denominator, 1))

    interface_rmse = masked_rmse(interface)
    bulk_rmse = masked_rmse(bulk)
    interface_energy_fraction = (
        np.sum(sq * interface, axis=(1, 2, 3))
        / (np.sum(sq, axis=(1, 2, 3)) + 1.0e-30)
    )
    return {
        "interface_rmse_mean": float(interface_rmse.mean()),
        "interface_rmse_std_test": float(interface_rmse.std()),
        "bulk_rmse_mean": float(bulk_rmse.mean()),
        "bulk_rmse_std_test": float(bulk_rmse.std()),
        "interface_error_energy_fraction_mean": float(
            interface_energy_fraction.mean()
        ),
        "interface_area_fraction": float(interface.mean()),
        "interface_grad_quantile": float(quantile),
    }


def mass_metrics(pred, true):
    pred_mass = np.asarray(pred, dtype=np.float64).mean(axis=(1, 2))
    true_mass = np.asarray(true, dtype=np.float64).mean(axis=(1, 2))
    initial = true_mass[:, :1]
    pred_drift = np.max(np.abs(pred_mass - initial), axis=1)
    mass_error = np.sqrt(np.mean((pred_mass - true_mass) ** 2, axis=1))
    return {
        "max_mass_drift_mean": float(pred_drift.mean()),
        "mass_rmse_mean": float(mass_error.mean()),
    }


@torch.no_grad()
def profile_inference(model, predict, u0_np, nt, window_k):
    if not PROFILE_COMPUTE:
        return {}
    model.eval()
    u0 = torch.from_numpy(
        np.asarray(u0_np[:1], dtype=np.float32)
    ).to(DEVICE)

    for _ in range(PROFILE_WARMUP):
        _ = predict(u0)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(PROFILE_REPEATS):
        _ = predict(u0)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    forward_ms = (
        time.perf_counter() - start
    ) * 1000.0 / max(PROFILE_REPEATS, 1)

    rollout_repeats = max(1, min(10, PROFILE_REPEATS))
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(rollout_repeats):
        current = u0
        cursor = 1
        while cursor < nt:
            block = predict(current)
            take = min(int(window_k), nt - cursor)
            current = block[..., take - 1]
            cursor += take
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_mb = float("nan")
    rollout_ms = (
        time.perf_counter() - start
    ) * 1000.0 / rollout_repeats
    return {
        "forward_batch1_ms": float(forward_ms),
        "rollout_batch1_ms": float(rollout_ms),
        "peak_allocated_mb": float(peak_mb),
        "rollout_calls": int(np.ceil((nt - 1) / int(window_k))),
    }


def evaluate_model(bundle, spec, U_test):
    pred = recursive_rollout(
        bundle["model"],
        bundle["predict"],
        U_test,
        spec["window_k"],
    )
    result = {}
    result.update(physical_l2_metrics(pred, U_test))
    result.update(
        frequency_band_metrics(pred, U_test, FREQUENCY_CUTOFF)
    )
    result.update(
        interface_metrics(pred, U_test, INTERFACE_GRAD_QUANTILE)
    )
    result.update(mass_metrics(pred, U_test))
    result.update(profile_inference(
        bundle["model"],
        bundle["predict"],
        U_test[..., 0],
        U_test.shape[-1],
        spec["window_k"],
    ))
    return result, pred


# ============================================================
# 7. TABLES / PLOTS / PREFLIGHT CHECKS
# ============================================================

SCALAR_METRICS = [
    "best_val_window_l2",
    "temporal_basis_energy",
    "train_time_sec",
    "full_l2_mean",
    "final_l2_mean",
    "low_frequency_l2_mean",
    "high_frequency_l2_mean",
    "high_frequency_error_fraction_mean",
    "interface_rmse_mean",
    "bulk_rmse_mean",
    "interface_error_energy_fraction_mean",
    "max_mass_drift_mean",
    "mass_rmse_mean",
    "forward_batch1_ms",
    "rollout_batch1_ms",
    "peak_allocated_mb",
]


def make_parameter_rows(specs):
    rows = []
    for spec in specs:
        model = build_model(spec)
        rows.append({
            "model": spec["name"],
            "family": spec["family"],
            "pattern": spec["pattern"],
            "window_k": spec["window_k"],
            "width": spec["width"],
            "n_layers": spec["n_layers"],
            "modes_x": spec["modes_x"],
            "modes_y": spec["modes_y"],
            "local_radius": spec["local_radius"],
            "n_time_modes": int(spec.get("n_time_modes", 0)),
            "tensor_parameter_count": count_tensor_params(model),
            "real_scalar_parameter_count": count_real_scalar_params(model),
        })
        del model
    return rows


def preflight_checks(specs, parameter_rows):
    print("\nPreflight")
    print("---------")
    by_name = {row["model"]: row for row in parameter_rows}
    for row in parameter_rows:
        print(
            f"{row['model']:20s} | tensor params "
            f"{row['tensor_parameter_count']:,} | real scalars "
            f"{row['real_scalar_parameter_count']:,}"
        )

    ordering_names = [
        name for name in ("AWFNO_LFL", "AWFNO_LLF", "AWFNO_FLL")
        if name in by_name
    ]
    if len(ordering_names) > 1:
        counts = {
            by_name[name]["real_scalar_parameter_count"]
            for name in ordering_names
        }
        if len(counts) != 1:
            raise RuntimeError(
                "Ordering ablations do not have identical parameter counts."
            )
        print("Ordering parameter equality: PASS")

    pm_name = f"WFNO_PM_W{WFNO_PM_WIDTH}"
    if "AWFNO_LFL" in by_name and pm_name in by_name:
        aw = by_name["AWFNO_LFL"]["real_scalar_parameter_count"]
        pm = by_name[pm_name]["real_scalar_parameter_count"]
        relative_gap = abs(pm - aw) / aw
        print(
            f"Parameter-match real-scalar gap: {100 * relative_gap:.2f}%"
        )
        if relative_gap > PARAM_MATCH_TOLERANCE:
            raise RuntimeError(
                f"Parameter-matched WFNO gap {relative_gap:.3%} exceeds "
                f"tolerance {PARAM_MATCH_TOLERANCE:.3%}. Adjust "
                "REVIEWER_WFNO_PM_WIDTH."
            )

    if "TB_AWFNO_LFL" in by_name and pm_name in by_name:
        tb_aw = by_name["TB_AWFNO_LFL"][
            "real_scalar_parameter_count"
        ]
        tb_pm = by_name[pm_name]["real_scalar_parameter_count"]
        relative_gap = abs(tb_pm - tb_aw) / tb_aw
        print(
            "TB parameter-match real-scalar gap: "
            f"{100 * relative_gap:.2f}%"
        )
        if relative_gap > PARAM_MATCH_TOLERANCE:
            raise RuntimeError(
                f"Parameter-matched WFNO gap {relative_gap:.3%} "
                f"exceeds tolerance {PARAM_MATCH_TOLERANCE:.3%}. "
                "Adjust REVIEWER_WFNO_PM_WIDTH."
            )


def raw_result_row(spec, seed, bundle, metrics, parameter_row):
    row = {
        "model": spec["name"],
        "seed": int(seed),
        "family": spec["family"],
        "pattern": spec["pattern"],
        "window_k": spec["window_k"],
        "width": spec["width"],
        "n_layers": spec["n_layers"],
        "modes_x": spec["modes_x"],
        "modes_y": spec["modes_y"],
        "local_radius": spec["local_radius"],
        "n_time_modes": int(spec.get("n_time_modes", 0)),
        "tensor_parameter_count": parameter_row[
            "tensor_parameter_count"
        ],
        "real_scalar_parameter_count": parameter_row[
            "real_scalar_parameter_count"
        ],
        "cached": int(bundle["cached"]),
        "checkpoint": bundle["checkpoint"],
        "best_val_window_l2": bundle["fit_info"][
            "best_val_window_l2"
        ],
        "train_time_sec": bundle["fit_info"]["train_time_sec"],
        "rollout_calls": int(
            np.ceil((NT - 1) / spec["window_k"])
        ),
    }
    if "temporal_basis_energy" in bundle["fit_info"]:
        row["temporal_basis_energy"] = bundle["fit_info"][
            "temporal_basis_energy"
        ]
    for key, value in metrics.items():
        if np.isscalar(value):
            row[key] = value
    return row


def aggregate_rows(raw_rows):
    aggregated = []
    names = []
    for row in raw_rows:
        if row["model"] not in names:
            names.append(row["model"])
    for name in names:
        group = [row for row in raw_rows if row["model"] == name]
        first = group[0]
        out = {
            "model": name,
            "n_seeds": len(group),
            "family": first["family"],
            "pattern": first["pattern"],
            "window_k": first["window_k"],
            "width": first["width"],
            "n_layers": first["n_layers"],
            "modes_x": first["modes_x"],
            "modes_y": first["modes_y"],
            "local_radius": first["local_radius"],
            "n_time_modes": first["n_time_modes"],
            "tensor_parameter_count": first["tensor_parameter_count"],
            "real_scalar_parameter_count": first[
                "real_scalar_parameter_count"
            ],
            "rollout_calls": first["rollout_calls"],
        }
        for metric in SCALAR_METRICS:
            values = np.asarray(
                [
                    row[metric] for row in group
                    if metric in row and np.isfinite(row[metric])
                ],
                dtype=np.float64,
            )
            if len(values):
                out[metric] = float(values.mean())
                out[f"{metric}_std_across_seeds"] = float(
                    values.std(ddof=1) if len(values) > 1 else 0.0
                )
        aggregated.append(out)
    return aggregated


def print_aggregated(rows):
    print("\nAggregated test results (mean +/- seed SD)")
    print("-----------------------------------------")
    for row in rows:
        def cell(metric):
            mean = row.get(metric, float("nan"))
            std = row.get(
                f"{metric}_std_across_seeds", float("nan")
            )
            return f"{mean:.4e} +/- {std:.2e}"

        print(
            f"{row['model']:20s} | full {cell('full_l2_mean')} | "
            f"final {cell('final_l2_mean')} | "
            f"high-f {cell('high_frequency_l2_mean')} | "
            f"interface {cell('interface_rmse_mean')}"
        )


def save_comparison_plot(rows, path):
    if not SAVE_PLOTS or plt is None or not rows:
        return
    labels = [r["model"] for r in rows]
    metrics = [
        ("full_l2_mean", "Full trajectory L2"),
        ("final_l2_mean", "Final-time L2"),
        ("high_frequency_l2_mean", "High-frequency L2"),
        ("interface_rmse_mean", "Interface RMSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (metric, title) in zip(axes.flat, metrics):
        values = [r.get(metric, np.nan) for r in rows]
        errors = [
            r.get(f"{metric}_std_across_seeds", 0.0) for r in rows
        ]
        ax.bar(np.arange(len(labels)), values, yerr=errors, capsize=3)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"Cahn--Hilliard controlled reviewer ablations: {REVIEWER_SUITE}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("Cahn--Hilliard 2D controlled reviewer ablations")
    print("================================================")
    print("device              :", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU                 :", torch.cuda.get_device_name(0))
    print("suite               :", REVIEWER_SUITE)
    print("seeds               :", REVIEWER_SEEDS)
    print("epochs              :", EPOCHS)
    print("identical optimizer : AdamW + cosine + clipping")
    print("mean enforcement    :", ENFORCE_MEAN, "(all models)")
    if REQUIRE_PM_CACHE and FORCE_RETRAIN:
        raise ValueError(
            "REVIEWER_REQUIRE_PM_CACHE=1 conflicts with FORCE_RETRAIN=1."
        )
    if REVIEWER_SUITE in {"tb", "all"}:
        print(
            "TB paper config     : "
            f"W={WIDTH}, L={N_LAYERS}, R={LOCAL_RADIUS}, "
            f"modes={MODES_X}x{MODES_Y}, K={WINDOW_K}, "
            f"temporal M={N_TIME_MODES}"
        )
        print("TB basis data       : training windows only")
    if not 0.0 < INTERFACE_GRAD_QUANTILE < 1.0:
        raise ValueError(
            "REVIEWER_INTERFACE_GRAD_QUANTILE must lie in (0,1)."
        )
    if FREQUENCY_CUTOFF <= 0.0:
        raise ValueError("REVIEWER_FREQUENCY_CUTOFF must be positive.")

    U = load_cahn_hilliard(DATA_FILE)
    (
        U_train,
        U_val,
        U_test,
        train_idx,
        val_idx,
        test_idx,
    ) = split_data(U)
    specs = make_specs()
    for spec in specs:
        validate_spec(spec)

    run_name = (
        f"{EXPERIMENT_NAME}_N{N_TRAIN}_"
        f"{NX}x{NY}x{NT}_split{SPLIT_SEED}"
    )
    run_dir = OUTPUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print("output directory     :", run_dir)

    np.savez(
        run_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    config = {
        "data_file": str(DATA_FILE),
        "output_directory": str(run_dir),
        "experiment_name": EXPERIMENT_NAME,
        "suite": REVIEWER_SUITE,
        "model_filter": REVIEWER_MODEL_FILTER,
        "seeds": REVIEWER_SEEDS,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "split_seed": SPLIT_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "final_loss_weight": FINAL_LOSS_WEIGHT,
        "width": WIDTH,
        "n_layers": N_LAYERS,
        "modes_x": MODES_X,
        "modes_y": MODES_Y,
        "local_radius": LOCAL_RADIUS,
        "window_k": WINDOW_K,
        "n_time_modes": N_TIME_MODES,
        "wfno_pm_width": WFNO_PM_WIDTH,
        "r_sweep": R_SWEEP,
        "modes_sweep": MODES_SWEEP,
        "k_sweep": K_SWEEP,
        "frequency_cutoff": FREQUENCY_CUTOFF,
        "interface_grad_quantile": INTERFACE_GRAD_QUANTILE,
        "enforce_mean": ENFORCE_MEAN,
        "require_pm_cache": REQUIRE_PM_CACHE,
        "dt_physical": DT_PHYSICAL,
        "nx": NX,
        "ny": NY,
        "nt": NT,
        "dx": DX,
        "dy": DY,
        "device": str(DEVICE),
        "gpu": (
            torch.cuda.get_device_name(0)
            if DEVICE.type == "cuda" else None
        ),
        "specs": specs,
    }
    with open(
        run_dir / f"config_{REVIEWER_SUITE}.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(jsonable(config), handle, indent=2)

    parameter_rows = make_parameter_rows(specs)
    preflight_checks(specs, parameter_rows)
    write_csv(
        parameter_rows,
        run_dir / f"parameter_table_{REVIEWER_SUITE}.csv",
    )
    parameters_by_name = {
        row["model"]: row for row in parameter_rows
    }

    raw_rows = []
    detailed = []
    for spec in specs:
        print("\n" + "=" * 72)
        print("MODEL:", spec["name"], "|", spec)
        print("=" * 72)
        for seed in REVIEWER_SEEDS:
            print(f"\nSeed {seed}")
            bundle = load_or_train(
                spec, seed, U_train, U_val, run_dir
            )
            metrics, pred = evaluate_model(bundle, spec, U_test)
            row = raw_result_row(
                spec,
                seed,
                bundle,
                metrics,
                parameters_by_name[spec["name"]],
            )
            raw_rows.append(row)
            detailed.append({
                "spec": spec,
                "seed": seed,
                "row": row,
                "l2_t_mean": metrics["l2_t_mean"],
                "all_full_l2": metrics["all_full_l2"],
                "all_final_l2": metrics["all_final_l2"],
            })
            print(
                f"test full={row['full_l2_mean']:.6e}, "
                f"final={row['final_l2_mean']:.6e}, "
                f"low={row['low_frequency_l2_mean']:.6e}, "
                f"high={row['high_frequency_l2_mean']:.6e}, "
                f"interface={row['interface_rmse_mean']:.6e}, "
                f"mass drift={row['max_mass_drift_mean']:.3e}"
            )
            if SAVE_ROLLOUTS:
                np.savez_compressed(
                    run_dir / (
                        f"rollout_{spec['name']}_seed{seed}.npz"
                    ),
                    pred=pred.astype(np.float32),
                    true=U_test.astype(np.float32),
                    test_idx=test_idx,
                )
            del pred, bundle
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    aggregated = aggregate_rows(raw_rows)
    write_csv(
        raw_rows, run_dir / f"summary_{REVIEWER_SUITE}_raw.csv"
    )
    write_csv(
        aggregated,
        run_dir / f"summary_{REVIEWER_SUITE}_aggregated.csv",
    )
    with open(
        run_dir / f"summary_{REVIEWER_SUITE}.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            jsonable({
                "config": config,
                "raw": raw_rows,
                "aggregated": aggregated,
                "detailed": detailed,
            }),
            handle,
            indent=2,
        )
    save_comparison_plot(
        aggregated,
        run_dir / f"reviewer_{REVIEWER_SUITE}_comparison.png",
    )
    print_aggregated(aggregated)
    print("\nSaved all results to:", run_dir)
    print(
        "Reuse the same EXPERIMENT_NAME for the next suite; identical "
        "AWFNO_LFL checkpoints will be loaded automatically."
    )


if __name__ == "__main__":
    main()
