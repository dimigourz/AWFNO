#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kolmogorov 2D vorticity benchmark for neural-operator surrogates.

The script compares five trained models (plus persistence):
    1. TB-AWFNO: temporal-basis alternating windowed Fourier neural operator
    2. AWFNO: alternating windowed Fourier neural operator
    3. WFNO: windowed Fourier neural operator baseline
    4. WFNO-PM: width-adjusted parameter-matched WFNO\n    5. FNO: non-windowed full-trajectory neuraloperator FNO baseline

Expected dataset:
    kolmogorov_dataset_parallel/kolmogorov_dataset.pt

Required keys:
    U or omega      : array of shape (N, Nx, Ny, Nt)
    x, y            : spatial grids
    times           : time grid
    train_indices   : training split
    val_indices     : validation split
    test_indices    : test split

Internal layout:
    S has shape (N, Nx, Ny, Nt, 1), where the last channel stores vorticity.

Evaluation protocol:
    - train, validation, and test trajectories are disjoint;
    - the same test set is used for all sample sizes;
    - the temporal basis is computed from training windows only;
    - test trajectories are evaluated without test-window fitting or basis projection;
    - windowed models are initialized only from the true initial condition;
    - subsequent window inputs are previous predictions.

Primary metric:
    absolute physical L2 error.
"""

import os, gc, json, copy, time, random, hashlib, csv
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 0. CONFIG
# ============================================================

SEED = 1234
RAW_DATA_DIR = Path(os.environ.get('NS_DATA_DIR','/content/drive/MyDrive/Kolmogorov_NS_Re500_start0.25_horizon0.25_dt0.0025_v1'))
DATA_PATH = RAW_DATA_DIR / "kolmogorov_dataset.pt"
FIELD_KEY = "U"  # "U" or "omega"

# Use 2 for 128x128 -> 64x64. Use 1 for full 128x128.
SPATIAL_STRIDE = 1
TIME_SUBSAMPLE = 1

SAMPLE_SIZES = [int(v) for v in os.environ.get('NS_SIZES','25,50,100,200').split(',')]
assert SAMPLE_SIZES and set(SAMPLE_SIZES).issubset({25,50,100,200})
N_TRAIN_POOL = max(SAMPLE_SIZES)
N_VAL = 30
N_TEST = 70
SPLIT_SEED = 2026
USE_FILE_SPLITS = True

WINDOW_K = int(os.environ.get('NS_K','20'))
WINDOW_STRIDE = int(os.environ.get('NS_STRIDE',str(WINDOW_K)))
N_TIME_MODES = int(os.environ.get('NS_TB_RANK','12'))
assert WINDOW_K>=1 and WINDOW_K<101 and 1<=N_TIME_MODES<=WINDOW_K

BATCH_SIZE = int(os.environ.get('NS_BATCH','8'))
EPOCHS = int(os.environ.get('NS_EPOCHS','120'))
assert BATCH_SIZE>=1 and EPOCHS>=1
LR = 1e-3
WEIGHT_DECAY = 1e-6

WIDTH = 32
N_LAYERS = 4
MODES_X = 16
MODES_Y = 16
LOCAL_RADIUS = 2

LOSS_KIND = "mse"
HUBER_BETA = 0.02
FINAL_LOSS_WEIGHT = 0.0
GRAD_LOSS_WEIGHT = 0.0
NORMALIZE_LOSS = True
ENFORCE_MEAN = False

FORCE_RETRAIN = os.environ.get('NS_FORCE_RETRAIN','0')=='1'
USE_SPLIT_CACHE = False
SAVE_SPLIT_CACHE = False

RUN_EXACT_HASH_OVERLAP_CHECK = True
RUN_DUPLICATE_REPORT_INSIDE_SPLITS = True
RUN_ROLLOUT_INPUT_GAP_CHECK = True

DEBUG_SHUFFLE_TRAIN_OUTPUTS = False
DEBUG_RANDOM_BASIS = False
INPUT_GAP_TOL = 1e-12

SAVE_PLOTS = True
MAKE_TRAINING_CURVE_PLOTS = True
MAKE_L2_PER_TIME_PLOT = True
MAKE_FIELD_PLOTS = True
PLOT_CHANNEL = 0
PLOT_TIME_INDICES_MANUAL = [0, 25, 50, 75, 'final']
MAX_TEST_CASES_TO_PLOT = 2
PLOT_DPI = 140
L2_CHUNK_SIZE = 8
SAVE_ROLLOUT_ARRAYS = False

X_LEFT, X_RIGHT = 0.0, 2*np.pi
Y_LEFT, Y_RIGHT = 0.0, 2*np.pi
DT_FALLBACK = 0.0025

CACHE_ROOT = os.environ.get('NS_OUTPUT_ROOT',
    '/content/drive/MyDrive/AWFNO_NS_Re500_JCP_FIVE_MODELS_v1')
assert 'FULL_ABLATIONS_v1' not in CACHE_ROOT

MODEL_DIR = os.path.join(CACHE_ROOT, "models")
PLOT_DIR = os.path.join(CACHE_ROOT, "plots")
DATA_DIR = os.path.join(CACHE_ROOT, "data")
RESULTS_DIR = os.path.join(CACHE_ROOT, "results")
for d in [MODEL_DIR, PLOT_DIR, DATA_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 1. DEVICE / GLOBAL GEOMETRY
# ============================================================

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

x_grid_np = y_grid_np = t_grid = None
Nx = Ny = Nt = C_DYN = None
DX = DY = DT_SNAPSHOT = None

def set_geometry(S_shape, x=None, y=None, t=None):
    global x_grid_np, y_grid_np, t_grid, Nx, Ny, Nt, C_DYN, DX, DY, DT_SNAPSHOT
    _, Nx, Ny, Nt, C_DYN = S_shape
    if x is None:
        x_grid_np = np.linspace(X_LEFT, X_RIGHT, Nx, endpoint=False, dtype=np.float32)
    else:
        x_grid_np = np.asarray(x, dtype=np.float32).reshape(-1)[:Nx]
    if y is None:
        y_grid_np = np.linspace(Y_LEFT, Y_RIGHT, Ny, endpoint=False, dtype=np.float32)
    else:
        y_grid_np = np.asarray(y, dtype=np.float32).reshape(-1)[:Ny]
    DX = float(np.mean(np.diff(x_grid_np.astype(np.float64)))) if Nx > 1 else 1.0
    DY = float(np.mean(np.diff(y_grid_np.astype(np.float64)))) if Ny > 1 else 1.0
    if t is None or len(t) != Nt:
        t_grid = np.arange(Nt, dtype=np.float32) * DT_FALLBACK
        DT_SNAPSHOT = float(DT_FALLBACK)
    else:
        t_grid = np.asarray(t, dtype=np.float32).reshape(-1)
        DT_SNAPSHOT = float(np.mean(np.diff(t_grid.astype(np.float64)))) if Nt > 1 else float(DT_FALLBACK)
    print("\nGeometry")
    print("--------")
    print("Nx, Ny, Nt, C:", Nx, Ny, Nt, C_DYN)
    print("DX, DY, DT   :", DX, DY, DT_SNAPSHOT)

# ============================================================
# 2. HELPERS
# ============================================================

def short_hash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:10]

def clean_state_dict(sd):
    sd = copy.deepcopy(sd)
    if isinstance(sd, dict) and "_metadata" in sd:
        del sd["_metadata"]
    if hasattr(sd, "_metadata"):
        try: del sd._metadata
        except Exception: pass
    return OrderedDict((k, v) for k, v in sd.items() if k != "_metadata")

def save_model_checkpoint(path, model, meta=None):
    torch.save({"state_dict": clean_state_dict(model.state_dict()), "meta": meta or {}}, path)

def load_model_checkpoint(path, model):
    payload = torch.load(path, map_location=device)
    model.load_state_dict(clean_state_dict(payload["state_dict"]), strict=True)
    return payload.get("meta", {})

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def gb(nbytes): return nbytes / 1024**3

def print_array_info(name, arr):
    print(f"{name:18s}: shape={arr.shape}, dtype={arr.dtype}, memory={gb(arr.nbytes):.3f} GB, "
          f"min={float(np.nanmin(arr)):+.4e}, max={float(np.nanmax(arr)):+.4e}, "
          f"mean={float(np.nanmean(arr)):+.4e}, std={float(np.nanstd(arr)):.4e}")

def time_weights(nt):
    w = np.ones(nt, dtype=np.float64)
    if nt > 1:
        w[0] = 0.5
        w[-1] = 0.5
    return w

# ============================================================
# 3. HDF5 LOADER
# ============================================================


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def split_cache_path():
    return os.path.join(DATA_DIR, "split_" + short_hash({
        "path": str(DATA_PATH),
        "field": FIELD_KEY,
        "stride": SPATIAL_STRIDE,
        "tsub": TIME_SUBSAMPLE,
        "sizes": SAMPLE_SIZES,
        "use_file_splits": USE_FILE_SPLITS,
        "seed": SPLIT_SEED,
    }) + ".npz")


def make_random_split_indices(n_total):
    need = N_TRAIN_POOL + N_VAL + N_TEST
    if need > n_total:
        raise ValueError(f"Need {need} trajectories but file has only {n_total}.")
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(n_total)
    train_idx = perm[:N_TRAIN_POOL]
    val_idx = perm[N_TRAIN_POOL:N_TRAIN_POOL+N_VAL]
    test_idx = perm[N_TRAIN_POOL+N_VAL:N_TRAIN_POOL+N_VAL+N_TEST]
    return train_idx, val_idx, test_idx


def load_split_cache():
    path = split_cache_path()
    if not USE_SPLIT_CACHE or not os.path.exists(path):
        return None
    print("\nLoading split cache:", path)
    z = np.load(path, allow_pickle=True)
    return (z["S_train_pool"].astype(np.float32), z["S_val"].astype(np.float32), z["S_test"].astype(np.float32),
            z["train_idx"], z["val_idx"], z["test_idx"],
            z["x_grid"].astype(np.float32), z["y_grid"].astype(np.float32), z["t_grid"].astype(np.float32))


def save_split_cache(S_train_pool, S_val, S_test, train_idx, val_idx, test_idx, x, y, t):
    if not SAVE_SPLIT_CACHE:
        return
    path = split_cache_path()
    np.savez(path, S_train_pool=S_train_pool, S_val=S_val, S_test=S_test,
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
             x_grid=x.astype(np.float32), y_grid=y.astype(np.float32), t_grid=t.astype(np.float32))
    print("Saved split cache:", path)


def load_or_create_split():
    cached = load_split_cache()
    if cached is not None:
        return cached
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    data = safe_torch_load(DATA_PATH)

    print("\nKeys in .pt file")
    print("----------------")
    for k in data.keys():
        print(" ", k)

    if FIELD_KEY not in data:
        raise KeyError(f"FIELD_KEY={FIELD_KEY!r} not found. Available keys: {list(data.keys())}")

    U = data[FIELD_KEY]
    U = U.float().cpu().numpy() if torch.is_tensor(U) else np.asarray(U, dtype=np.float32)

    if U.ndim != 4:
        raise ValueError(f"Expected {FIELD_KEY} with shape (N,Nx,Ny,Nt). Got {U.shape}")

    x = data["x"]
    y = data["y"]
    t = data["times"]
    x = x.float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)
    y = y.float().cpu().numpy() if torch.is_tensor(y) else np.asarray(y, dtype=np.float32)
    t = t.float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, dtype=np.float32)

    U = U[:, ::SPATIAL_STRIDE, ::SPATIAL_STRIDE, ::TIME_SUBSAMPLE].astype(np.float32, copy=False)
    x = x[::SPATIAL_STRIDE].astype(np.float32)
    y = y[::SPATIAL_STRIDE].astype(np.float32)
    t = t[::TIME_SUBSAMPLE].astype(np.float32)

    S = U[..., None].astype(np.float32, copy=False)  # (N,Nx,Ny,Nt,1)

    if USE_FILE_SPLITS:
        required = ["train_indices", "val_indices", "test_indices"]
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"USE_FILE_SPLITS=True, but missing split keys: {missing}")

        train_idx = data["train_indices"]
        val_idx = data["val_indices"]
        test_idx = data["test_indices"]

        train_idx = train_idx.long().cpu().numpy() if torch.is_tensor(train_idx) else np.asarray(train_idx, dtype=np.int64)
        val_idx = val_idx.long().cpu().numpy() if torch.is_tensor(val_idx) else np.asarray(val_idx, dtype=np.int64)
        test_idx = test_idx.long().cpu().numpy() if torch.is_tensor(test_idx) else np.asarray(test_idx, dtype=np.int64)

        if max(SAMPLE_SIZES) > len(train_idx):
            raise ValueError(f"max(SAMPLE_SIZES)={max(SAMPLE_SIZES)} but train split has only {len(train_idx)} samples.")

        S_train_pool = S[train_idx]
        S_val = S[val_idx]
        S_test = S[test_idx]

    else:
        train_idx, val_idx, test_idx = make_random_split_indices(S.shape[0])
        S_train_pool = S[train_idx]
        S_val = S[val_idx]
        S_test = S[test_idx]

    print("\nLoaded Kolmogorov data")
    print("----------------------")
    print("FIELD_KEY:", FIELD_KEY)
    print("S shape  :", S.shape, "= (N,Nx,Ny,Nt,1)")
    print("x shape  :", x.shape)
    print("y shape  :", y.shape)
    print("t shape  :", t.shape)

    for key in ["Re", "nu", "forcing_mode", "dt", "T_spinup", "T_forecast", "axis_order"]:
        if key in data:
            val = data[key]
            if torch.is_tensor(val):
                val = val.item() if val.numel() == 1 else val.cpu().tolist()
            print(f"{key}: {val}")

    print("\nSplit")
    print("-----")
    print("S_train_pool:", S_train_pool.shape)
    print("S_val       :", S_val.shape)
    print("S_test      :", S_test.shape)
    print("train_idx first/last:", train_idx[:5], train_idx[-5:])
    print("val_idx   first/last:", val_idx[:5], val_idx[-5:])
    print("test_idx  first/last:", test_idx[:5], test_idx[-5:])

    save_split_cache(S_train_pool, S_val, S_test, train_idx, val_idx, test_idx, x, y, t)
    return S_train_pool, S_val, S_test, train_idx, val_idx, test_idx, x, y, t

# ============================================================
# 4. CHECKS / METRICS
# ============================================================

def trajectory_hashes(S):
    S32 = np.ascontiguousarray(S.astype(np.float32, copy=False))
    return [hashlib.sha1(S32[i].tobytes()).hexdigest() for i in range(S32.shape[0])]

def check_exact_overlap(A, B, name_a, name_b):
    inter = set(trajectory_hashes(A)).intersection(set(trajectory_hashes(B)))
    print(f"\nExact overlap check {name_a} vs {name_b}: duplicates={len(inter)}")
    if inter:
        raise RuntimeError(f"Duplicate trajectories found between {name_a} and {name_b}.")

def report_duplicates(S, name):
    h = trajectory_hashes(S)
    seen, dup = {}, []
    for i, x in enumerate(h):
        if x in seen: dup.append((seen[x], i))
        else: seen[x] = i
    print(f"\nDuplicate check inside {name}: total={len(h)}, unique={len(seen)}, duplicates={len(dup)}")
    if dup: print("first duplicate pairs:", dup[:10])
    return dup

def abs_l2_window_batch(pred, true):
    return torch.sqrt(torch.sum((pred - true)**2, dim=(1,2,3,4))*DX*DY*DT_SNAPSHOT + 1e-30)

def abs_l2_field_batch(pred, true):
    return torch.sqrt(torch.sum((pred - true)**2, dim=(1,2,3))*DX*DY + 1e-30)

def l2_per_timestep_np(pred, true):
    n, nx, ny, nt, c = pred.shape
    out = np.zeros((n, nt), dtype=np.float64)
    for i0 in range(0, n, L2_CHUNK_SIZE):
        i1 = min(i0 + L2_CHUNK_SIZE, n)
        err = pred[i0:i1].astype(np.float32) - true[i0:i1].astype(np.float32)
        out[i0:i1] = np.sqrt(np.sum(err*err, axis=(1,2,4), dtype=np.float64)*DX*DY + 1e-30)
    return out

def rollout_l2_metrics(pred, true):
    l2_t = l2_per_timestep_np(pred, true)
    w = time_weights(l2_t.shape[1]).reshape(1,-1)
    full = np.sqrt(np.sum(w*l2_t**2, axis=1)*DT_SNAPSHOT + 1e-30)
    final = l2_t[:, -1]
    return {
        "full_l2_mean": float(np.mean(full)), "full_l2_std": float(np.std(full)), "full_l2_median": float(np.median(full)),
        "final_l2_mean": float(np.mean(final)), "final_l2_std": float(np.std(final)), "final_l2_median": float(np.median(final)),
        "mean_t": np.mean(l2_t, axis=0), "median_t": np.median(l2_t, axis=0), "std_t": np.std(l2_t, axis=0),
        "all_full_l2": full, "all_final_l2": final, "all_l2_t": l2_t,
    }

# ============================================================
# 5. DATASETS / BASIS / NORMALIZER
# ============================================================

def make_window_start_indices(nt, K, stride):
    if nt <= K:
        raise ValueError(f"Need Nt > K. Got Nt={nt}, K={K}")
    return list(range(0, nt-K, stride))

class WindowDataset(Dataset):
    def __init__(self, S, starts, shuffle_targets=False):
        self.S = S.astype(np.float32, copy=False)
        self.starts = list(starts)
        self.n = self.S.shape[0]
        self.ns = len(self.starts)
        self.shuffle_targets = shuffle_targets
        self.target_perm = np.random.default_rng(SEED).permutation(self.n*self.ns) if shuffle_targets else None
    def __len__(self): return self.n*self.ns
    def decode(self, idx):
        i = idx // self.ns
        s = self.starts[idx % self.ns]
        return i, s
    def __getitem__(self, idx):
        i, s = self.decode(idx)
        if self.target_perm is None:
            it, st = i, s
        else:
            it, st = self.decode(int(self.target_perm[idx]))
        return {
            "state_start": torch.tensor(self.S[i,:,:,s,:], dtype=torch.float32),
            "state_future": torch.tensor(self.S[it,:,:,st+1:st+WINDOW_K+1,:], dtype=torch.float32),
        }

class FullDataset(Dataset):
    def __init__(self, S):
        self.S = S.astype(np.float32, copy=False)
    def __len__(self): return self.S.shape[0]
    def __getitem__(self, idx):
        return {
            "state0": torch.tensor(self.S[idx,:,:,0,:], dtype=torch.float32),
            "future": torch.tensor(self.S[idx,:,:,1:,:], dtype=torch.float32),
            "full": torch.tensor(self.S[idx], dtype=torch.float32),
        }

def compute_tb_basis(S, starts, n_modes):
    K = WINDOW_K
    cov = np.zeros((K,K), dtype=np.float64)
    rows = 0
    for i in range(S.shape[0]):
        for s in starts:
            S0 = S[i,:,:,s,:]
            Sf = S[i,:,:,s+1:s+K+1,:]
            R = Sf - S0[:,:,None,:]
            X = np.transpose(R, (0,1,3,2)).reshape(-1,K).astype(np.float64)
            cov += X.T @ X
            rows += X.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:,order]
    r = min(n_modes, K)
    basis = eigvecs[:,:r].astype(np.float32)
    energy = np.maximum(eigvals, 0)
    ratio = float(np.sum(energy[:r])/(np.sum(energy)+1e-30))
    print("\nTemporal basis")
    print("--------------")
    print("rows:", rows, "rank:", r, "energy:", f"{ratio:.10f}", "singular:", np.sqrt(energy)[:10])
    return basis, ratio, np.sqrt(energy).astype(np.float32)

def random_basis(K, M):
    Q, _ = np.linalg.qr(np.random.randn(K, M))
    return Q[:, :M].astype(np.float32)

def reconstruct_tb(state_start, coeff, basis, enforce_mean=False):
    R = torch.einsum("bxycm,km->bxykc", coeff, basis)
    Sf = state_start.unsqueeze(3) + R
    if enforce_mean:
        m0 = state_start.mean(dim=(1,2), keepdim=True).unsqueeze(3)
        Sf = Sf + m0 - Sf.mean(dim=(1,2), keepdim=True)
    return Sf

def enforce_mean_future(Sf, state_start):
    m0 = state_start.mean(dim=(1,2), keepdim=True).unsqueeze(3)
    return Sf + m0 - Sf.mean(dim=(1,2), keepdim=True)

class ChannelNormalizer:
    def __init__(self, mean, std, eps=1e-6):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(np.maximum(std, 1e-8), dtype=torch.float32)
        self.eps = eps
    def _view(self, x):
        if x.dim() == 4:
            return self.mean.view(1,1,1,-1), self.std.view(1,1,1,-1)
        if x.dim() == 5:
            return self.mean.view(1,1,1,1,-1), self.std.view(1,1,1,1,-1)
        raise ValueError(x.shape)
    def encode(self, x):
        m, s = self._view(x); return (x-m)/(s+self.eps)
    def decode(self, x):
        m, s = self._view(x); return x*(s+self.eps)+m
    def to(self, device):
        self.mean = self.mean.to(device); self.std = self.std.to(device); return self

def channel_stats(S):
    mean = np.mean(S, axis=(0,1,2,3), dtype=np.float64).astype(np.float32)
    std = np.std(S, axis=(0,1,2,3), dtype=np.float64).astype(np.float32)
    return mean, std

# ============================================================
# 6. MODELS
# ============================================================

class SpectralConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, modes_x, modes_y):
        super().__init__()
        scale = 1.0/np.sqrt(in_ch*out_ch)
        self.modes_x = modes_x; self.modes_y = modes_y
        self.weights_pos = nn.Parameter(scale*torch.randn(in_ch, out_ch, modes_x, modes_y, dtype=torch.cfloat))
        self.weights_neg = nn.Parameter(scale*torch.randn(in_ch, out_ch, modes_x, modes_y, dtype=torch.cfloat))
    @staticmethod
    def cmul(a, b): return torch.einsum("bixy,ioxy->boxy", a, b)
    def forward(self, x):
        B,C,nx,ny = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.weights_pos.shape[1], nx, ny//2+1, dtype=torch.cfloat, device=x.device)
        mx = min(self.modes_x, nx); my = min(self.modes_y, ny//2+1)
        out_ft[:,:,:mx,:my] = self.cmul(x_ft[:,:,:mx,:my], self.weights_pos[:,:,:mx,:my])
        out_ft[:,:,-mx:,:my] = self.cmul(x_ft[:,:,-mx:,:my], self.weights_neg[:,:,:mx,:my])
        return torch.fft.irfft2(out_ft, s=(nx,ny), norm="ortho")

class ExactLKLBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.L1 = nn.Conv2d(width, width, 2*LOCAL_RADIUS+1, padding=LOCAL_RADIUS, padding_mode="circular")
        self.K = SpectralConv2d(width, width, MODES_X, MODES_Y)
        self.L2 = nn.Conv2d(width, width, 2*LOCAL_RADIUS+1, padding=LOCAL_RADIUS, padding_mode="circular")
        self.sigma = nn.GELU()
    def forward(self, v):
        v1 = v + self.sigma(self.L1(v))
        v2 = v1 + self.sigma(self.K(v1))
        return v2 + self.sigma(self.L2(v2))

class GridMixin:
    def make_grid(self):
        # True 2pi-periodic torus coordinates, with known Kolmogorov forcing phase.
        X,Y=torch.meshgrid(torch.as_tensor(x_grid_np,dtype=torch.float32),
                           torch.as_tensor(y_grid_np,dtype=torch.float32),indexing='ij')
        return torch.stack((X.sin(),X.cos(),Y.sin(),Y.cos(),
                            (4*Y).sin(),(4*Y).cos()),dim=-1)

class AWEncoder(nn.Module, GridMixin):
    def __init__(self, c_in):
        super().__init__()
        self.register_buffer("grid", self.make_grid().unsqueeze(0), persistent=False)
        self.input_proj = nn.Linear(c_in+6, WIDTH)
        self.blocks = nn.ModuleList([ExactLKLBlock(WIDTH) for _ in range(N_LAYERS)])
    def forward(self, s_norm):
        B = s_norm.shape[0]
        z = torch.cat([s_norm, self.grid.repeat(B,1,1,1)], dim=-1)
        z = self.input_proj(z).permute(0,3,1,2).contiguous()
        for b in self.blocks: z = b(z)
        return z.permute(0,2,3,1).contiguous()

class FNOEncoder(nn.Module, GridMixin):
    def __init__(self, c_in):
        super().__init__()
        self.register_buffer("grid", self.make_grid().unsqueeze(0), persistent=False)
        self.fc0 = nn.Linear(c_in+6, WIDTH)
        self.convs = nn.ModuleList([SpectralConv2d(WIDTH, WIDTH, MODES_X, MODES_Y) for _ in range(N_LAYERS)])
        self.ws = nn.ModuleList([nn.Conv2d(WIDTH, WIDTH, 1) for _ in range(N_LAYERS)])
    def forward(self, s_norm):
        B = s_norm.shape[0]
        z = torch.cat([s_norm, self.grid.repeat(B,1,1,1)], dim=-1)
        z = self.fc0(z).permute(0,3,1,2).contiguous()
        for i,(conv,w) in enumerate(zip(self.convs, self.ws)):
            z = conv(z) + w(z)
            if i != len(self.convs)-1: z = F.gelu(z)
        return z.permute(0,2,3,1).contiguous()

class TBAWFNO(nn.Module):
    def __init__(self, c_dyn, m):
        super().__init__()
        self.c_dyn = c_dyn; self.m = m
        self.encoder = AWEncoder(c_dyn)
        self.fc1 = nn.Linear(WIDTH, 128)
        self.fc2 = nn.Linear(128, c_dyn*m)
    def forward(self, s_norm):
        z = self.encoder(s_norm)
        out = self.fc2(F.gelu(self.fc1(z)))
        B,nx,ny,_ = out.shape
        return out.view(B,nx,ny,self.c_dyn,self.m)

class DirectWindowModel(nn.Module):
    def __init__(self, c_dyn, K, encoder_kind):
        super().__init__()
        self.c_dyn = c_dyn; self.K = K
        self.encoder = AWEncoder(c_dyn) if encoder_kind == "AW" else FNOEncoder(c_dyn)
        self.fc1 = nn.Linear(WIDTH, 128)
        self.fc2 = nn.Linear(128, c_dyn*K)
    def forward(self, s_norm):
        z = self.encoder(s_norm)
        out = self.fc2(F.gelu(self.fc1(z)))
        B,nx,ny,_ = out.shape
        return out.view(B,nx,ny,self.K,self.c_dyn)

class FNOFull(nn.Module):
    """Direct full-trajectory neuraloperator FNO, matching Boussinesq protocol."""
    def __init__(self,c_dyn,nt_future):
        super().__init__()
        from neuralop.models import FNO as ReferenceFNO
        self.c_dyn=int(c_dyn)
        self.nt_future=int(nt_future)
        grid=GridMixin().make_grid().permute(2,0,1).unsqueeze(0)
        self.register_buffer('grid',grid,persistent=False)
        self.fno=ReferenceFNO(n_modes=(MODES_X,MODES_Y),
                              in_channels=self.c_dyn+6,
                              out_channels=self.c_dyn*self.nt_future,
                              hidden_channels=WIDTH,n_layers=N_LAYERS,
                              positional_embedding=None)
    def forward(self,s0_norm):
        b,nx,ny,c=s0_norm.shape
        grid=self.grid.expand(b,-1,-1,-1)
        x=torch.cat((s0_norm.permute(0,3,1,2),grid),dim=1)
        out=self.fno(x)
        if isinstance(out,(tuple,list)):out=out[0]
        assert out.shape==(b,self.c_dyn*self.nt_future,nx,ny)
        return out.permute(0,2,3,1).reshape(b,nx,ny,self.nt_future,self.c_dyn)

class MatchedWFNO(nn.Module,GridMixin):
    """Windowed Fourier operator with optimally adjusted latent width."""
    def __init__(self,c_dyn,k,width):
        super().__init__()
        self.width=int(width);self.k=int(k);self.c_dyn=int(c_dyn)
        self.register_buffer('grid',self.make_grid().unsqueeze(0),persistent=False)
        self.fc0=nn.Linear(c_dyn+6,width)
        self.convs=nn.ModuleList([SpectralConv2d(width,width,MODES_X,MODES_Y)
                                  for _ in range(N_LAYERS)])
        self.ws=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(N_LAYERS)])
        self.fc1=nn.Linear(width,128)
        self.fc2=nn.Linear(128,c_dyn*k)
    def forward(self,s_norm):
        b,nx,ny,_=s_norm.shape
        x=self.fc0(torch.cat((s_norm,self.grid.expand(b,-1,-1,-1)),dim=-1))
        x=x.permute(0,3,1,2).contiguous()
        for i,(conv,skip) in enumerate(zip(self.convs,self.ws)):
            x=conv(x)+skip(x)
            if i<len(self.convs)-1:x=F.gelu(x)
        out=self.fc2(F.gelu(self.fc1(x.permute(0,2,3,1))))
        return out.view(b,nx,ny,self.k,self.c_dyn)

def scalar_parameters(m):
    """Complex elements count as two trainable real scalar degrees of freedom."""
    return sum(p.numel()*(2 if p.is_complex() else 1)
               for p in m.parameters() if p.requires_grad)

def matched_width(target_real,k):
    # Use actual model parameter counts, not an approximate formula.
    candidates=[]
    for width in range(8,97):
        model=MatchedWFNO(C_DYN,k,width)
        score=scalar_parameters(model)
        candidates.append((abs(score-target_real),width,score))
        del model
    return min(candidates)

def make_tb_predict(model, normer, basis):
    def predict(s):
        coeff = model(normer.encode(s))
        return coeff, reconstruct_tb(s, coeff, basis, ENFORCE_MEAN)
    return predict

def make_direct_predict(model, normer, out_normer, enforce_mean=False):
    def predict(s):
        out = out_normer.decode(model(normer.encode(s)))
        return enforce_mean_future(out, s) if enforce_mean else out
    return predict

def make_full_predict(model, normer, out_normer):
    def predict(s0):
        return out_normer.decode(model(normer.encode(s0)))
    return predict

# ============================================================
# 7. LOSSES / FIT
# ============================================================

def data_loss(pred, true):
    if LOSS_KIND == "mse": return F.mse_loss(pred, true)
    if LOSS_KIND == "huber": return F.smooth_l1_loss(pred, true, beta=HUBER_BETA)
    raise ValueError(LOSS_KIND)

def norm_loss(pred, true, normer):
    return data_loss(normer.encode(pred), normer.encode(true)) if NORMALIZE_LOSS else data_loss(pred, true)

def grad_loss(pred, true, normer):
    p = normer.encode(pred) if NORMALIZE_LOSS else pred
    t = normer.encode(true) if NORMALIZE_LOSS else true
    px = (torch.roll(p,-1,1)-torch.roll(p,1,1))/(2*DX)
    py = (torch.roll(p,-1,2)-torch.roll(p,1,2))/(2*DY)
    tx = (torch.roll(t,-1,1)-torch.roll(t,1,1))/(2*DX)
    ty = (torch.roll(t,-1,2)-torch.roll(t,1,2))/(2*DY)
    return F.smooth_l1_loss(px,tx,beta=HUBER_BETA) + F.smooth_l1_loss(py,ty,beta=HUBER_BETA)

def window_loss(pred, true, normer):
    return (norm_loss(pred, true, normer)
            + FINAL_LOSS_WEIGHT*norm_loss(pred[:,:,:,-1,:], true[:,:,:,-1,:], normer)
            + GRAD_LOSS_WEIGHT*grad_loss(pred, true, normer))

def train_epoch_windowed(model, loader, opt, predict, normer, kind):
    model.train(); tot=tot_l2=n=0
    for batch in loader:
        s0 = batch["state_start"].to(device); true = batch["state_future"].to(device)
        opt.zero_grad(set_to_none=True)
        pred = predict(s0)[1] if kind == "TB-AWFNO" else predict(s0)
        loss = window_loss(pred, true, normer)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # common clipping policy
        opt.step()
        bs=s0.shape[0]; tot += loss.item()*bs; tot_l2 += abs_l2_window_batch(pred.detach(), true).mean().item()*bs; n += bs
    return {"loss": tot/max(n,1), "abs_l2": tot_l2/max(n,1)}

@torch.no_grad()
def eval_epoch_windowed(model, loader, predict, normer, kind):
    model.eval(); tot=tot_l2=tot_final=n=0
    for batch in loader:
        s0 = batch["state_start"].to(device); true = batch["state_future"].to(device)
        pred = predict(s0)[1] if kind == "TB-AWFNO" else predict(s0)
        loss = window_loss(pred, true, normer)
        bs=s0.shape[0]; tot += loss.item()*bs; tot_l2 += abs_l2_window_batch(pred, true).mean().item()*bs
        tot_final += abs_l2_field_batch(pred[:,:,:,-1,:], true[:,:,:,-1,:]).mean().item()*bs; n += bs
    return {"loss": tot/max(n,1), "abs_l2": tot_l2/max(n,1), "final_l2": tot_final/max(n,1)}

def fit_windowed(model, train_loader, val_loader, predict, normer, kind):
    if kind in ["TB-AWFNO", "AWFNO"]:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(EPOCHS//3,1), gamma=0.5)
    best=float("inf"); best_state=None
    hist={"train_loss":[],"train_l2":[],"val_loss":[],"val_l2":[],"val_final_l2":[]}
    t0=time.time()
    for ep in range(1,EPOCHS+1):
        tr = train_epoch_windowed(model, train_loader, opt, predict, normer, kind)
        va = eval_epoch_windowed(model, val_loader, predict, normer, kind)
        sched.step()
        for k,v in [("train_loss",tr["loss"]),("train_l2",tr["abs_l2"]),("val_loss",va["loss"]),("val_l2",va["abs_l2"]),("val_final_l2",va["final_l2"])]:
            hist[k].append(v)
        if va["abs_l2"] < best:
            best = va["abs_l2"]; best_state = clean_state_dict(model.state_dict())
        if ep == 1 or ep % 10 == 0:
            print(f"[{kind}] {ep:04d}/{EPOCHS} | train loss {tr['loss']:.6e} | train L2 {tr['abs_l2']:.6e} | val loss {va['loss']:.6e} | val L2 {va['abs_l2']:.6e} | val final {va['final_l2']:.6e}")
    if best_state is not None: model.load_state_dict(best_state, strict=True)
    return {"model": model, "best_val_l2": best, "train_time_sec": time.time()-t0, "history": hist}

def train_epoch_full(model, loader, opt, predict, normer):
    model.train(); tot=tot_l2=n=0
    for batch in loader:
        s0=batch["state0"].to(device); true=batch["future"].to(device)
        opt.zero_grad(set_to_none=True)
        pred=predict(s0)
        loss = window_loss(pred, true, normer)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        bs=s0.shape[0]; tot += loss.item()*bs; tot_l2 += abs_l2_window_batch(pred.detach(), true).mean().item()*bs; n += bs
    return {"loss": tot/max(n,1), "abs_l2": tot_l2/max(n,1)}

@torch.no_grad()
def eval_epoch_full(model, loader, predict, normer):
    model.eval(); tot=tot_l2=tot_final=n=0
    for batch in loader:
        s0=batch["state0"].to(device); true=batch["future"].to(device)
        pred=predict(s0)
        loss=window_loss(pred, true, normer)
        bs=s0.shape[0]; tot += loss.item()*bs; tot_l2 += abs_l2_window_batch(pred, true).mean().item()*bs
        tot_final += abs_l2_field_batch(pred[:,:,:,-1,:], true[:,:,:,-1,:]).mean().item()*bs; n += bs
    return {"loss": tot/max(n,1), "abs_l2": tot_l2/max(n,1), "final_l2": tot_final/max(n,1)}

def fit_full(model, train_loader, val_loader, predict, normer):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(EPOCHS//3,1), gamma=0.5)
    best=float("inf"); best_state=None
    hist={"train_loss":[],"train_l2":[],"val_loss":[],"val_l2":[],"val_final_l2":[]}
    t0=time.time()
    for ep in range(1,EPOCHS+1):
        tr=train_epoch_full(model, train_loader, opt, predict, normer)
        va=eval_epoch_full(model, val_loader, predict, normer)
        sched.step()
        for k,v in [("train_loss",tr["loss"]),("train_l2",tr["abs_l2"]),("val_loss",va["loss"]),("val_l2",va["abs_l2"]),("val_final_l2",va["final_l2"])]:
            hist[k].append(v)
        if va["abs_l2"] < best:
            best=va["abs_l2"]; best_state=clean_state_dict(model.state_dict())
        if ep == 1 or ep % 10 == 0:
            print(f"[FNO] {ep:04d}/{EPOCHS} | train loss {tr['loss']:.6e} | train L2 {tr['abs_l2']:.6e} | val loss {va['loss']:.6e} | val L2 {va['abs_l2']:.6e} | val final {va['final_l2']:.6e}")
    if best_state is not None: model.load_state_dict(best_state, strict=True)
    return {"model": model, "best_val_l2": best, "train_time_sec": time.time()-t0, "history": hist}

# ============================================================
# 8. ROLLOUT / EVALUATION
# ============================================================

def check_input_gap(current, S_batch, cursor, name):
    if not RUN_ROLLOUT_INPUT_GAP_CHECK or cursor <= 0: return
    true = torch.tensor(S_batch[:,:,:,cursor,:], dtype=torch.float32, device=device)
    gap = abs_l2_field_batch(current, true).mean().item()
    print(f"  CHECK {name}: input gap from true at cursor {cursor}: {gap:.6e}")
    if gap < INPUT_GAP_TOL:
        raise RuntimeError(f"Possible cheating in {name}: input equals true state at cursor {cursor}")

@torch.no_grad()
def rollout_tb(model, S_test, predict):
    model.eval(); n,nx,ny,nt,c = S_test.shape
    out = np.zeros_like(S_test, dtype=np.float32)
    print("\nSequential rollout: TB-AWFNO")
    for i0 in range(0,n,BATCH_SIZE):
        i1=min(i0+BATCH_SIZE,n); batch=S_test[i0:i1]
        cur=torch.tensor(batch[:,:,:,0,:], dtype=torch.float32, device=device)
        out[i0:i1,:,:,0,:]=batch[:,:,:,0,:]
        cursor=0; blk=0
        while cursor < nt-1:
            check_input_gap(cur,batch,cursor,"TB-AWFNO")
            _, pred = predict(cur)
            use_k = min(WINDOW_K, nt-1-cursor)
            out[i0:i1,:,:,cursor+1:cursor+1+use_k,:] = pred[:,:,:,:use_k,:].detach().cpu().numpy()
            cur = pred[:,:,:,use_k-1,:].detach()
            cursor += use_k; blk += 1
        print(f"  TB-AWFNO predicted {i0:04d}:{i1:04d}")
    return out

@torch.no_grad()
def rollout_direct(model, S_test, predict, name):
    model.eval(); n,nx,ny,nt,c = S_test.shape
    out = np.zeros_like(S_test, dtype=np.float32)
    print(f"\nSequential rollout: {name}")
    for i0 in range(0,n,BATCH_SIZE):
        i1=min(i0+BATCH_SIZE,n); batch=S_test[i0:i1]
        cur=torch.tensor(batch[:,:,:,0,:], dtype=torch.float32, device=device)
        out[i0:i1,:,:,0,:]=batch[:,:,:,0,:]
        cursor=0
        while cursor < nt-1:
            check_input_gap(cur,batch,cursor,name)
            pred = predict(cur)
            use_k = min(WINDOW_K, nt-1-cursor)
            out[i0:i1,:,:,cursor+1:cursor+1+use_k,:] = pred[:,:,:,:use_k,:].detach().cpu().numpy()
            cur = pred[:,:,:,use_k-1,:].detach()
            cursor += use_k
        print(f"  {name} predicted {i0:04d}:{i1:04d}")
    return out

@torch.no_grad()
def predict_full(model, S_test, predict):
    model.eval(); n,nx,ny,nt,c = S_test.shape
    out = np.zeros_like(S_test, dtype=np.float32)
    out[:,:,:,0,:] = S_test[:,:,:,0,:]
    print("\nFull prediction: FNO")
    for i0 in range(0,n,BATCH_SIZE):
        i1=min(i0+BATCH_SIZE,n)
        s0=torch.tensor(S_test[i0:i1,:,:,0,:], dtype=torch.float32, device=device)
        future = predict(s0)
        out[i0:i1,:,:,1:,:] = future.detach().cpu().numpy()
        print(f"  FNO predicted {i0:04d}:{i1:04d}")
    return out

def persistence(S):
    out = np.zeros_like(S, dtype=np.float32)
    out[:] = S[:,:,:,0:1,:]
    return out

def plot_training_curves(hist, name, tag):
    if not SAVE_PLOTS or not MAKE_TRAINING_CURVE_PLOTS: return
    plt.figure(figsize=(8,5))
    plt.semilogy(hist["train_loss"], label="train")
    plt.semilogy(hist["val_loss"], label="val")
    plt.grid(True, alpha=0.3); plt.legend(); plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"{name} loss")
    plt.tight_layout(); p=os.path.join(PLOT_DIR, f"{tag}_{name}_loss.png"); plt.savefig(p, dpi=PLOT_DPI); plt.close(); print("Saved:", p)
    plt.figure(figsize=(8,5))
    plt.semilogy(hist["train_l2"], label="train L2")
    plt.semilogy(hist["val_l2"], label="val L2")
    plt.semilogy(hist["val_final_l2"], label="val final L2")
    plt.grid(True, alpha=0.3); plt.legend(); plt.xlabel("epoch"); plt.ylabel("absolute L2"); plt.title(f"{name} L2")
    plt.tight_layout(); p=os.path.join(PLOT_DIR, f"{tag}_{name}_l2.png"); plt.savefig(p, dpi=PLOT_DPI); plt.close(); print("Saved:", p)

def plot_l2_per_time(stats, tag):
    if not SAVE_PLOTS or not MAKE_L2_PER_TIME_PLOT: return
    plt.figure(figsize=(8,5))
    for name, st in stats.items():
        plt.plot(t_grid, st["mean_t"], label=name)
    plt.grid(True, alpha=0.3); plt.legend(); plt.xlabel("time"); plt.ylabel("absolute vorticity L2")
    plt.title("2D Kolmogorov L2 per time")
    plt.tight_layout(); p=os.path.join(PLOT_DIR, f"{tag}_l2_per_time.png"); plt.savefig(p, dpi=PLOT_DPI); plt.close(); print("Saved:", p)

def plot_fields(S_true, rollouts, tag):
    if not SAVE_PLOTS or not MAKE_FIELD_PLOTS: return
    out_dir=os.path.join(PLOT_DIR, f"{tag}_fields_ch{PLOT_CHANNEL}"); os.makedirs(out_dir, exist_ok=True)
    times=[]
    for item in PLOT_TIME_INDICES_MANUAL:
        k = S_true.shape[-2]-1 if item == "final" else int(item)
        if 0 <= k < S_true.shape[-2] and k not in times: times.append(k)
    ch = min(PLOT_CHANNEL, S_true.shape[-1]-1)
    names = list(rollouts.keys())
    for i in range(min(MAX_TEST_CASES_TO_PLOT, S_true.shape[0])):
        for k in times:
            true = S_true[i,:,:,k,ch]
            fields=[true] + [rollouts[n][i,:,:,k,ch] for n in names]
            errs=[np.abs(rollouts[n][i,:,:,k,ch]-true) for n in names]
            vmin=min(float(a.min()) for a in fields); vmax=max(float(a.max()) for a in fields); emax=max(float(e.max()) for e in errs)+1e-12
            fig, axes = plt.subplots(1, 1+2*len(names), figsize=(3*(1+2*len(names)),3.2), constrained_layout=True)
            axes=np.asarray(axes).ravel(); col=0
            im=axes[col].imshow(true.T, origin="lower", vmin=vmin, vmax=vmax); axes[col].set_title("true"); axes[col].axis("off"); col+=1
            for n in names:
                axes[col].imshow(rollouts[n][i,:,:,k,ch].T, origin="lower", vmin=vmin, vmax=vmax); axes[col].set_title(n); axes[col].axis("off"); col+=1
            imerr=None
            for n in names:
                imerr=axes[col].imshow(np.abs(rollouts[n][i,:,:,k,ch]-true).T, origin="lower", vmin=0, vmax=emax); axes[col].set_title(n+" err"); axes[col].axis("off"); col+=1
            fig.colorbar(im, ax=axes[:1+len(names)].tolist(), shrink=0.7)
            fig.colorbar(imerr, ax=axes[1+len(names):].tolist(), shrink=0.7)
            fig.suptitle(f"test {i}, channel {ch}, t index {k}")
            p=os.path.join(out_dir, f"test_{i:03d}_t{k:04d}.png"); plt.savefig(p, dpi=PLOT_DPI); plt.close(fig); print("Saved:", p)

# ============================================================
# 9. TRAIN/EVAL SWEEP
# ============================================================

def base_tag(n_train):
    return short_hash({"N": n_train, "Nx": Nx, "Ny": Ny, "Nt": Nt, "C": C_DYN, "K": WINDOW_K,
                       "M": N_TIME_MODES, "W": WIDTH, "L": N_LAYERS, "mx": MODES_X, "my": MODES_Y,
                       "stride": SPATIAL_STRIDE, "tsub": TIME_SUBSAMPLE})

def build_objects(n_train, S_train_pool, S_val):
    S_train = S_train_pool[:n_train]
    starts = make_window_start_indices(Nt, WINDOW_K, WINDOW_STRIDE)
    val_starts = make_window_start_indices(Nt, WINDOW_K, WINDOW_K)
    print("\nWindow starts:", starts[:10], "n=", len(starts), "val n=", len(val_starts))
    # The temporal basis is a training-data object, independent of the model
    # architecture and random initialization. Cache its COMPLETE eigensystem
    # once per (N,K,stride), and truncate to TB ranks 6/12/18 as necessary.
    # This avoids recomputing an expensive covariance matrix for every ablation.
    cachebasis=Path(DATA_DIR)/f'temporal_basis_N{n_train}_K{WINDOW_K}_s{WINDOW_STRIDE}.npz'
    if cachebasis.exists() and not DEBUG_RANDOM_BASIS:
        with np.load(cachebasis) as cached:
            vectors=cached['vectors']
            sing=cached['singular_values']
        print('[BASIS CACHE]',cachebasis,flush=True)
    else:
        vectors, _, sing = compute_tb_basis(S_train, starts, WINDOW_K)
        np.savez_compressed(cachebasis,vectors=vectors,singular_values=sing)
        print('[BASIS SAVED]',cachebasis,flush=True)
    basis=vectors[:,:N_TIME_MODES].astype(np.float32)
    energy=float(np.sum(sing[:N_TIME_MODES].astype(np.float64)**2)/
                 max(np.sum(sing.astype(np.float64)**2),1e-30))
    # Temporal covariance N/K cache does not depend on parameter ablations.
    # Full-rank eigensystem is stored to Drive and safely reused across runs.
    if DEBUG_RANDOM_BASIS:
        basis = random_basis(WINDOW_K, N_TIME_MODES); energy=float("nan"); sing=np.ones(N_TIME_MODES, dtype=np.float32)
    basis_t = torch.tensor(basis, dtype=torch.float32, device=device)
    mean, std = channel_stats(S_train)
    normer = ChannelNormalizer(mean, std).to(device)
    for c in range(C_DYN): print(f"normalizer ch {c}: mean {mean[c]:+.6e}, std {std[c]:.6e}")
    train_ds = WindowDataset(S_train, starts, DEBUG_SHUFFLE_TRAIN_OUTPUTS)
    val_ds = WindowDataset(S_val, val_starts, False)
    train_full_ds = FullDataset(S_train)
    val_full_ds = FullDataset(S_val)
    gen=torch.Generator(); gen.manual_seed(SEED+n_train)
    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,generator=gen,num_workers=0,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    train_full_loader=DataLoader(train_full_ds,batch_size=BATCH_SIZE,shuffle=True,generator=gen,num_workers=0,pin_memory=torch.cuda.is_available())
    val_full_loader=DataLoader(val_full_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    b=next(iter(train_loader)); bf=next(iter(train_full_loader))
    print("batch window:", tuple(b["state_start"].shape), tuple(b["state_future"].shape))
    print("batch full  :", tuple(bf["state0"].shape), tuple(bf["future"].shape))
    return {"S_train": S_train, "basis": basis, "basis_t": basis_t, "energy": energy,
            "train_loader": train_loader, "val_loader": val_loader,
            "train_full_loader": train_full_loader, "val_full_loader": val_full_loader, "normer": normer}


def _checkpoint_fit(model,train_loader,val_loader,predict,normer,kind,path,is_full=False):
    """Boussinesq-style epoch training with resumable optimizer state and Drive logs.
    Validation is WINDOW loss for windowed models, FULL direct-time loss for FNO.
    Independent physical-space rollouts are calculated in the evaluation routine.
    """
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,
                                                         eta_min=LR*.03)
    meta={'kind':kind,'epochs':EPOCHS,'k':WINDOW_K,'basis_rank':N_TIME_MODES,
          'modes':(MODES_X,MODES_Y),'radius':LOCAL_RADIUS,'layers':N_LAYERS,
          'batch':BATCH_SIZE,'width':WIDTH,'seed':SEED,'data_hash':os.environ.get(
          'NS_DATA_SHA','7030bc43e6331cdf472182ae0ed4f2684c18da8d336ca9ed92c37baf0bfc8a06')}
    epoch0=0;best=float('inf');best_model=None;elapsed=0.
    hist={'train_loss':[],'train_l2':[],'val_loss':[],'val_l2':[],
          'val_final_l2':[]}
    if path.exists() and not FORCE_RETRAIN:
        try:state=torch.load(path,map_location='cpu',weights_only=False)
        except TypeError:state=torch.load(path,map_location='cpu')
        assert state['config']==meta, 'Checkpoint configuration mismatch: '+str(path)
        model.load_state_dict(state['weights'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch0=int(state['epoch']);best=float(state['best'])
        best_model=state['best_model'];hist=state['history'];elapsed=float(state['elapsed'])
        if 'rng_cpu' in state:torch.set_rng_state(state['rng_cpu'])
        if 'rng_cuda' in state and state['rng_cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        if 'shuffle_generator' in state and train_loader.generator is not None:
            train_loader.generator.set_state(state['shuffle_generator'])
        print(f'[RESUME] {kind} epoch {epoch0}/{EPOCHS}',flush=True)
    since=time.perf_counter()
    for ep in range(epoch0+1,EPOCHS+1):
        if is_full:
            tr=train_epoch_full(model,train_loader,optimizer,predict,normer)
            va=eval_epoch_full(model,val_loader,predict,normer)
        else:
            tr=train_epoch_windowed(model,train_loader,optimizer,predict,normer,kind)
            va=eval_epoch_windowed(model,val_loader,predict,normer,kind)
        scheduler.step()
        for h,key in [('train_loss','loss'),('train_l2','abs_l2')]:
            hist[h].append(float(tr[key]))
        for h,key in [('val_loss','loss'),('val_l2','abs_l2'),('val_final_l2','final_l2')]:
            hist[h].append(float(va[key]))
        if va['abs_l2']<best:
            best=float(va['abs_l2'])
            best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f'[EPOCH] {kind}: {ep}/{EPOCHS} train={tr["loss"]:.4e}, '
              f'val={va["loss"]:.4e}, val_window_full_L2={va["abs_l2"]:.4f}, '
              f'best={best:.4f}; elapsed={(elapsed+time.perf_counter()-since)/60:.1f}min',
              flush=True)
        if ep==1 or ep%5==0 or ep==EPOCHS:
            if torch.cuda.is_available():torch.cuda.synchronize()
            state={'config':meta,'epoch':ep,
                   'weights':{k:v.detach().cpu().clone()
                              for k,v in model.state_dict().items()},
                   'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                   'best':best,'best_model':best_model,'history':hist,
                   'elapsed':elapsed+time.perf_counter()-since,
                   'rng_cpu':torch.get_rng_state(),
                   'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                   'shuffle_generator':train_loader.generator.get_state()
                       if train_loader.generator is not None else None}
            temporary=path.with_suffix('.tmp')
            torch.save(state,temporary);temporary.replace(path)
        # Epoch-based patience, similar to Boussinesq, with warm-up.
        if ep>=60 and len(hist['val_l2'])>=26:
            recent=hist['val_l2'][-25:]
            if min(recent)>=min(hist['val_l2'][:-25]):
                print('[EARLY STOP] no new validation best for 25 epochs',flush=True)
                break
    if best_model is None:raise RuntimeError('Never validated '+kind)
    model.load_state_dict(best_model)
    return {'model':model,'best_val_l2':best,
            'train_time_sec':elapsed+time.perf_counter()-since,'history':hist}


def train_models(n_train, S_train_pool, S_val):
    tag = base_tag(n_train)
    print("\n"+"="*80); print(f"TRAINING N_train={n_train} | tag={tag}"); print("="*80)
    obj = build_objects(n_train, S_train_pool, S_val)
    normer = obj["normer"]
    tb = TBAWFNO(C_DYN, N_TIME_MODES).to(device)
    aw = DirectWindowModel(C_DYN, WINDOW_K, "AW").to(device)
    wf = DirectWindowModel(C_DYN, WINDOW_K, "FNO").to(device)
    target_dof=scalar_parameters(aw)
    difference,pm_width,pm_dof=matched_width(target_dof,WINDOW_K)
    wf_pm=MatchedWFNO(C_DYN,WINDOW_K,pm_width).to(device)
    print(f'[MATCHED] WFNO width={pm_width}, AWFNO real DOF={target_dof:,}, '
          f'WFNO-PM real DOF={pm_dof:,}, relative mismatch={difference/target_dof:.2%}',flush=True)
    fno = FNOFull(C_DYN, Nt-1).to(device)
    preds = {
        "TB-AWFNO": make_tb_predict(tb, normer, obj["basis_t"]),
        "AWFNO": make_direct_predict(aw, normer, normer, ENFORCE_MEAN),
        "WFNO": make_direct_predict(wf, normer, normer, False),
        "WFNO-PM": make_direct_predict(wf_pm,normer,normer,False),
        "FNO": make_full_predict(fno, normer, normer),
    }
    models = {"TB-AWFNO": tb, "AWFNO": aw, "WFNO": wf, "WFNO-PM": wf_pm, "FNO": fno}
    print("\nParameter counts")
    for n,m in models.items():
        print(f'{n:10s}: {count_params(m):,} tensor elements / '
              f'{scalar_parameters(m):,} real DOF',flush=True)
    fits={}; paths={}
    for name in ["TB-AWFNO", "AWFNO", "WFNO", "WFNO-PM"]:
        path=os.path.join(MODEL_DIR, f"N{n_train}_{name}_{tag}.pt"); paths[name]=path
        if os.path.exists(path) and not FORCE_RETRAIN:
            print("Loading cached", name, path); meta=load_model_checkpoint(path, models[name])
            fits[name]={"model": models[name], "best_val_l2": meta.get("best_val_l2", float("nan")),
                        "train_time_sec": meta.get("train_time_sec", 0.0), "history": meta.get("history")}
        else:
            fits[name]=_checkpoint_fit(models[name],obj['train_loader'],obj['val_loader'],
                                       preds[name],normer,name,
                                       Path(MODEL_DIR)/f'N{n_train}_{name}_{tag}.training.pt')
            save_model_checkpoint(path, fits[name]["model"], {k:v for k,v in fits[name].items() if k!="model"})
        if fits[name]["history"] is not None: plot_training_curves(fits[name]["history"], f"N{n_train}_{name}", tag)
    name="FNO"; path=os.path.join(MODEL_DIR, f"N{n_train}_{name}_{tag}.pt"); paths[name]=path
    if os.path.exists(path) and not FORCE_RETRAIN:
        print("Loading cached FNO", path); meta=load_model_checkpoint(path, models[name])
        fits[name]={"model": models[name], "best_val_l2": meta.get("best_val_l2", float("nan")),
                    "train_time_sec": meta.get("train_time_sec", 0.0), "history": meta.get("history")}
    else:
        fits[name]=_checkpoint_fit(models[name],obj['train_full_loader'],
                                  obj['val_full_loader'],preds[name],normer,'FNO',
                                  Path(MODEL_DIR)/f'N{n_train}_FNO_{tag}.training.pt',
                                  is_full=True)
        save_model_checkpoint(path, fits[name]["model"], {k:v for k,v in fits[name].items() if k!="model"})
    if fits[name]["history"] is not None: plot_training_curves(fits[name]["history"], f"N{n_train}_{name}", tag)
    return {"n_train": n_train, "tag": tag, "objects": obj, "models": models, "predict": preds, "fits": fits, "paths": paths}


def physically_localized_error_diagnostics(pred,true):
    """Final-time, 2π-periodic gradient and Fourier-band relative L2 diagnostics.
    Masks are determined from ground truth ONLY. Frequencies are full-complex
    FFTs; the energy split is physically interpretable on the 2π torus.
    """
    exact=np.asarray(true[:,:,:,-1,0],dtype=np.float64)
    estimate=np.asarray(pred[:,:,:,-1,0],dtype=np.float64)
    assert exact.shape==estimate.shape and exact.ndim==3
    e2=(estimate-exact)**2
    ux=(np.roll(exact,-1,axis=1)-np.roll(exact,1,axis=1))/(2*DX)
    uy=(np.roll(exact,-1,axis=2)-np.roll(exact,1,axis=2))/(2*DY)
    magnitude=np.hypot(ux,uy)
    cutoff=np.quantile(magnitude,.90,axis=(1,2),keepdims=True)
    sharp=magnitude>=cutoff
    sharp_relative=np.sqrt(
        np.sum(e2*sharp,axis=(1,2))/
        np.maximum(np.sum(exact**2*sharp,axis=(1,2)),1e-25)
    )
    kx=np.fft.fftfreq(exact.shape[1])*exact.shape[1]
    ky=np.fft.fftfreq(exact.shape[2])*exact.shape[2]
    shell=np.maximum(np.abs(kx[:,None]),np.abs(ky[None,:]))
    fft_error=np.fft.fft2(estimate-exact,axes=(1,2))
    fft_true=np.fft.fft2(exact,axes=(1,2))
    result={'sharp_gradient_final_relative_l2_mean':float(sharp_relative.mean())}
    for lo,hi in ((0,16),(16,24),(24,32),(32,43)):
        mask=(shell>=lo)&(shell<hi)
        values=np.sqrt(
          np.sum(np.abs(fft_error[:,mask])**2,axis=1)/
          np.maximum(np.sum(np.abs(fft_true[:,mask])**2,axis=1),1e-25)
        )
        result[f'fourier_band_{lo}_{hi-1}_final_relative_l2_mean']=float(values.mean())
    return result


def evaluate(bundle, S_test, test_idx):
    tag=bundle["tag"]; n_train=bundle["n_train"]
    print("\n"+"="*80); print(f"EVALUATION N_train={n_train} on common test set"); print("="*80)
    R={};timings={};peaks={}
    def timed(name,fn):
        if torch.cuda.is_available():
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        t0=time.perf_counter()
        result=fn()
        if torch.cuda.is_available():torch.cuda.synchronize()
        timings[name]=time.perf_counter()-t0
        peaks[name]=(torch.cuda.max_memory_allocated()/1024**2
                     if torch.cuda.is_available() else float('nan'))
        print(f'[INFERENCE] {name}: 70 trajectories in {timings[name]:.3f}s; '
              f'{peaks[name]:.1f} GPU MiB peak',flush=True)
        return result
    R["TB-AWFNO"] = timed("TB-AWFNO",lambda:rollout_tb(bundle["fits"]["TB-AWFNO"]["model"], S_test, bundle["predict"]["TB-AWFNO"]))
    R["AWFNO"] = timed("AWFNO",lambda:rollout_direct(bundle["fits"]["AWFNO"]["model"], S_test, bundle["predict"]["AWFNO"], "AWFNO"))
    R["WFNO"] = timed("WFNO",lambda:rollout_direct(bundle["fits"]["WFNO"]["model"], S_test, bundle["predict"]["WFNO"], "WFNO"))
    R["WFNO-PM"] = timed("WFNO-PM",lambda:rollout_direct(bundle["fits"]["WFNO-PM"]["model"], S_test, bundle["predict"]["WFNO-PM"], "WFNO-PM"))
    R["FNO"] = timed("FNO",lambda:predict_full(bundle["fits"]["FNO"]["model"], S_test, bundle["predict"]["FNO"]))
    R["Persistence"] = persistence(S_test)
    stats={name: rollout_l2_metrics(arr, S_test) for name, arr in R.items()}
    diagnostics={name:physically_localized_error_diagnostics(array,S_test)
                 for name,array in R.items() if name!='Persistence'}

    print("\n================ ABSOLUTE PHYSICAL VORTICITY L2 TEST RESULTS ================")
    for name in ["TB-AWFNO","AWFNO","WFNO","WFNO-PM","FNO","Persistence"]:
        st=stats[name]
        tt="" if name=="Persistence" else f" | train time {bundle['fits'][name]['train_time_sec']:.2f}s"
        print(f"{name:12s} | L2(full) mean {st['full_l2_mean']:.6e} | std {st['full_l2_std']:.6e} | median {st['full_l2_median']:.6e} | L2(final) mean {st['final_l2_mean']:.6e}{tt}")
    winner=min(stats, key=lambda k: stats[k]["full_l2_mean"])
    print("Winner:", winner)
    plot_l2_per_time(stats, tag)
    plot_fields(S_test, R, tag)
    if SAVE_ROLLOUT_ARRAYS:
        np.savez_compressed(os.path.join(RESULTS_DIR, f"N{n_train}_rollouts_{tag}.npz"), S_true=S_test, **{k.replace("-","_"):v for k,v in R.items()})
    row={"N_train": n_train, "winner": winner, "pod_energy_ratio_train_only": float(bundle["objects"]["energy"]), "base_tag": tag}
    for name in ["TB-AWFNO","AWFNO","WFNO","WFNO-PM","FNO","Persistence"]:
        key=name.replace("-","_")
        row[f"{key}_full_l2_mean"]=stats[name]["full_l2_mean"]
        row[f"{key}_final_l2_mean"]=stats[name]["final_l2_mean"]
        if name != "Persistence":
            row[f"{key}_best_val_l2"]=float(bundle["fits"][name]["best_val_l2"])
            row[f"{key}_train_time_sec"]=float(bundle["fits"][name]["train_time_sec"])
            row[f"{key}_inference_test70_sec"]=float(timings[name])
            row[f"{key}_inference_per_trajectory_sec"]=float(timings[name]/len(S_test))
            row[f"{key}_peak_inference_allocated_gpu_MiB"]=float(peaks[name])
            row[f"{key}_params"]=int(count_params(bundle['models'][name]))
            row[f"{key}_real_scalar_dof"]=int(scalar_parameters(bundle['models'][name]))
            row[f"{key}_full_l2_std"]=float(stats[name]['full_l2_std'])
            row[f"{key}_full_l2_median"]=float(stats[name]['full_l2_median'])
            row[f"{key}_final_l2_std"]=float(stats[name]['final_l2_std'])
            row[f"{key}_final_l2_median"]=float(stats[name]['final_l2_median'])
            for diagkey,diagvalue in diagnostics[name].items():
                row[f'{key}_{diagkey}']=float(diagvalue)
    with open(os.path.join(RESULTS_DIR, f"N{n_train}_summary_{tag}.json"), "w") as f:
        json.dump({"row": row, "grid": {"Nx": Nx, "Ny": Ny, "Nt": Nt, "C": C_DYN}}, f, indent=2)
    del R
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return row


def save_results(rows):
    rows=sorted(rows,key=lambda x:x['N_train'])
    # Original Boussinesq-style wide sample-size table.
    widepath=Path(RESULTS_DIR)/'sample_size_sweep_kolmogorov_absolute_L2.csv'
    columns=list(rows[0].keys())
    with widepath.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)
    (Path(RESULTS_DIR)/'sample_size_sweep_kolmogorov_absolute_L2.json').write_text(
        json.dumps(rows,indent=2))
    names=['TB-AWFNO','AWFNO','WFNO','WFNO-PM','FNO']
    flattened=[]
    for r in rows:
        for name in names:
            prefix=name.replace('-','_')
            flattened.append({
                'model':name,'N_train':r['N_train'],
                'protocol':'direct_full_100' if name=='FNO' else f'window_K{WINDOW_K}',
                'full_l2_mean':r[f'{prefix}_full_l2_mean'],
                'full_l2_std':r[f'{prefix}_full_l2_std'],
                'full_l2_median':r[f'{prefix}_full_l2_median'],
                'final_l2_mean':r[f'{prefix}_final_l2_mean'],
                'final_l2_std':r[f'{prefix}_final_l2_std'],
                'final_l2_median':r[f'{prefix}_final_l2_median'],
                'params':r[f'{prefix}_params'],
                'real_scalar_dof':r[f'{prefix}_real_scalar_dof'],
                'training_seconds':r[f'{prefix}_train_time_sec'],
                'full_test_set_inference_seconds':r[f'{prefix}_inference_test70_sec'],
                'inference_seconds_per_trajectory':r[f'{prefix}_inference_per_trajectory_sec'],
                'peak_inference_allocated_gpu_MiB':
                    r[f'{prefix}_peak_inference_allocated_gpu_MiB'],
                'sharp_gradient_final_relative_l2_mean':
                    r[f'{prefix}_sharp_gradient_final_relative_l2_mean'],
                'fourier_band_0_15_final_relative_l2_mean':
                    r[f'{prefix}_fourier_band_0_15_final_relative_l2_mean'],
                'fourier_band_16_23_final_relative_l2_mean':
                    r[f'{prefix}_fourier_band_16_23_final_relative_l2_mean'],
                'fourier_band_24_31_final_relative_l2_mean':
                    r[f'{prefix}_fourier_band_24_31_final_relative_l2_mean'],
                'fourier_band_32_42_final_relative_l2_mean':
                    r[f'{prefix}_fourier_band_32_42_final_relative_l2_mean']})
    with (Path(RESULTS_DIR)/'results_long.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=flattened[0].keys())
        w.writeheader();w.writerows(flattened)
    for metric in ['full_l2_mean','final_l2_mean']:
        fig,ax=plt.subplots(figsize=(9,5.5))
        for name in names:
            subset=[r for r in flattened if r['model']==name]
            ax.plot([r['N_train'] for r in subset],[r[metric] for r in subset],
                    marker='o',label=name)
        ax.set_xlabel('Number of training trajectories')
        ax.set_ylabel('Absolute physical L2')
        ax.set_title('Kolmogorov Navier–Stokes: '+metric+' vs training size')
        ax.legend();ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(Path(PLOT_DIR)/f'all_models_{metric}_vs_n.png',dpi=160)
        plt.close(fig)
    print('[SAVED] Comparable sample-size tables / plots:',widepath,
          Path(RESULTS_DIR)/'results_long.csv',flush=True)
    for n in sorted(set(r['N_train'] for r in flattened)):
        vals=[r for r in flattened if r['N_train']==n]
        print(f'[TABLE N={n}] '+ ' | '.join(
            f'{v["model"]}: full={v["full_l2_mean"]:.4f}, '
            f'final={v["final_l2_mean"]:.4f}' for v in vals),flush=True)


# ============================================================
# JCP REVIEWER CONTROLLED ABLATIONS — INDEPENDENT VALIDATION ONLY
# ============================================================

def controlled_matrix():
    """One-factor changes about fixed LGL/K20/M16/D4/r2/N100/seed1234."""
    base={'order':'LGL','kind':'AWFNO','k':20,'modes':16,
          'radius':2,'depth':4,'tb_rank':12,'seed':1234,'ntrain':100}
    jobs={}

    def add(family,label,**kwargs):
        q=dict(base,**kwargs)
        key=(q['kind'],q['order'],q['k'],q['modes'],q['radius'],
             q['depth'],q['tb_rank'],q['seed'],q['ntrain'])
        if key in jobs:
            if family not in jobs[key]['families']:jobs[key]['families'].append(family)
        else:
            jobs[key]=dict(q,label=label,families=[family])

    for order in ('LGL','GLL','LLG'):
        add('layer_ordering','ordering_'+order,order=order)
    for order in ('LLL','GGG'):
        add('operator_components','component_'+order,order=order)
    for k in (2,5,10,20):
        add('window_K',f'K_{k}',k=k)
    for m in (8,16,24,32):
        add('Fourier_modes',f'M_{m}',modes=m)
    for radius in (1,2,3):
        add('local_radius',f'r_{radius}',radius=radius)
    for depth in (2,4,6):
        add('depth',f'D_{depth}',depth=depth)
    for rank in (6,12,18):
        add('temporal_basis_rank',f'TB_rank_{rank}',kind='TB-AWFNO',tb_rank=rank)
    for seed in (1235,1236):
        for order in ('LGL','GLL'):
            add('order_repeated_seeds',f'{order}_seed_{seed}',order=order,seed=seed)
    add('matched_width','WFNO_PM_K20',kind='WFNO-PM')
    add('standard_wfno','WFNO_K20',kind='WFNO')
    return list(jobs.values())


def make_ordered_block(order):
    class OrderedBlock(nn.Module):
        def __init__(self,width):
            super().__init__()
            ops=[]
            for part in order:
                if part=='L':
                    op=nn.Conv2d(width,width,2*LOCAL_RADIUS+1,
                                 padding=LOCAL_RADIUS,padding_mode='circular')
                else:
                    op=SpectralConv2d(width,width,MODES_X,MODES_Y)
                ops.append(op)
            self.ops=nn.ModuleList(ops)
        def forward(self,z):
            for op in self.ops:z=z+F.gelu(op(z))
            return z
    return OrderedBlock


def run_controlled(train_pool,val):
    """Validate without touching test targets. Never import the previous 22-run outputs."""
    global WINDOW_K,WINDOW_STRIDE,MODES_X,MODES_Y,LOCAL_RADIUS,N_LAYERS
    global N_TIME_MODES,SEED,ExactLKLBlock
    import csv
    root=Path(CACHE_ROOT)/'controlled_corrected_v1'
    root.mkdir(exist_ok=True,parents=True)
    matrix=controlled_matrix()
    plan=root/'declared_matrix.json'
    if plan.is_file():
        if json.loads(plan.read_text())!=matrix:
            raise ValueError('Ablation matrix changed; choose a NEW output directory.')
    else:plan.write_text(json.dumps(matrix,indent=2))
    block_orig=ExactLKLBlock
    original=(WINDOW_K,WINDOW_STRIDE,MODES_X,MODES_Y,
              LOCAL_RADIUS,N_LAYERS,N_TIME_MODES,SEED)
    summary=root/'results.json'
    rows=json.loads(summary.read_text()) if summary.exists() else []
    already={row['configuration_hash'] for row in rows}
    for i,config in enumerate(matrix,1):
        WINDOW_K=config['k'];WINDOW_STRIDE=WINDOW_K
        MODES_X=MODES_Y=config['modes']
        LOCAL_RADIUS=config['radius'];N_LAYERS=config['depth']
        N_TIME_MODES=config['tb_rank'];SEED=config['seed']
        ExactLKLBlock=make_ordered_block(config['order'])
        random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
        signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:14]
        label=f"{config['label']}_{signature}"
        if signature in already:
            print(f'[REVIEWER CACHE] {i}/{len(matrix)} {label}',flush=True)
            continue
        run_dir=root/label;run_dir.mkdir(exist_ok=True)
        print(f'[REVIEWER START] {i}/{len(matrix)} {label}',flush=True)
        training=train_pool[:config['ntrain']]
        obj=build_objects(config['ntrain'],train_pool,val)
        kind=config['kind']
        if kind=='AWFNO':
            model=DirectWindowModel(C_DYN,WINDOW_K,'AW').to(device)
            predict=make_direct_predict(model,obj['normer'],obj['normer'],False)
        elif kind=='TB-AWFNO':
            model=TBAWFNO(C_DYN,N_TIME_MODES).to(device)
            predict=make_tb_predict(model,obj['normer'],obj['basis_t'])
        elif kind=='WFNO':
            model=DirectWindowModel(C_DYN,WINDOW_K,'FNO').to(device)
            predict=make_direct_predict(model,obj['normer'],obj['normer'],False)
        elif kind=='WFNO-PM':
            ref=DirectWindowModel(C_DYN,WINDOW_K,'AW')
            _,width,_=matched_width(scalar_parameters(ref),WINDOW_K)
            del ref
            model=MatchedWFNO(C_DYN,WINDOW_K,width).to(device)
            predict=make_direct_predict(model,obj['normer'],obj['normer'],False)
        else:raise RuntimeError(kind)
        params=count_params(model)
        scalar_dof=scalar_parameters(model)
        ckpt=run_dir/'checkpoint_epoch.pt'
        fit=_checkpoint_fit(model,obj['train_loader'],obj['val_loader'],
                            predict,obj['normer'],kind,ckpt)
        file=run_dir/'best_model.pt'
        save_model_checkpoint(file,fit['model'],{'config':config,
                                               'training_seconds':fit['train_time_sec']})
        if torch.cuda.is_available():
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        if kind=='TB-AWFNO':pred=rollout_tb(model,val,predict)
        else:pred=rollout_direct(model,val,predict,kind)
        if torch.cuda.is_available():torch.cuda.synchronize()
        inference=time.perf_counter()-started
        peak=(float(torch.cuda.max_memory_allocated()/1024**2)
              if torch.cuda.is_available() else float('nan'))
        numbers=rollout_l2_metrics(pred,val)
        special=physically_localized_error_diagnostics(pred,val)
        persist=rollout_l2_metrics(persistence(val),val)
        fig,ax=plt.subplots(figsize=(9,5.2))
        ax.plot(t_grid,numbers['mean_t'],lw=2,label=kind+' full-rollout error')
        ax.plot(t_grid,persist['mean_t'],'--',label='persistence baseline')
        ax.set_xlabel('Time since forecast start')
        ax.set_ylabel('Mean spatial absolute physical L2 error')
        ax.set_title('Validation error vs time — '+config['label'])
        ax.grid(alpha=.2);ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir/'validation_error_vs_time.png',dpi=150)
        plt.close(fig)

        np.savez_compressed(run_dir/'validation_error_vs_time.npz',
                            times=np.asarray(t_grid),mean=numbers['mean_t'],
                            median=numbers['median_t'],std=numbers['std_t'])
        np.savez_compressed(run_dir/'validation_preview.npz',
                            truth=val[[0,15]],prediction=pred[[0,15]])
        # Standalone comparison panels are saved for two validation trajectories.
        for case in (0,15):
            indices=[0,25,50,75,100]
            true=val[case,:,:,:,0]
            predicted=pred[case,:,:,:,0]
            vmax=max(float(np.abs(true).max()),1.e-12)
            errcap=max(float(np.percentile(np.abs(predicted-true),99)),1.e-12)
            fig,axes=plt.subplots(3,5,figsize=(15,8),constrained_layout=True)
            for j,k in enumerate(indices):
                axes[0,j].imshow(true[:,:,k].T,origin='lower',cmap='RdBu_r',
                                 vmin=-vmax,vmax=vmax)
                axes[1,j].imshow(predicted[:,:,k].T,origin='lower',cmap='RdBu_r',
                                 vmin=-vmax,vmax=vmax)
                axes[2,j].imshow(np.abs(predicted[:,:,k]-true[:,:,k]).T,
                                 origin='lower',cmap='magma',vmin=0,vmax=errcap)
                axes[0,j].set_title(f't={float(t_grid[k]):.4f}')
                for row in axes[:,j]:row.set_xticks([]);row.set_yticks([])
            for ax,t in zip(axes[:,0],['True','Prediction','Absolute error']):
                ax.set_ylabel(t)
            fig.savefig(run_dir/f'validation_vorticity_case_{case}.png',dpi=130)
            plt.close(fig)
        row={**config,'configuration_hash':signature,'families':config['families'],
             'model':kind,'params':params,'real_scalar_dof':scalar_dof,
             'best_val_window_l2':float(fit['best_val_l2']),
             'full_l2_mean':float(numbers['full_l2_mean']),
             'full_l2_std':float(numbers['full_l2_std']),
             'final_l2_mean':float(numbers['final_l2_mean']),
             'final_l2_std':float(numbers['final_l2_std']),
             'training_seconds':float(fit['train_time_sec']),
             'validation_inference_seconds':float(inference),
             'peak_validation_gpu_mib':float(peak),
             **special,
             'test_evaluated':False}
        (run_dir/'result.json').write_text(json.dumps(row,indent=2))
        rows.append(row);already.add(signature)
        summary.write_text(json.dumps(rows,indent=2))
        fields=['label','families','model','order','k','modes','radius','depth',
                'tb_rank','seed','ntrain','params','real_scalar_dof',
                'best_val_window_l2','full_l2_mean','full_l2_std',
                'final_l2_mean','final_l2_std',
                'training_seconds','validation_inference_seconds',
                'sharp_gradient_final_relative_l2_mean',
                'fourier_band_0_15_final_relative_l2_mean',
                'fourier_band_16_23_final_relative_l2_mean',
                'fourier_band_24_31_final_relative_l2_mean',
                'fourier_band_32_42_final_relative_l2_mean',
                'peak_validation_gpu_mib','configuration_hash']
        with (root/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            writer.writeheader()
            for r in rows:
                r0=dict(r);r0['families']=';'.join(r['families'])
                writer.writerow(r0)
        fig,ax=plt.subplots(figsize=(15,5))
        labels=[x['label'] for x in rows]
        ax.bar(range(len(rows)),[x['full_l2_mean'] for x in rows])
        ax.set_xticks(range(len(rows)),labels,rotation=80,fontsize=8)
        ax.set_ylabel('Validation absolute physical full L2')
        ax.set_title('Controlled reviewer ablations — corrected Navier–Stokes')
        ax.grid(axis='y',alpha=.25);fig.tight_layout()
        fig.savefig(root/'ALL_corrected_ablation_results.png',dpi=150)
        plt.close(fig)
        print(f'[REVIEWER COMPLETE] {i}/{len(matrix)} {label} '
              f'full={row["full_l2_mean"]:.5f} final={row["final_l2_mean"]:.5f}',
              flush=True)
        del model,fit,pred,obj
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    (WINDOW_K,WINDOW_STRIDE,MODES_X,MODES_Y,LOCAL_RADIUS,
     N_LAYERS,N_TIME_MODES,SEED)=original
    ExactLKLBlock=block_orig
    print('[REVIEWER SUITE STATUS]',len(rows),'of',len(matrix),
          'validation-only configurations complete',flush=True)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\nProtocol")
    print("--------")
    print("Dataset:", DATA_PATH)
    print("Models: TB-AWFNO, AWFNO, WFNO, WFNO-PM, neuraloperator FNO, Persistence")
    print("SPATIAL_STRIDE:", SPATIAL_STRIDE, "| TIME_SUBSAMPLE:", TIME_SUBSAMPLE)
    print("Absolute physical vorticity L2 only; no relative denominator.")

    print('[DATA] Loading verified 101-frame dataset; the original 22-run outputs are NOT used.',flush=True)
    S_train_pool, S_val, S_test, train_idx, val_idx, test_idx, x, y, t = load_or_create_split()

    set_geometry(S_train_pool.shape, x, y, t)

    if os.environ.get('NS_SMOKE_ONLY','0')=='1':
        print('[SMOKE] Five architectures; synthetic fields, no model training.',flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        xfake=torch.randn(1,Nx,Ny,1,device=device)
        tb=TBAWFNO(C_DYN,N_TIME_MODES).to(device)
        aw=DirectWindowModel(C_DYN,WINDOW_K,'AW').to(device)
        wf=DirectWindowModel(C_DYN,WINDOW_K,'FNO').to(device)
        delta,width,m_count=matched_width(scalar_parameters(aw),WINDOW_K)
        pm=MatchedWFNO(C_DYN,WINDOW_K,width).to(device)
        full=FNOFull(C_DYN,Nt-1).to(device)
        for name,model,n in [('TB-AWFNO',tb,N_TIME_MODES),
                             ('AWFNO',aw,WINDOW_K),('WFNO',wf,WINDOW_K),
                             ('WFNO-PM',pm,WINDOW_K),('FNO',full,Nt-1)]:
            output=model(xfake)
            want=(1,Nx,Ny,1,N_TIME_MODES) if name=='TB-AWFNO' else (1,Nx,Ny,n,1)
            assert tuple(output.shape)==want,(name,tuple(output.shape),want)
            assert torch.isfinite(output).all()
            output.square().mean().backward()
            assert any(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.requires_grad)
            print('[SMOKE PASS]',name,'output',tuple(output.shape),
                  'params',count_params(model),'real DOF',scalar_parameters(model),
                  flush=True)
            model.zero_grad(set_to_none=True)
        print('[SMOKE PASS] Five models; matched WFNO differs from AWFNO by',
              f'{100*delta/scalar_parameters(aw):.2f}% in real DOF',flush=True)
        raise SystemExit(0)

    if os.environ.get('NS_CONTROLLED_ONLY','0')=='1':
        print('[MODE] Corrected independent controlled ablations; test metrics SEALED',
              flush=True)
        run_controlled(S_train_pool,S_val)
        raise SystemExit(0)

    print("\nLoaded split arrays")
    print_array_info("S_train_pool", S_train_pool)
    print_array_info("S_val", S_val)
    print_array_info("S_test", S_test)

    for c in range(S_train_pool.shape[-1]):
        mid=min(5,S_train_pool.shape[-2]-1)
        print(f"ch {c}: t0 {S_train_pool[0,:,:,0,c].min():+.3e}/{S_train_pool[0,:,:,0,c].max():+.3e} | "
              f"t{mid} {S_train_pool[0,:,:,mid,c].min():+.3e}/{S_train_pool[0,:,:,mid,c].max():+.3e} | "
              f"final {S_train_pool[0,:,:,-1,c].min():+.3e}/{S_train_pool[0,:,:,-1,c].max():+.3e}")

    if not np.all(np.isfinite(S_train_pool)) or not np.all(np.isfinite(S_val)) or not np.all(np.isfinite(S_test)):
        raise ValueError("NaN/Inf found in loaded arrays.")

    if RUN_EXACT_HASH_OVERLAP_CHECK:
        check_exact_overlap(S_train_pool, S_val, "train_pool", "validation")
        check_exact_overlap(S_train_pool, S_test, "train_pool", "common_test")
        check_exact_overlap(S_val, S_test, "validation", "common_test")

    if RUN_DUPLICATE_REPORT_INSIDE_SPLITS:
        report_duplicates(S_train_pool, "train_pool")
        report_duplicates(S_val, "validation")
        report_duplicates(S_test, "common_test")

    # Recover completed sample-size rows after a Colab disconnection.
    summary=Path(RESULTS_DIR)/'sample_size_sweep_kolmogorov_absolute_L2.json'
    rows=json.loads(summary.read_text()) if summary.exists() else []
    completed={int(q['N_train']) for q in rows}
    print('[RESUME] Completed training sizes:',sorted(completed),flush=True)
    # Report each N immediately: no accumulation of twenty models in GPU memory.
    for n_train in SAMPLE_SIZES:
        if n_train in completed and not FORCE_RETRAIN:
            print(f'[CACHE] N={n_train} already fully evaluated; skipping',flush=True)
            continue
        b=train_models(n_train,S_train_pool,S_val)
        rows.append(evaluate(b,S_test,test_idx))
        save_results(rows)
        del b
        print(f'[N COMPLETE] {n_train}: saved metrics and plots',flush=True)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\nDone.")
