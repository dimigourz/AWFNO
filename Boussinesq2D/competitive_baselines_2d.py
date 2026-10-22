#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D annular Boussinesq: independent F-FNO, CNO-style, U-Net, U-NO-style baselines.

All networks learn exactly the same map as the original windowed AWFNO:
  one scalar state c(r,theta,t) -> next K=30 scalar states.
Test prediction is fully autoregressive from t=0 (no test teacher forcing).

Imported functions from annulus2d_jcp_experiments.py are the sole source of
dataset splitting, train-only normalizers and physical annulus L2 quadrature.

IMPORTANT: The F-FNO follows factorized axis-wise spectral convolutions; the
CNO and U-NO versions are explicit research adaptations, NOT official
authors' implementations. See README for limitations and references.
"""
import os
import gc
import csv
import json
import time
import random
import hashlib
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import annulus2d_jcp_experiments as base

NAMES = ("FFNO", "CNO", "UNet", "UNO")
BASELINES = tuple(s.strip() for s in os.environ.get("BASELINES", ",".join(NAMES)).split(",") if s.strip())
SIZES = tuple(int(s) for s in os.environ.get("BASELINE_SAMPLE_SIZES", "25,50,100,200,400").split(",") if s.strip())
SEEDS = tuple(int(s) for s in os.environ.get("BASELINE_SEEDS", "1234").split(",") if s.strip())
EVAL_BATCH = int(os.environ.get("BASELINE_EVAL_BATCH", "8"))
WIDTH = int(os.environ.get("BASELINE_WIDTH", "32"))
SAVE_ROLLOUTS = os.environ.get("SAVE_BASELINE_ROLLOUTS", "1") == "1"
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"
RESULT_ROOT = Path(os.environ.get("BASELINE_OUTPUT", str(base.OUTPUT_ROOT / "competitive_2d_K30")))
RUN_VERSION = "annulus2d_competitive_v1"
assert all(name in NAMES for name in BASELINES), BASELINES
assert all(n in base.ALL_SAMPLE_SIZES for n in SIZES), SIZES
assert min(SEEDS) >= 0 and EVAL_BATCH >= 1
assert base.WINDOW_K == 30 and base.WINDOW_STRIDE == 30, "Use same K=30, stride=30 as original AWFNO"
assert base.TIME_SUBSAMPLE == 1 and base.SPLIT_MODE == "contiguous", "Original annulus experiment uses full time and contiguous split."
assert base.N_TRAIN_POOL == 400 and base.N_VAL == 30 and base.N_TEST == 70
assert base.FINAL_LOSS_WEIGHT == 0., "Comparison uses original MSE-only objective."
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
PLOT_ROOT = RESULT_ROOT / "plots"
PLOT_ROOT.mkdir(parents=True, exist_ok=True)
CKPT_ROOT = RESULT_ROOT / "checkpoints"
CKPT_ROOT.mkdir(parents=True, exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def polar_features(r, theta):
    rr = torch.as_tensor(r, dtype=torch.float32)
    tt = torch.as_tensor(theta, dtype=torch.float32)
    rr = (rr - rr.min()) / (rr.max() - rr.min() + 1.e-12)
    R, TH = torch.meshgrid(rr, tt, indexing="ij")
    return torch.stack((R, torch.sin(TH), torch.cos(TH)), dim=0).unsqueeze(0)


class PolarInput(nn.Module):
    def __init__(self, r, theta):
        super().__init__()
        self.register_buffer("grid", polar_features(r, theta), persistent=False)
    def forward(self, u):
        return torch.cat((u.unsqueeze(1), self.grid.expand(u.shape[0], -1, -1, -1)), dim=1)


def polar_pad(x, radius):
    if radius < 1:
        return x
    x = F.pad(x, (radius, radius, 0, 0), mode="circular")
    return F.pad(x, (0, 0, radius, radius), mode="replicate")


class PolarConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=3):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(cin, cout, kernel_size, padding=0)
    def forward(self, x):
        return self.conv(polar_pad(x, self.pad))


class AxisFactorizedSpectral(nn.Module):
    """Independent 1D Fourier multipliers along radius and azimuth (F-FNO)."""
    def __init__(self, channels, modes_r, modes_t):
        super().__init__()
        self.mr, self.mt = int(modes_r), int(modes_t)
        scale = 1. / (channels ** 2) ** .5
        self.weight_r = nn.Parameter(scale * torch.randn(channels, channels, self.mr, dtype=torch.cfloat))
        self.weight_t = nn.Parameter(scale * torch.randn(channels, channels, self.mt, dtype=torch.cfloat))
    def forward(self, x):
        B, C, nr, nt = x.shape
        fr = torch.fft.rfft(x, dim=-2, norm="ortho")
        nr_modes = min(self.mr, nr // 2 + 1)
        yr = torch.zeros((B, C, nr // 2 + 1, nt), device=x.device, dtype=fr.dtype)
        yr[:, :, :nr_modes] = torch.einsum(
            "birt,ior->bort", fr[:, :, :nr_modes], self.weight_r[:, :, :nr_modes])
        yr = torch.fft.irfft(yr, n=nr, dim=-2, norm="ortho")

        ft = torch.fft.rfft(x, dim=-1, norm="ortho")
        nt_modes = min(self.mt, nt // 2 + 1)
        yt = torch.zeros((B, C, nr, nt // 2 + 1), device=x.device, dtype=ft.dtype)
        yt[:, :, :, :nt_modes] = torch.einsum(
            "birm,iom->borm", ft[:, :, :, :nt_modes], self.weight_t[:, :, :nt_modes])
        yt = torch.fft.irfft(yt, n=nt, dim=-1, norm="ortho")
        return yr + yt


class FFNOBlock(nn.Module):
    def __init__(self, width, mr, mt):
        super().__init__()
        self.spectral = AxisFactorizedSpectral(width, mr, mt)
        self.pointwise = nn.Conv2d(width, width, 1)
        self.mlp = nn.Sequential(nn.Conv2d(width, width * 2, 1), nn.GELU(),
                                 nn.Conv2d(width * 2, width, 1))
    def forward(self, x):
        return x + self.mlp(F.gelu(self.spectral(x) + self.pointwise(x)))


class FFNO2D(nn.Module):
    """Fourier-factorized, residual 2D windowed operator (adaptation of Tran et al.)."""
    def __init__(self, k, r, theta, width=WIDTH):
        super().__init__()
        self.coords = PolarInput(r, theta)
        self.lift = nn.Conv2d(4, width, 1)
        self.blocks = nn.Sequential(*[FFNOBlock(width, 16, 24) for _ in range(6)])
        self.head = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, k, 1))
    def forward(self, u):
        x = self.blocks(self.lift(self.coords(u)))
        return self.head(x).permute(0, 2, 3, 1).contiguous()


class UNetBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.seq = nn.Sequential(PolarConv(ci, co), nn.GELU(), PolarConv(co, co), nn.GELU())
    def forward(self, x):
        return self.seq(x)


class UNet2D(nn.Module):
    """Conventional U-Net with pooling, learned convolutions and skip connections."""
    def __init__(self, k, r, theta, width=WIDTH):
        super().__init__()
        self.coords = PolarInput(r, theta)
        self.e0 = UNetBlock(4, width)
        self.e1 = UNetBlock(width, 2 * width)
        self.e2 = UNetBlock(2 * width, 4 * width)
        self.bottleneck = UNetBlock(4 * width, 4 * width)
        self.d1 = UNetBlock(6 * width, 2 * width)
        self.d0 = UNetBlock(3 * width, width)
        self.head = nn.Conv2d(width, k, 1)
    def forward(self, u):
        e0 = self.e0(self.coords(u))
        e1 = self.e1(F.max_pool2d(e0, 2))
        e2 = self.e2(F.max_pool2d(e1, 2))
        z = self.bottleneck(e2)
        z = self.d1(torch.cat((F.interpolate(z, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), dim=1))
        z = self.d0(torch.cat((F.interpolate(z, size=e0.shape[-2:], mode="bilinear", align_corners=False), e0), dim=1))
        return self.head(z).permute(0, 2, 3, 1).contiguous()


class LowPass2D(nn.Module):
    """Fixed binomial low-pass filter with periodic theta and replicated radial boundary."""
    def __init__(self):
        super().__init__()
        v = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32) / 16.
        self.register_buffer("kernel", torch.outer(v, v).reshape(1, 1, 5, 5))
    def forward(self, x):
        return F.conv2d(polar_pad(x, 2), self.kernel.to(dtype=x.dtype).expand(x.shape[1], 1, 5, 5),
                        groups=x.shape[1])


class FilteredActivation(nn.Module):
    """Oversample -> nonlinear activation -> low-pass -> resample."""
    def __init__(self):
        super().__init__()
        self.lp = LowPass2D()
    def forward(self, x):
        original = x.shape[-2:]
        z = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        z = self.lp(F.gelu(z))
        return F.interpolate(z, size=original, mode="bilinear", align_corners=False, antialias=True)


class CNOBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.first = PolarConv(ci, co)
        self.activation = FilteredActivation()
        self.second = PolarConv(co, co)
    def forward(self, x):
        return self.activation(self.second(self.activation(self.first(x))))


class CNODown(nn.Module):
    def __init__(self):
        super().__init__()
        self.lp = LowPass2D()
    def forward(self, x):
        return self.lp(x)[:, :, ::2, ::2]


class CNO2D(nn.Module):
    """CNO-style filtered multiresolution CNN (research adaptation; not official CNO)."""
    def __init__(self, k, r, theta, width=WIDTH):
        super().__init__()
        self.coords = PolarInput(r, theta)
        self.e0 = CNOBlock(4, width)
        self.down = CNODown()
        self.e1 = CNOBlock(width, 2 * width)
        self.e2 = CNOBlock(2 * width, 4 * width)
        self.bottleneck = CNOBlock(4 * width, 4 * width)
        self.d1 = CNOBlock(6 * width, 2 * width)
        self.d0 = CNOBlock(3 * width, width)
        self.head = nn.Conv2d(width, k, 1)
    def forward(self, u):
        e0 = self.e0(self.coords(u))
        e1 = self.e1(self.down(e0))
        e2 = self.e2(self.down(e1))
        z = self.bottleneck(e2)
        z = self.d1(torch.cat((F.interpolate(z, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), 1))
        z = self.d0(torch.cat((F.interpolate(z, size=e0.shape[-2:], mode="bilinear", align_corners=False), e0), 1))
        return self.head(z).permute(0, 2, 3, 1).contiguous()


class UNOBlock(nn.Module):
    """Multi-resolution residual Fourier operator block."""
    def __init__(self, cin, cout, mr, mt):
        super().__init__()
        self.lift = nn.Conv2d(cin, cout, 1)
        self.spec = base.SpectralConv2d(cout, cout, mr, mt)
        self.pw = nn.Conv2d(cout, cout, 1)
    def forward(self, x):
        z = self.lift(x)
        return z + F.gelu(self.spec(z) + self.pw(z))


class UNO2D(nn.Module):
    """U-shaped Fourier neural operator with spectral encoder-decoder + skip connections."""
    def __init__(self, k, r, theta, width=WIDTH):
        super().__init__()
        self.coords = PolarInput(r, theta)
        self.entry = nn.Conv2d(4, width, 1)
        self.e0 = UNOBlock(width, width, 16, 24)
        self.e1 = UNOBlock(width, 2 * width, 12, 20)
        self.e2 = UNOBlock(2 * width, 4 * width, 6, 12)
        self.d1 = UNOBlock(6 * width, 2 * width, 12, 20)
        self.d0 = UNOBlock(3 * width, width, 16, 24)
        self.head = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, k, 1))
    def forward(self, u):
        x = self.entry(self.coords(u))
        e0 = self.e0(x)
        e1 = self.e1(F.interpolate(e0, scale_factor=.5, mode="bilinear", align_corners=False))
        e2 = self.e2(F.interpolate(e1, scale_factor=.5, mode="bilinear", align_corners=False))
        d1 = self.d1(torch.cat((F.interpolate(e2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1), 1))
        d0 = self.d0(torch.cat((F.interpolate(d1, size=e0.shape[-2:], mode="bilinear", align_corners=False), e0), 1))
        return self.head(d0).permute(0, 2, 3, 1).contiguous()


def make_model(name, r, theta):
    constructors = {"FFNO": FFNO2D, "CNO": CNO2D, "UNet": UNet2D, "UNO": UNO2D}
    return constructors[name](base.WINDOW_K, r, theta, WIDTH)


def param_stats(model):
    return (base.count_params(model),
            sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad))


def network_predict(model, u_start, u_normer, target_normer):
    return target_normer.decode(model(u_normer.encode(u_start)))


def fit_model(model, train_loader, val_loader, normalizers, name, seed, ntrain):
    u_normer, target_normer = normalizers
    opt = torch.optim.AdamW(model.parameters(), lr=base.LR, weight_decay=base.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=base.EPOCHS)
    hist = {"train_loss": [], "train_l2": [], "val_loss": [], "val_l2": [], "val_final_l2": []}
    best = float("inf")
    best_state = None
    t0 = time.perf_counter()
    for epoch in range(1, base.EPOCHS + 1):
        for stage, loader in (("train", train_loader), ("val", val_loader)):
            is_train = stage == "train"
            model.train(is_train)
            loss_sum = l2_sum = final_sum = count = 0.
            with torch.set_grad_enabled(is_train):
                for data in loader:
                    u = data["u_start"].to(base.device, non_blocking=True)
                    truth = data["U_future"].to(base.device, non_blocking=True)
                    if is_train:
                        opt.zero_grad(set_to_none=True)
                    pred = network_predict(model, u, u_normer, target_normer)
                    loss = F.mse_loss(pred, truth)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"{name} N={ntrain} seed={seed}: nonfinite {stage} loss at epoch {epoch}")
                    if is_train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        opt.step()
                    bs = len(u)
                    count += bs
                    loss_sum += float(loss.detach()) * bs
                    l2_sum += float(base.abs_l2_window_batch(pred.detach(), truth).mean()) * bs
                    if not is_train:
                        final_sum += float(base.abs_l2_field_batch(pred[:, :, :, -1], truth[:, :, :, -1]).mean()) * bs
            hist[f"{stage}_loss"].append(loss_sum / count)
            hist[f"{stage}_l2"].append(l2_sum / count)
            if not is_train:
                hist["val_final_l2"].append(final_sum / count)
                if l2_sum / count < best:
                    best = l2_sum / count
                    best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
        sched.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == base.EPOCHS:
            print(f"[{name} N={ntrain} seed={seed}] {epoch:03d}/{base.EPOCHS} | "
                  f"train={hist['train_l2'][-1]:.5e} | val={hist['val_l2'][-1]:.5e} "
                  f"| final={hist['val_final_l2'][-1]:.5e}", flush=True)
    if best_state is None:
        raise RuntimeError(f"{name}: no finite validation checkpoint")
    model.load_state_dict(best_state)
    return hist, best, time.perf_counter() - t0


@torch.no_grad()
def rollout(model, test_u0, normalizers, nt):
    """Per-batch recursive prediction; never uses subsequent true test states."""
    u_normer, target_normer = normalizers
    nr, nth = test_u0.shape[1:3]
    result = np.empty((len(test_u0), nr, nth, nt), dtype=np.float32)
    for start in range(0, len(test_u0), EVAL_BATCH):
        stop = min(start + EVAL_BATCH, len(test_u0))
        initial = test_u0[start:stop]
        result[start:stop, :, :, 0] = initial
        current = torch.from_numpy(initial).to(base.device)
        for t in range(0, nt - 1, base.WINDOW_K):
            pred = network_predict(model, current, u_normer, target_normer)
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f"Non-finite test rollout at t={t} for trajectories {start}:{stop}")
            kk = min(base.WINDOW_K, nt - 1 - t)
            result[start:stop, :, :, t + 1:t + kk + 1] = pred[..., :kk].cpu().numpy()
            current = pred[..., kk - 1].detach()
    return result


def plot_curves(history, name, dest):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogy(history["train_loss"], label="train MSE")
    ax.semilogy(history["val_loss"], label="validation MSE")
    ax.set(xlabel="epoch", ylabel="mean squared error", title=name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "loss_curves.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key in ("train_l2", "val_l2", "val_final_l2"):
        ax.semilogy(history[key], label=key)
    ax.set(xlabel="epoch", ylabel="absolute physical L2", title=name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "validation_l2_curves.png", dpi=140)
    plt.close(fig)


def plot_errors(metrics, t_grid, name, dest):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(t_grid, metrics["mean_t"], label="mean")
    ax.plot(t_grid, metrics["median_t"], label="median")
    ax.fill_between(t_grid, np.maximum(0, metrics["mean_t"] - metrics["std_t"]),
                    metrics["mean_t"] + metrics["std_t"], alpha=.15, label="±1 SD")
    ax.set(xlabel="time", ylabel="absolute spatial L2 error", title=name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "test_l2_vs_time.png", dpi=140)
    plt.close(fig)


def plot_annulus_cases(pred, truth, name, dest):
    # Use original annular Cartesian projection with identical color bars.
    test_full = base.abs_l2_full_traj_np(pred, truth)
    ids = list(dict.fromkeys([0, int(np.argmin(test_full)), int(np.argmax(test_full))]))
    times = [0, min(35, truth.shape[-1] - 1), min(60, truth.shape[-1] - 1), truth.shape[-1] - 1]
    for idx in ids:
        fig, axes = plt.subplots(len(times), 3, figsize=(11, 12))
        vmin = min(float(truth[idx].min()), float(pred[idx].min()))
        vmax = max(float(truth[idx].max()), float(pred[idx].max()))
        for row, it in enumerate(times):
            for col, arr in enumerate((truth[idx, :, :, it], pred[idx, :, :, it],
                                       np.abs(truth[idx, :, :, it] - pred[idx, :, :, it]))):
                x, y, z = base.polar_field_to_closed_annulus(arr)
                im = axes[row, col].pcolormesh(x, y, z, shading="auto",
                                              vmin=0 if col == 2 else vmin,
                                              vmax=None if col == 2 else vmax)
                axes[row, col].set_aspect("equal")
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                axes[row, col].set_title(("true", name, "absolute error")[col] + f" | t={it}")
                fig.colorbar(im, ax=axes[row, col], fraction=.046, pad=.02)
        fig.tight_layout()
        fig.savefig(dest / f"annulus_test_{idx:03d}.png", dpi=130)
        plt.close(fig)


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def write_table(rows):
    if not rows:
        return
    atomic_json(RESULT_ROOT / "results.json", rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temp = RESULT_ROOT / "results.csv.tmp"
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temp.replace(RESULT_ROOT / "results.csv")
    print(f"Saved {len(rows)} completed baseline comparisons -> {RESULT_ROOT / 'results.csv'}", flush=True)


def _baseline_plot(rows):
    """Show results alongside EXISTING (not retrained) AWFNO/FNO/WFNO outputs."""
    try:
        reference = next((base.CACHE_ROOT / "results").glob("sample_size_sweep_summary_run_*.json"))
        original = json.loads(reference.read_text())
    except (StopIteration, FileNotFoundError):
        original = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted({r["model"] for r in rows}):
        subset = sorted([r for r in rows if r["model"] == model and r["seed"] == SEEDS[0]],
                        key=lambda r: r["N_train"])
        if subset:
            ax.plot([q["N_train"] for q in subset],
                    [q["full_l2_mean"] for q in subset], marker="o", label=model)
    for model in ("AWFNO", "FNO", "WFNO"):
        if original:
            ax.plot([r["N_train"] for r in original],
                    [r[f"{model}_full_l2_mean"] for r in original], marker="x", ls="--",
                    label=f"{model} (original cached)")
    ax.set(xlabel="training trajectories", ylabel="mean absolute full-trajectory L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "all_models_full_l2_vs_n.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted({r["model"] for r in rows}):
        subset = sorted([r for r in rows if r["model"] == model and r["seed"] == SEEDS[0]],
                        key=lambda r: r["N_train"])
        if subset:
            ax.plot([q["N_train"] for q in subset], [q["final_l2_mean"] for q in subset],
                    marker="o", label=model)
    for model in ("AWFNO", "FNO", "WFNO"):
        if original:
            ax.plot([r["N_train"] for r in original],
                    [r[f"{model}_final_l2_mean"] for r in original], marker="x", ls="--",
                    label=f"{model} (original cached)")
    ax.set(xlabel="training trajectories", ylabel="mean absolute final-time L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_ROOT / "all_models_final_l2_vs_n.png", dpi=140)
    plt.close(fig)


def main():
    print(f"Competitive Boussinesq operators: {BASELINES}; N={SIZES}; seeds={SEEDS}", flush=True)
    print(f"Original evaluation: K={base.WINDOW_K}, Nr×Ntheta=64×128, N_test=70", flush=True)
    print(f"GPU: {base.device}, epochs={base.EPOCHS}, batch={base.BATCH_SIZE}", flush=True)
    original_sha = hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
    U, r, theta, t, sample_ids = base.load_annulus_dataset(base.DATA_PATH)
    base.set_geometry_from_arrays(U.shape, r, theta, t)
    pool, trids, Uval, vids, Utest, teids = base.make_train_val_test_splits(U)
    if base.RUN_EXACT_HASH_OVERLAP_CHECK:
        base.check_exact_hash_overlap(pool, Uval, "train", "val")
        base.check_exact_hash_overlap(pool, Utest, "train", "test")
        base.check_exact_hash_overlap(Uval, Utest, "val", "test")
    assert (base.Nr, base.Ntheta, base.Nt) == (64, 128, 101), "Expected original 64×128×101 annulus data"
    manifest = {
        "protocol": RUN_VERSION,
        "baseline_names": BASELINES,
        "sample_sizes": SIZES,
        "seeds": SEEDS,
        "dataset": str(base.DATA_PATH),
        "dataset_bytes": base.DATA_PATH.stat().st_size,
        "original_driver_sha256": original_sha,
        "training": dict(optimizer="AdamW", lr=base.LR, wd=base.WEIGHT_DECAY,
                         scheduler="CosineAnnealingLR", epochs=base.EPOCHS,
                         batch_size=base.BATCH_SIZE, loss="physical-output MSE",
                         clip_norm=1., selection="validation absolute window L2"),
        "geometry": dict(nr=64, ntheta=128, nt=101, K=30, stride=30, train_pool=400, val=30, test=70),
        "note": "Original authors' CNO and U-NO packages are NOT used; research adaptations are explicitly labelled."
    }
    atomic_json(RESULT_ROOT / "manifest.json", manifest)
    rows = []
    for ntrain in SIZES:
        print(f"\n{'=' * 76}\nN_train = {ntrain}\n{'=' * 76}", flush=True)
        for seed in SEEDS:
            seed_everything(seed)
            # Training windows only; no test-window fitting, no validation normalization.
            objects = base.build_train_val_objects_for_sample_size(ntrain, pool, Uval, compute_tb_basis=False)
            norms = objects["u_normer"], objects["future_normer"]
            for name in BASELINES:
                seed_everything(seed)
                cfg = {
                    "version": RUN_VERSION, "model": name, "n": ntrain, "seed": seed,
                    "K": 30, "epochs": base.EPOCHS, "batch_size": base.BATCH_SIZE,
                    "lr": base.LR, "wd": base.WEIGHT_DECAY, "width": WIDTH,
                    "original_driver": original_sha, "dataset_size": base.DATA_PATH.stat().st_size
                }
                tag = base.short_hash(cfg)
                dest = RESULT_ROOT / name / f"N{ntrain}_seed{seed}_cfg{tag}"
                dest.mkdir(parents=True, exist_ok=True)
                ckpt = dest / "best.pt"
                rec = dest / "result.json"
                if rec.exists() and ckpt.exists() and not FORCE_RETRAIN:
                    record = json.loads(rec.read_text())
                    print(f"[{name} N={ntrain} seed={seed}] REUSED result {rec}", flush=True)
                    rows.append(record)
                    write_table(rows)
                    _baseline_plot(rows)
                    continue

                model = make_model(name, r, theta).to(base.device)
                nparam, ndof = param_stats(model)
                print(f"[{name} N={ntrain} seed={seed}] "
                      f"parameters={nparam:,}, real degrees of freedom={ndof:,}", flush=True)
                try:
                    if ckpt.exists() and not FORCE_RETRAIN:
                        payload = torch.load(ckpt, map_location=base.device, weights_only=False)
                        model.load_state_dict(payload["state_dict"])
                        history = payload["history"]
                        best_val = payload["best_val_l2"]
                        train_seconds = payload["train_seconds"]
                        print("Loaded completed model checkpoint; evaluating without retraining.", flush=True)
                    else:
                        history, best_val, train_seconds = fit_model(
                            model, objects["train_loader"], objects["val_loader"],
                            norms, name, seed, ntrain)
                        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                    "history": history, "best_val_l2": best_val,
                                    "train_seconds": train_seconds, "configuration": cfg}, ckpt)
                        print(f"Saved trained checkpoint {ckpt}", flush=True)
                    model.eval()
                    if base.device.type == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    pred = rollout(model, Utest[..., 0], norms, base.Nt)
                    if base.device.type == "cuda":
                        torch.cuda.synchronize()
                    infer_seconds = time.perf_counter() - start
                    peak_mib = (torch.cuda.max_memory_allocated() / 1024 ** 2
                                if base.device.type == "cuda" else None)
                    errors = base.rollout_l2_metrics(pred, Utest)
                    diag = base.decomposition_diagnostics(pred, Utest)
                    record = {
                        "model": name, "N_train": ntrain, "seed": seed, "K": base.WINDOW_K,
                        "params": nparam, "real_scalar_dof": ndof,
                        "full_l2_mean": errors["full_l2_mean"],
                        "full_l2_std_over_test_cases": errors["full_l2_std"],
                        "full_l2_median": errors["full_l2_median"],
                        "final_l2_mean": errors["final_l2_mean"],
                        "final_l2_std_over_test_cases": errors["final_l2_std"],
                        "final_l2_median": errors["final_l2_median"],
                        "best_validation_window_l2": best_val,
                        "training_seconds": train_seconds,
                        "full_test_set_inference_seconds": infer_seconds,
                        "per_test_trajectory_inference_seconds": infer_seconds / len(Utest),
                        "peak_inference_allocated_gpu_MiB": peak_mib,
                        **diag, "config_hash": tag
                    }
                    case_table = dest / "test_case_errors.csv"
                    with case_table.open("w", newline="") as output:
                        w = csv.writer(output)
                        w.writerow(["test_index", "original_sample_id", "full_l2", "final_l2"])
                        for i in range(len(Utest)):
                            w.writerow([i, int(sample_ids[teids[i]]), float(errors["all_full_l2"][i]),
                                        float(errors["all_final_l2"][i])])
                    np.savez_compressed(dest / "l2_per_timestep.npz",
                                        time=t, per_test_case=errors["all_l2_t"],
                                        mean=errors["mean_t"], median=errors["median_t"], std=errors["std_t"])
                    if SAVE_ROLLOUTS:
                        np.savez_compressed(dest / "test_rollouts.npz", prediction=pred)
                    plot_curves(history, f"{name} N={ntrain} seed={seed}", dest)
                    plot_errors(errors, t, f"{name} N={ntrain} seed={seed}", dest)
                    plot_annulus_cases(pred, Utest, name, dest)
                    atomic_json(rec, record)
                    rows.append(record)
                    write_table(rows)
                    _baseline_plot(rows)
                    print(f"[{name} N={ntrain} seed={seed}] COMPLETE | full L2="
                          f"{record['full_l2_mean']:.6e}, final L2="
                          f"{record['final_l2_mean']:.6e}, train={train_seconds:.1f}s, "
                          f"inference={infer_seconds:.3f}s, peak GPU={peak_mib}", flush=True)
                    del pred, errors
                except Exception as exc:
                    (dest / "ERROR.txt").write_text(traceback.format_exc())
                    print(f"ERROR: {name} N={ntrain} seed={seed} -> {exc}\n"
                          f"See {dest / 'ERROR.txt'}", flush=True)
                    raise
                finally:
                    del model
                    gc.collect()
                    if base.device.type == "cuda":
                        torch.cuda.empty_cache()
            del objects
            gc.collect()
    print(f"\nCOMPLETE: {len(rows)} baseline evaluations. Summary: {RESULT_ROOT / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
