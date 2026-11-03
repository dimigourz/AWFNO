#!/usr/bin/env python3
"""TB-AWFNO-only temporal-rank experiments for the four paper benchmarks.

Inputs are provided by the matching Colab notebook through environment variables.
Each subprocess trains exactly one (PDE, N_train, K, M, seed) configuration,
evaluates on the fixed test set, and saves self-describing artifacts to Drive.
No baseline models are trained. Source implementations remain unmodified.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

PDE = os.environ["TB_PDE"]
SOURCE_FILE = Path(os.environ["TB_SOURCE_FILE"])
OUT = Path(os.environ["TB_RANK_OUTPUT"])
N = int(os.environ["TB_N_TRAIN"])
K = int(os.environ["TB_WINDOW_K"])
M = int(os.environ["TB_RANK_M"])
SEED = int(os.environ.get("TB_SEED", "1234"))
TRAIN_POOL_SIZE = int(os.environ["TB_TRAIN_POOL_SIZE"])
if PDE not in {"burgers_1d", "cahn_hilliard_2d", "boussinesq_2d", "navier_stokes_2d"}:
    raise ValueError("Unknown PDE: " + PDE)
if not (1 <= M <= K and 1 <= N <= TRAIN_POOL_SIZE):
    raise ValueError(f"Invalid N={N}, K={K}, M={M}, pool={TRAIN_POOL_SIZE}")
if not SOURCE_FILE.is_file():
    raise FileNotFoundError(SOURCE_FILE)
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_SHA = hashlib.sha256(SOURCE_FILE.read_bytes()).hexdigest()
DATA_FILE = Path(os.environ.get("DATA_FILE", os.environ.get("DATA_PATH", "")))
if not DATA_FILE.is_file():
    raise FileNotFoundError(f"Missing original dataset: {DATA_FILE}")
DATA_BYTES = DATA_FILE.stat().st_size

module_spec = importlib.util.spec_from_file_location("tb_rank_base", SOURCE_FILE)
base = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = base
module_spec.loader.exec_module(base)
if PDE == "cahn_hilliard_2d":
    assert base.WINDOW_K == K and base.N_TIME_MODES == M
    assert base.N_TRAIN == N
    assert base.SPLIT_SEED == 2026, "Set the original split seed 2026."
else:
    assert base.WINDOW_K == K and base.N_TIME_MODES == M
if PDE == "navier_stokes_2d":
    assert base.SEED == 1234, "Original NS source hardcodes seed 1234."
elif PDE != "cahn_hilliard_2d":
    assert base.SEED == SEED


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def write_json(obj, path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str))
    tmp.replace(path)


def split_record(train_ids, val_ids, test_ids):
    train_ids = np.asarray(train_ids, dtype=np.int64)
    val_ids = np.asarray(val_ids, dtype=np.int64)
    test_ids = np.asarray(test_ids, dtype=np.int64)
    if len(set(train_ids) & set(val_ids)) or len(set(train_ids) & set(test_ids)) or len(set(val_ids) & set(test_ids)):
        raise RuntimeError("Overlapping train, validation and test trajectory indices")
    np.savez(OUT / "split_indices.npz", train_indices=train_ids,
             val_indices=val_ids, test_indices=test_ids)
    h = hashlib.sha256()
    for ids in (train_ids, val_ids, test_ids):
        h.update(ids.tobytes())
    return h.hexdigest()


def config_signature(split_sha):
    config = {
        "pde": PDE, "source_sha256": SOURCE_SHA, "dataset_file": DATA_FILE.name,
        "dataset_bytes": DATA_BYTES, "split_sha256": split_sha,
        "n_train": N, "training_pool": TRAIN_POOL_SIZE, "K": K, "M": M,
        "seed": SEED, "epochs": int(base.EPOCHS),
        "batch_size": int(base.BATCH_SIZE),
        "lr": float(base.LR), "weight_decay": float(base.WEIGHT_DECAY),
        "width": int(base.WIDTH), "layers": int(base.N_LAYERS),
        "final_loss_weight": float(base.FINAL_LOSS_WEIGHT),
    }
    write_json(config, OUT / "configuration.json")
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def checkpoint_fit(model, tr_loader, va_loader, train_step, val_step,
                   score_key, config_hash):
    """Exact native TB training epochs with resumable optimizer/scheduler state.

    Four main model sources use AdamW, cosine decay, best validation-window
    physical L2, and train-only temporal basis. Save full state every 5 epochs.
    """
    best_path = OUT / "best.pt"
    last_path = OUT / "last_epoch.pt"
    if best_path.exists() and os.environ.get("FORCE_RETRAIN", "0") != "1":
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        if state.get("config_hash") != config_hash:
            raise RuntimeError(f"Finished checkpoint configuration mismatch: {best_path}")
        model.load_state_dict(state["model"], strict=True)
        print("[CHECKPOINT] Reusing fully trained model", best_path, flush=True)
        return state["fit"]

    opt = torch.optim.AdamW(model.parameters(), lr=float(base.LR),
                            weight_decay=float(base.WEIGHT_DECAY))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(base.EPOCHS))
    first = 1
    best = float("inf")
    best_state = None
    training_seconds = 0.0
    history = []
    if last_path.exists() and os.environ.get("FORCE_RETRAIN", "0") != "1":
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["config_hash"] != config_hash:
            raise RuntimeError(f"Partial checkpoint configuration mismatch: {last_path}")
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        best = float(state["best"])
        best_state = state["best_model"]
        first = int(state["epoch"]) + 1
        training_seconds = float(state["training_seconds"])
        history = list(state["history"])
        torch.set_rng_state(state["rng_cpu"])
        if torch.cuda.is_available() and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        if tr_loader.generator is not None and state.get("shuffle_state") is not None:
            tr_loader.generator.set_state(state["shuffle_state"])
        print(f"[RESUME] epoch {first}/{base.EPOCHS}", flush=True)

    for epoch in range(first, int(base.EPOCHS) + 1):
        start = time.perf_counter()
        tr = train_step(model, tr_loader, opt)
        va = val_step(model, va_loader)
        sched.step()
        training_seconds += time.perf_counter() - start
        current = float(va[score_key])
        if not np.isfinite(current):
            raise FloatingPointError("Nonfinite validation L2")
        if current < best:
            best = current
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        line = {"epoch": epoch, "train_loss": float(tr["loss"]),
                "val_loss": float(va["loss"]), "val_window_l2": current,
                "best_val_window_l2": best}
        history.append(line)
        if epoch == 1 or epoch % 10 == 0 or epoch == base.EPOCHS:
            print(f"[EPOCH] N={N} M={M} {epoch}/{base.EPOCHS} "
                  f"train={line['train_loss']:.5e} val_l2={current:.5e} "
                  f"best={best:.5e}", flush=True)
        if epoch % 5 == 0 or epoch == base.EPOCHS:
            atomic_save({
                "config_hash": config_hash, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "epoch": epoch, "best": best, "best_model": best_state,
                "training_seconds": training_seconds, "history": history,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "shuffle_state": tr_loader.generator.get_state()
                                 if tr_loader.generator is not None else None,
            }, last_path)

    if best_state is None:
        raise RuntimeError("No valid best validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    fit = {"best_val_l2": best, "train_time_sec": training_seconds,
           "epochs_completed": int(base.EPOCHS)}
    atomic_save({"config_hash": config_hash, "model": best_state, "fit": fit}, best_path)
    write_json(history, OUT / "training_history.json")
    return fit


def save_metrics(model, predicted, true, best_val, train_time, basis_energy,
                 split_sha, epochs_completed, extras=None):
    if PDE == "cahn_hilliard_2d":
        metrics = base.physical_l2_metrics(predicted, true)
        params = base.count_tensor_params(model)
    else:
        metrics = base.rollout_l2_metrics(predicted, true)
        params = base.count_params(model)
    result = {
        "pde": PDE, "model": "TB-AWFNO", "N_train": N,
        "K": K, "M": M, "seed": SEED, "n_test": int(len(true)),
        "source_sha256": SOURCE_SHA,
        "dataset_path": str(DATA_FILE), "dataset_size_bytes": DATA_BYTES,
        "split_sha256": split_sha,
        "parameters": int(params),
        "epochs_configured": int(base.EPOCHS),
        "epochs_completed": int(epochs_completed),
        "best_validation_window_l2": float(best_val),
        "train_seconds": float(train_time),
        "full_l2_mean": float(metrics["full_l2_mean"]),
        "full_l2_std": float(metrics["full_l2_std"]),
        "final_l2_mean": float(metrics["final_l2_mean"]),
        "final_l2_std": float(metrics["final_l2_std"]),
        "train_only_temporal_basis_energy": float(basis_energy),
        "metric": "absolute physical L2, full trajectory and final time",
        "split": "fixed full training pool, fixed validation/test across sample sizes",
        "test_selection": "best checkpoint chosen on validation window L2 only",
    }
    result.update(extras or {})
    l2t = base.l2_per_timestep_np(predicted, true) if PDE != "cahn_hilliard_2d" else None
    if l2t is not None:
        np.savetxt(OUT / "l2_vs_time.csv",
                   np.column_stack((np.arange(l2t.shape[1]), l2t.mean(0), l2t.std(0))),
                   delimiter=",", header="frame,mean_spatial_l2,std_spatial_l2", comments="")
    write_json(result, OUT / "result.json")
    print("[FINISHED]", PDE, f"N={N} K={K} M={M}",
          f"full L2={result['full_l2_mean']:.6e}, final L2={result['final_l2_mean']:.6e}",
          flush=True)


def run_burgers():
    U, x, t = base.load_hdf5_1d_dataset(base.DATA_PATH)
    U, x = base.downsample_space_stride(U, x, base.TARGET_NX)
    base.set_geometry_from_arrays(U.shape, x, t)
    pool, pool_ids, val, val_ids, test, test_ids = base.make_train_val_test_splits(U)
    assert len(pool) == TRAIN_POOL_SIZE
    split_sha = split_record(pool_ids[:N], val_ids, test_ids)
    obj = base.build_train_val_objects_for_sample_size(N, pool, val)
    set_seed()
    model = base.TBAWFNO1d(M, base.x_grid_np).to(base.device)
    predict = base.make_tb_awfno_predict(model, obj["u_normer"], obj["basis_torch"])
    digest = config_signature(split_sha)
    fit = checkpoint_fit(model, obj["train_loader"], obj["val_loader"],
                         lambda m,l,o: base.train_epoch_tb_awfno(m,l,o,predict),
                         lambda m,l: base.eval_epoch_tb_awfno(m,l,predict),
                         "abs_l2", digest)
    predicted = base.autoregressive_rollout_tb_awfno(
        model, test[..., 0], test.shape[-1], predict, False)
    save_metrics(model,predicted,test,fit["best_val_l2"],fit["train_time_sec"],
                 obj["basis_energy_ratio"],split_sha,fit["epochs_completed"])


def run_boussinesq():
    U,r,theta,t,sample_ids = base.load_annulus_dataset(base.DATA_PATH)
    U,r,theta = base.downsample_polar_grid(U,r,theta,base.TARGET_NR,base.TARGET_NTHETA)
    base.set_geometry_from_arrays(U.shape,r,theta,t)
    pool,pool_ids,val,val_ids,test,test_ids = base.make_train_val_test_splits(U)
    assert len(pool) == TRAIN_POOL_SIZE
    split_sha = split_record(pool_ids[:N],val_ids,test_ids)
    obj = base.build_train_val_objects_for_sample_size(N,pool,val,compute_tb_basis=True)
    set_seed()
    model = base.TBAWFNO2d(M,base.r_grid_np,base.theta_grid_np).to(base.device)
    predict = base.make_tb_awfno_predict(model,obj["u_normer"],obj["basis_torch"])
    digest = config_signature(split_sha)
    fit = checkpoint_fit(model,obj["train_loader"],obj["val_loader"],
                         lambda m,l,o: base.train_epoch_tb_awfno(m,l,o,predict),
                         lambda m,l: base.eval_epoch_tb_awfno(m,l,predict),
                         "abs_l2",digest)
    predicted = base.autoregressive_rollout_tb_awfno(
        model,test[...,0],test.shape[-1],predict,False)
    save_metrics(model,predicted,test,fit["best_val_l2"],fit["train_time_sec"],
                 obj["basis_energy_ratio"],split_sha,fit["epochs_completed"])


def run_cahn_hilliard():
    U = base.load_cahn_hilliard(base.DATA_FILE)
    needed = TRAIN_POOL_SIZE + base.N_VAL + base.N_TEST
    if len(U) < needed:
        raise ValueError(f"Need {needed} trajectories, found {len(U)}")
    # Exactly the original AWFNO main-study permutation and fixed pool of 400.
    order = np.random.default_rng(base.SPLIT_SEED).permutation(len(U))
    pool_ids = order[:TRAIN_POOL_SIZE]
    val_ids = order[TRAIN_POOL_SIZE:TRAIN_POOL_SIZE+base.N_VAL]
    test_ids = order[TRAIN_POOL_SIZE+base.N_VAL:needed]
    train, val, test = U[pool_ids[:N]], U[val_ids], U[test_ids]
    split_sha = split_record(pool_ids[:N],val_ids,test_ids)
    tb_spec = base.tb_alt_spec("TB_AWFNO_LFL",n_time_modes=M)
    base.validate_spec(tb_spec)
    set_seed()
    tr,va,input_norm,future_norm = base.make_loaders(train,val,K,SEED)
    model = base.build_model(tb_spec).to(base.DEVICE)
    start,future = base.build_windows(train,K)
    basis,energy,singular = base.compute_time_pod_basis_from_windows(start,future,M)
    del start,future
    predict = base.make_tb_predict(model,input_norm,basis)
    digest = config_signature(split_sha)
    fit = checkpoint_fit(model,tr,va,
                         lambda m,l,o: base.run_epoch(m,l,predict,future_norm,optimizer=o),
                         lambda m,l: base.run_epoch(m,l,predict,future_norm,optimizer=None),
                         "window_l2",digest)
    predicted = base.recursive_rollout(model,predict,test,K)
    save_metrics(model,predicted,test,fit["best_val_l2"],fit["train_time_sec"],
                 energy,split_sha,fit["epochs_completed"])


def run_navier_stokes():
    pool,val,test,pool_ids,val_ids,test_ids,x,y,t = base.load_or_create_split()
    base.set_geometry(pool.shape,x,y,t)
    assert len(pool) >= TRAIN_POOL_SIZE, "Original file must include 200 training IDs."
    split_sha = split_record(pool_ids[:N],val_ids,test_ids)
    obj = base.build_objects(N,pool,val)
    set_seed()
    model = base.TBAWFNO(base.C_DYN,M).to(base.device)
    predict = base.make_tb_predict(model,obj["normer"],obj["basis_t"])
    digest = config_signature(split_sha)
    best_path=OUT/"best.pt"
    if best_path.exists() and os.environ.get("FORCE_RETRAIN","0")!="1":
        state=torch.load(best_path,map_location="cpu",weights_only=False)
        if state.get("config_hash")!=digest:
            raise RuntimeError("NS best-checkpoint configuration mismatch.")
        model.load_state_dict(state["model"],strict=True)
        fit=state["fit"]
    else:
        fit=base._checkpoint_fit(model,obj["train_loader"],obj["val_loader"],
                                 predict,obj["normer"],"TB-AWFNO",
                                 OUT/"training_checkpoint.pt")
        fit={"best_val_l2":float(fit["best_val_l2"]),
             "train_time_sec":float(fit["train_time_sec"]),
             "epochs_completed":len(fit["history"]["val_l2"])}
        atomic_save({"config_hash":digest,
                     "model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                     "fit":fit},best_path)
    predicted=base.rollout_tb(model,test,predict)
    save_metrics(model,predicted,test,fit["best_val_l2"],fit["train_time_sec"],
                 obj["energy"],split_sha,fit["epochs_completed"])


def main():
    print("[TB RANK STUDY]",PDE,f"N={N}, K={K}, M={M}, seed={SEED}",
          "source SHA256",SOURCE_SHA,flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before training.")
    set_seed()
    if PDE=="burgers_1d":
        run_burgers()
    elif PDE=="cahn_hilliard_2d":
        run_cahn_hilliard()
    elif PDE=="boussinesq_2d":
        run_boussinesq()
    else:
        run_navier_stokes()


if __name__ == "__main__":
    main()
