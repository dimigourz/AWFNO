#!/usr/bin/env python3
"""Rank-only TB-AWFNO study using the original benchmark's models and metrics.

This driver intentionally trains no direct AWFNO, WFNO, FNO, CNO, or U-Net.
The Colab launcher supplies SOURCE_FILE and all benchmark environment variables
*before* this module imports the original engine, which reads its configuration
at import time. Each (PDE, N, K, M) has an isolated Drive output directory.
"""
import os, sys, json, time, hashlib, importlib.util
from pathlib import Path
import numpy as np
import torch

PDE = os.environ["TB_PDE"]
SOURCE_FILE = Path(os.environ["TB_SOURCE_FILE"])
OUT = Path(os.environ["TB_RANK_OUTPUT"])
N = int(os.environ["TB_N_TRAIN"])
K = int(os.environ["TB_WINDOW_K"])
M = int(os.environ["TB_RANK_M"])
SEED = int(os.environ.get("TB_SEED", "1234"))
if PDE not in ("burgers_1d","boussinesq_2d","navier_stokes_2d"):
    raise ValueError("Unknown PDE: "+PDE)
if not (1 <= M <= K):
    raise ValueError(f"Invalid integer rank M={M} for K={K}")
assert SOURCE_FILE.is_file()
OUT.mkdir(parents=True, exist_ok=True)
SOURCE_SHA256 = hashlib.sha256(SOURCE_FILE.read_bytes()).hexdigest()

spec = importlib.util.spec_from_file_location("awfno_rank_reference", SOURCE_FILE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
assert base.WINDOW_K == K and base.N_TIME_MODES == M, (base.WINDOW_K,base.N_TIME_MODES,K,M)

def save_scores(fit, model, predicted, truth, energy, label):
    metrics = base.rollout_l2_metrics(predicted, truth)
    result = {
        "pde": PDE, "model": "TB-AWFNO", "N_train": N,
        "K": K, "M": M, "seed": SEED,
        "source_sha256": SOURCE_SHA256,
        "n_test": int(len(truth)),
        "best_validation_window_l2": float(fit["best_val_l2"]),
        "train_seconds": float(fit["train_time_sec"]),
        "parameters": int(base.count_params(model)),
        "full_l2_mean": float(metrics["full_l2_mean"]),
        "full_l2_std": float(metrics["full_l2_std"]),
        "final_l2_mean": float(metrics["final_l2_mean"]),
        "final_l2_std": float(metrics["final_l2_std"]),
        "train_only_temporal_basis_energy": float(energy),
        "metric": "absolute physical L2 over full trajectory and final time",
        "test_selection": "model checkpoint chosen on validation windows, not test",
    }
    (OUT/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savetxt(OUT/"l2_vs_time.csv",
               np.column_stack((np.arange(len(metrics["mean_t"])),metrics["mean_t"],metrics["std_t"])),
               delimiter=",",header="frame,mean_spatial_l2,std_spatial_l2",comments="")
    print("FINISHED",label, json.dumps(result,indent=2),flush=True)

def train_or_load_model(model, predict, obj, checkpoint, mode):
    if checkpoint.exists() and os.environ.get("FORCE_RETRAIN","0") != "1":
        payload = torch.load(checkpoint, map_location=base.device,weights_only=False)
        model.load_state_dict(payload["state_dict"], strict=True)
        meta = payload.get("meta", {})
        print("REUSING COMPLETED CHECKPOINT",checkpoint,flush=True)
        return {"model":model,"best_val_l2":meta.get("best_val_l2",float("nan")),
                "train_time_sec":meta.get("train_time_sec",float("nan"))}
    fit=base.fit_tb_awfno_model(model,obj["train_loader"],obj["val_loader"],predict)
    base.save_model_checkpoint(str(checkpoint),fit["model"],
                               {"best_val_l2":fit["best_val_l2"],
                                "train_time_sec":fit["train_time_sec"],
                                "M":M,"K":K,"N":N,"seed":SEED,
                                "source_sha256":SOURCE_SHA256})
    return fit

def run_burgers():
    U,x,t=base.load_hdf5_1d_dataset(base.DATA_PATH)
    U,x=base.downsample_space_stride(U,x,base.TARGET_NX)
    base.set_geometry_from_arrays(U.shape,x,t)
    pool,train_idx,val,val_idx,test,test_idx=base.make_train_val_test_splits(U)
    if base.RUN_EXACT_HASH_OVERLAP_CHECK:
        base.check_exact_hash_overlap(pool,val,"train","validation")
        base.check_exact_hash_overlap(pool,test,"train","test")
    obj=base.build_train_val_objects_for_sample_size(N,pool,val)
    model=base.TBAWFNO1d(M,base.x_grid_np).to(base.device)
    predict=base.make_tb_awfno_predict(model,obj["u_normer"],obj["basis_torch"])
    fit=train_or_load_model(model,predict,obj,OUT/"best.pt","burgers")
    pred=base.autoregressive_rollout_tb_awfno(
        model,test[:,:,0],test.shape[-1],predict,False)
    save_scores(fit,model,pred,test,obj["basis_energy_ratio"],"Burgers")

def run_boussinesq():
    U,r,theta,t,sample_ids=base.load_annulus_dataset(base.DATA_PATH)
    U,r,theta=base.downsample_polar_grid(U,r,theta,base.TARGET_NR,base.TARGET_NTHETA)
    base.set_geometry_from_arrays(U.shape,r,theta,t)
    pool,train_idx,val,val_idx,test,test_idx=base.make_train_val_test_splits(U)
    if base.RUN_EXACT_HASH_OVERLAP_CHECK:
        base.check_exact_hash_overlap(pool,val,"train","validation")
        base.check_exact_hash_overlap(pool,test,"train","test")
    obj=base.build_train_val_objects_for_sample_size(N,pool,val,compute_tb_basis=True)
    model=base.TBAWFNO2d(M,base.r_grid_np,base.theta_grid_np).to(base.device)
    predict=base.make_tb_awfno_predict(model,obj["u_normer"],obj["basis_torch"])
    fit=train_or_load_model(model,predict,obj,OUT/"best.pt","boussinesq")
    pred=base.autoregressive_rollout_tb_awfno(
        model,test[:,:,:,0],test.shape[-1],predict,False)
    save_scores(fit,model,pred,test,obj["basis_energy_ratio"],"Boussinesq")

def run_navier_stokes():
    pool,val,test,train_idx,val_idx,test_idx,x,y,t=base.load_or_create_split()
    base.set_geometry(pool.shape,x,y,t)
    if base.RUN_EXACT_HASH_OVERLAP_CHECK:
        base.check_exact_overlap(pool,val,"train","validation")
        base.check_exact_overlap(pool,test,"train","test")
    obj=base.build_objects(N,pool,val)
    model=base.TBAWFNO(base.C_DYN,M).to(base.device)
    predict=base.make_tb_predict(model,obj["normer"],obj["basis_t"])
    completed=OUT/"best.pt"
    if completed.exists() and os.environ.get("FORCE_RETRAIN","0")!="1":
        payload=torch.load(completed,map_location=base.device,weights_only=False)
        model.load_state_dict(payload["state_dict"],strict=True)
        fit={"model":model,**payload.get("meta",{})}
        print("REUSING COMPLETED CHECKPOINT",completed,flush=True)
    else:
        # Native NS implementation writes resumable optimizer checkpoints
        # every five epochs and restores the best validation-window model.
        fit=base._checkpoint_fit(model,obj["train_loader"],obj["val_loader"],
                                 predict,obj["normer"],"TB-AWFNO",
                                 OUT/"training_checkpoint.pt")
        base.save_model_checkpoint(str(completed),fit["model"],
                                   {"best_val_l2":fit["best_val_l2"],
                                    "train_time_sec":fit["train_time_sec"],
                                    "K":K,"M":M,"N":N,"seed":SEED,
                                    "source_sha256":SOURCE_SHA256})
    pred=base.rollout_tb(model,test,predict)
    save_scores(fit,model,pred,test,obj["energy"],"Navier-Stokes")

def main():
    print("TB rank experiment:",PDE,"N",N,"K",K,"M",M,"source",SOURCE_SHA256,flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU before training")
    if PDE=="burgers_1d":run_burgers()
    elif PDE=="boussinesq_2d":run_boussinesq()
    else:run_navier_stokes()

if __name__ == "__main__":
    main()
