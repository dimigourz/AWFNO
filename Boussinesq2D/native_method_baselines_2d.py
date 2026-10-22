#!/usr/bin/env python3
"""Native temporal protocols for Boussinesq competitive neural operators.

DO NOT confuse with competitive_baselines_2d.py (K30-window adaptations).
The baseline methods here operate as in our Burgers/Cahn–Hilliard studies:
    F-FNO : u(t) -> u(t+dt), one-step autoregressive, 100 model calls.
    U-Net : u(t) -> u(t+dt), one-step autoregressive, 100 model calls.
    CNO   : (u(0), normalized physical time) -> u(t), 100 independent queries.
    U-NO  : u(t-9:t) -> u(t+dt), H10->H1, 91 autoregressive calls.

CRITICAL: U-NO receives 10 ground-truth initial frames, NOT one. Its
full-trajectory score is deliberately NOT reported, because frames t=1..9
would count as a free perfect prediction. All methods also report L2 on
t>=10 (although U-NO has a conditioning-information advantage).

The same original HyTeG annulus, trajectory partitions, and weighted physical
L2 quadrature are used for all runs. None of the previous K30 baseline
checkpoints or experiment tables are touched.
"""
import os
import csv
import gc
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
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import annulus2d_jcp_experiments as base
from reference_architectures_2d import make_native_model

VERSION = "boussinesq_native_methods_v2_autobatch_earlystop"
VALID_MODELS = ("FFNO", "CNO", "UNet", "UNO")
SELECTED = tuple(s.strip() for s in os.environ.get("NATIVE_MODELS", ",".join(VALID_MODELS)).split(",") if s.strip())
SIZES = tuple(int(s) for s in os.environ.get("NATIVE_SAMPLE_SIZES", "50,100,200").split(",") if s.strip())
SEEDS = tuple(int(s) for s in os.environ.get("NATIVE_SEEDS", "1234").split(",") if s.strip())
EPOCHS = int(os.environ.get("NATIVE_EPOCHS", "200"))
VAL_INTERVAL = int(os.environ.get("NATIVE_VAL_INTERVAL", "10"))
EARLY_STOP_PATIENCE = int(os.environ.get("NATIVE_EARLY_STOP_PATIENCE", "40"))
EARLY_STOP_MIN_EPOCHS = int(os.environ.get("NATIVE_EARLY_STOP_MIN_EPOCHS", "80"))
EARLY_STOP_REL_DELTA = float(os.environ.get("NATIVE_EARLY_STOP_REL_DELTA", "0.001"))
AUTOTUNE_BATCH = os.environ.get("NATIVE_AUTOTUNE_BATCH", "1") == "1"
MAX_GPU_FRACTION = float(os.environ.get("NATIVE_MAX_GPU_FRACTION", "0.75"))
AUTOTUNE_STEPS = int(os.environ.get("NATIVE_AUTOTUNE_STEPS", "2"))
BATCH_CANDIDATES = {
    "FFNO": (8, 16, 32, 64),
    "CNO": (8, 16, 32, 64),
    "UNet": (32, 64, 128, 256),
    "UNO": (16, 32, 64, 128),
}

SAVE_INTERVAL = int(os.environ.get("NATIVE_CHECKPOINT_INTERVAL", "10"))
EVAL_BATCH_SIZE = int(os.environ.get("NATIVE_EVAL_BATCH", "8"))
UNO_HISTORY = 10
OUT = Path(os.environ.get("NATIVE_OUTPUT", str(base.OUTPUT_ROOT / "native_reference_2d")))
CHECKPOINT_DIR = OUT / "checkpoints"
RESULTS_DIR = OUT / "runs"
PLOT_DIR = OUT / "plots"
for p in (OUT, CHECKPOINT_DIR, RESULTS_DIR, PLOT_DIR):
    p.mkdir(parents=True, exist_ok=True)
assert SELECTED and all(m in VALID_MODELS for m in SELECTED), SELECTED
assert SEEDS and all(s >= 0 for s in SEEDS), SEEDS
assert SIZES and all(s in (25,50,100,200,400) for s in SIZES), SIZES
assert base.N_TRAIN_POOL == 400 and base.N_VAL == 30 and base.N_TEST == 70
assert base.TIME_SUBSAMPLE == 1 and base.SPLIT_MODE == "contiguous"
assert EPOCHS > 0 and VAL_INTERVAL >= 1 and SAVE_INTERVAL >= 1
assert EARLY_STOP_PATIENCE >= VAL_INTERVAL and EARLY_STOP_MIN_EPOCHS >= 1
assert 0.0 <= EARLY_STOP_REL_DELTA < 0.1
assert 0.1 < MAX_GPU_FRACTION <= 0.95 and AUTOTUNE_STEPS >= 1

# Cahn–Hilliard-inspired model-specific optimizer and batch-size settings.
# The one-step protocols have ~100 examples/trajectory, so using K30 batch
# and LR schedules would NOT reproduce the original method implementations.
SETTINGS = {
    "FFNO": {"batch": 8, "lr": 5.e-4, "wd": 1.e-4, "noise": .01, "history": 1},
    "CNO": {"batch": 8, "lr": 1.e-3, "wd": 1.e-6, "noise": 0., "history": 1},
    "UNet": {"batch": 32, "lr": 1.e-3, "wd": 1.e-6, "noise": 0., "history": 1},
    "UNO": {"batch": 16, "lr": 1.e-3, "wd": 1.e-6, "noise": 0., "history": UNO_HISTORY},
}

def seed_everything(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

class StateNormalizer:
    def __init__(self, train):
        self.mean = float(np.mean(train, dtype=np.float64))
        self.std = float(np.std(train, dtype=np.float64))
        if not np.isfinite(self.std) or self.std < 1.e-12:
            raise ValueError("Invalid training-only scalar standard deviation.")
    def encode(self, u):
        return (u-self.mean) / (self.std+1.e-6)
    def decode(self, u):
        return u * (self.std+1.e-6) + self.mean

class NativeDataset(Dataset):
    """Lazily index original trajectories: no materialized duplicate windows."""
    def __init__(self, traj, time_grid, kind):
        self.U = torch.from_numpy(np.asarray(traj, dtype=np.float32))
        self.t = torch.as_tensor(time_grid, dtype=torch.float32)
        self.kind = kind
        self.nt = self.U.shape[-1]
        self.hist = UNO_HISTORY if kind=="UNO" else 1
        self.steps = self.nt-1 if kind!="UNO" else self.nt-self.hist
        if self.steps < 1:
            raise ValueError("Not enough snapshots for autoregressive history")
        scale = self.t[-1]-self.t[0]
        self.normalized_t = (self.t-self.t[0])/scale
        assert self.U.ndim == 4

    def __len__(self):
        return len(self.U)*self.steps

    def __getitem__(self,i):
        trajectory = i//self.steps
        step = i%self.steps
        series = self.U[trajectory]
        if self.kind=="CNO":
            future=step+1
            return series[...,0],self.normalized_t[future],series[...,future]
        if self.kind=="UNO":
            # The first 10 fields in rollout are observed; predict the 11th.
            h = series[...,step:step+self.hist]
            return h,torch.tensor(0,dtype=torch.float32),series[...,step+self.hist]
        return series[...,step],torch.tensor(0,dtype=torch.float32),series[...,step+1]

def predict_normalized(model,name,x,t=None,add_noise=False):
    if name=="CNO":
        return model(x,t)
    if name=="FFNO":
        if add_noise:
            x=x+SETTINGS[name]["noise"]*torch.randn_like(x)
        return model(x)[...,0]
    if name=="UNet":
        return model(x)[...,0]
    if name=="UNO":
        return model(x)
    raise ValueError(name)

def single_pass(model,name,loader,normalizer,optimizer=None):
    train = optimizer is not None
    model.train(train)
    agg_loss=agg_l2=0.
    count=0
    with torch.set_grad_enabled(train):
        for x, t, y in loader:
            x=x.to(base.device,non_blocking=True)
            t=t.to(base.device,non_blocking=True)
            y=y.to(base.device,non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            raw=predict_normalized(model,name,normalizer.encode(x),t,add_noise=train)
            target=normalizer.encode(y)
            loss=F.mse_loss(raw,target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: nonfinite {'train' if train else 'validation'} loss")
            if train:
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                if not torch.isfinite(grad):
                    raise FloatingPointError(f"{name}: nonfinite gradient norm")
                optimizer.step()
            pred=normalizer.decode(raw.detach())
            l2=base.abs_l2_field_batch(pred,y).mean()
            n=len(y)
            agg_loss+=float(loss.detach())*n
            agg_l2+=float(l2)*n
            count+=n
    return {"mse":agg_loss/count,"physical_field_l2":agg_l2/count}

def select_batch_size(name, dataset, norm, r, theta, seed):
    """Choose the best measured samples/sec among memory-safe CUDA batches.

    Probes are forward/backward only, without optimizer updates. The actual
    training model is created anew with the requested seed after tuning.
    This changes optimizer batch/update counts; both are saved in output.
    """
    override = int(os.environ.get("NATIVE_BATCH_" + name.upper(), "0"))
    if override:
        print(f"[BATCH] {name}: explicit override batch={override}",flush=True)
        return override
    if not AUTOTUNE_BATCH or base.device.type != "cuda":
        print(f"[BATCH] {name}: using original batch={SETTINGS[name]['batch']}",flush=True)
        return SETTINGS[name]["batch"]
    total = torch.cuda.get_device_properties(base.device).total_memory
    budget = MAX_GPU_FRACTION * total
    candidates = [b for b in BATCH_CANDIDATES[name] if b <= len(dataset)]
    print(f"[BATCH] {name}: GPU={total/2**30:.1f} GiB, "
          f"allocated-memory safety budget={budget/2**30:.1f} GiB, "
          f"candidates={candidates}",flush=True)
    seed_everything(seed)
    model = make_native_model(name,r,theta).to(base.device)
    model.train()
    measurements = []
    try:
        for bs in candidates:
            examples = [dataset[(i*71) % len(dataset)] for i in range(bs)]
            x = torch.stack([v[0] for v in examples]).to(base.device)
            t = torch.stack([v[1] for v in examples]).to(base.device)
            y = torch.stack([v[2] for v in examples]).to(base.device)
            good = True
            periods = []
            peak = 0
            try:
                for attempt in range(AUTOTUNE_STEPS+1):
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    pred = predict_normalized(model,name,norm.encode(x),t)
                    loss = F.mse_loss(pred,norm.encode(y))
                    loss.backward()
                    torch.cuda.synchronize()
                    dt = time.perf_counter()-start
                    if attempt:
                        periods.append(dt)
                        peak = max(peak,torch.cuda.max_memory_allocated())
                if peak >= budget:
                    good = False
                    print(f"[BATCH] {name} batch={bs}: peak={peak/2**30:.2f}GiB "
                          "exceeds safety budget; stopping batch search",flush=True)
                else:
                    throughput = bs/float(np.median(periods))
                    measurements.append((throughput,bs,peak))
                    print(f"[BATCH] {name} batch={bs}: throughput={throughput:.1f} "
                          f"examples/s; peak={peak/2**30:.2f}GiB",flush=True)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                good=False
                print(f"[BATCH] {name} batch={bs}: CUDA OOM; "
                      "selecting a smaller tested size",flush=True)
            finally:
                model.zero_grad(set_to_none=True)
                del examples,x,t,y
                gc.collect()
                torch.cuda.empty_cache()
            if not good:
                break
        if not measurements:
            raise RuntimeError(f"{name}: no batch fits the chosen memory budget")
        choice=max(measurements,key=lambda item:item[0])
        print(f"[BATCH] {name}: selected {choice[1]}, "
              f"measured {choice[0]:.1f} examples/s",flush=True)
        return choice[1]
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def make_scheduler(opt,epochs):
    # Cahn–Hilliard reference: 10-epoch linear warm-up then epoch cosine.
    warm = min(10,max(1,epochs//5))
    if epochs <= warm:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    warmup = torch.optim.lr_scheduler.LinearLR(opt,start_factor=.1,end_factor=1.,total_iters=warm)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,T_max=epochs-warm,eta_min=opt.param_groups[0]['lr']*.02)
    return torch.optim.lr_scheduler.SequentialLR(
        opt,schedulers=[warmup,cosine],milestones=[warm])

def real_dof(model):
    return sum(p.numel()*(2 if p.is_complex() else 1)
               for p in model.parameters() if p.requires_grad)

def snapshot(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def atomic_json(p,obj):
    temp=p.with_suffix(p.suffix+".tmp")
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False))
    temp.replace(p)

def train_or_load(model,name,seed,ntrain,train_loader,val_loader,norm,run_dir,config):
    opt=torch.optim.AdamW(model.parameters(),lr=SETTINGS[name]["lr"],weight_decay=SETTINGS[name]["wd"])
    sched=make_scheduler(opt,EPOCHS)
    ckpt=run_dir/"training_checkpoint.pt"
    history=[]
    best=float("inf")
    best_epoch=0
    best_state=None
    plateau_anchor=float("inf")
    plateau_epoch=0
    epoch0=0
    elapsed=0.
    early_stopped=False
    if ckpt.is_file() and os.environ.get("NATIVE_FORCE_RETRAIN","0")!="1":
        state=torch.load(ckpt,map_location="cpu",weights_only=False)
        if state["config"]!=config:
            raise RuntimeError(f"Checkpoint configuration mismatch: {ckpt}")
        model.load_state_dict(state["model_state"])
        opt.load_state_dict(state["optimizer_state"])
        sched.load_state_dict(state["scheduler_state"])
        epoch0=state["epoch"]
        history=state["history"]
        best=state["best"]
        best_state=state["best_model"]
        best_epoch=state.get("best_epoch",0)
        plateau_anchor=state.get("plateau_anchor",best)
        plateau_epoch=state.get("plateau_epoch",best_epoch)
        early_stopped=state.get("early_stopped",False)
        elapsed=state["elapsed"]
        for state_values in opt.state.values():
            for k,v in state_values.items():
                if torch.is_tensor(v):
                    state_values[k]=v.to(base.device)
        print(f"[RESUME] {name} N={ntrain} seed={seed}, checkpoint epoch "
              f"{epoch0}/{EPOCHS}; early_stop={early_stopped}",flush=True)
    start=time.perf_counter()
    step_times=[]
    if not early_stopped:
        for epoch in range(epoch0+1,EPOCHS+1):
            epoch_start=time.perf_counter()
            tr=single_pass(model,name,train_loader,norm,opt)
            val=None
            if epoch==1 or epoch==EPOCHS or epoch%VAL_INTERVAL==0:
                val=single_pass(model,name,val_loader,norm,None)
                candidate=val["physical_field_l2"]
                if candidate<best:
                    best=candidate
                    best_state=snapshot(model)
                    best_epoch=epoch
                if candidate<plateau_anchor*(1.-EARLY_STOP_REL_DELTA):
                    plateau_anchor=candidate
                    plateau_epoch=epoch
            sched.step()
            history.append({
                "epoch":epoch,"train_mse":tr["mse"],
                "train_physical_field_l2":tr["physical_field_l2"],
                "val_mse":val["mse"] if val else None,
                "val_physical_field_l2":val["physical_field_l2"] if val else None,
                "lr":opt.param_groups[0]["lr"],
            })
            step_times.append(time.perf_counter()-epoch_start)
            early_stopped=(epoch>=EARLY_STOP_MIN_EPOCHS and
                           epoch-plateau_epoch>=EARLY_STOP_PATIENCE)
            if epoch==1 or epoch%VAL_INTERVAL==0 or epoch==EPOCHS or early_stopped:
                avg_sec=float(np.mean(step_times[-min(len(step_times),10):]))
                estimate=(EPOCHS-epoch)*avg_sec/3600.
                print(f"[{name} N={ntrain} seed={seed}] {epoch:03d}/{EPOCHS} "
                      f"train={tr['physical_field_l2']:.5e} "
                      f"val={(val['physical_field_l2'] if val else float('nan')):.5e} "
                      f"best={best:.5e}@{best_epoch}, "
                      f"recent={avg_sec:.1f}s/epoch, "
                      f"max_remaining={estimate:.2f}h",flush=True)
            if epoch%SAVE_INTERVAL==0 or epoch==EPOCHS or early_stopped:
                total=elapsed+time.perf_counter()-start
                tmp=ckpt.with_suffix(".pt.tmp")
                torch.save({
                    "config":config,"epoch":epoch,"model_state":snapshot(model),
                    "optimizer_state":opt.state_dict(),
                    "scheduler_state":sched.state_dict(),
                    "best":best,"best_model":best_state,"best_epoch":best_epoch,
                    "plateau_anchor":plateau_anchor,"plateau_epoch":plateau_epoch,
                    "early_stopped":early_stopped,"history":history,"elapsed":total
                },tmp)
                tmp.replace(ckpt)
            if early_stopped:
                print(f"[EARLY STOP] {name} N={ntrain} seed={seed}, "
                      f"epoch {epoch}; last significant validation gain "
                      f"at epoch {plateau_epoch}; best epoch={best_epoch}",flush=True)
                break
    if best_state is None:
        raise RuntimeError("No valid best-validation checkpoint was recorded")
    model.load_state_dict(best_state)
    completed_epoch=int(history[-1]["epoch"])
    return (history,best,elapsed+time.perf_counter()-start,
            completed_epoch,best_epoch,early_stopped)


@torch.inference_mode()
def native_predict_test(model,name,truth,norm,tgrid):
    """Return predictions over original full grid, including observed prefix.

    Output convention for UNO: t=0..9 filled with OBSERVATIONS; excluded from
    reported full L2 (None); common_tail_l2 t>=10 computed for every model.
    """
    model.eval()
    N,nr,ntheta,n_time=truth.shape
    out=np.empty_like(truth)
    first=UNO_HISTORY if name=="UNO" else 1
    out[...,:first]=truth[...,:first]
    norm_t=np.asarray((tgrid-tgrid[0])/(tgrid[-1]-tgrid[0]),dtype=np.float32)
    for start in range(0,N,EVAL_BATCH_SIZE):
        stop=min(start+EVAL_BATCH_SIZE,N)
        xx=torch.from_numpy(truth[start:stop,...,:first]).to(base.device)
        if name=="UNO":
            state=xx  # (B,Nr,Ntheta,10)
            for ts in range(UNO_HISTORY,n_time):
                yn=model(norm.encode(state))
                y=norm.decode(yn)
                if not torch.isfinite(y).all():
                    raise FloatingPointError(f"UNO: nonfinite evaluation at t={ts}")
                out[start:stop,...,ts]=y.cpu().numpy()
                state=torch.cat((state[...,1:],y[...,None]),dim=-1)
        elif name=="CNO":
            u0=xx[...,0]
            for ts in range(1,n_time):
                query=torch.full((stop-start,),float(norm_t[ts]),device=base.device)
                y=norm.decode(model(norm.encode(u0),query))
                if not torch.isfinite(y).all():
                    raise FloatingPointError(f"CNO: nonfinite evaluation at t={ts}")
                out[start:stop,...,ts]=y.cpu().numpy()
        else:
            state=xx[...,0]
            for ts in range(1,n_time):
                yn=predict_normalized(model,name,norm.encode(state))
                y=norm.decode(yn)
                if not torch.isfinite(y).all():
                    raise FloatingPointError(f"{name}: nonfinite evaluation at t={ts}")
                out[start:stop,...,ts]=y.cpu().numpy()
                state=y
    return out

def plot_run(pred, truth, history, time_grid, run_dir, name):
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([e["epoch"] for e in history],[e["train_physical_field_l2"] for e in history],label="train")
    valid=[e for e in history if e["val_physical_field_l2"] is not None]
    ax.plot([e["epoch"] for e in valid],[e["val_physical_field_l2"] for e in valid],
            label="validation")
    ax.set(xlabel="epoch",ylabel="one-step or direct-time physical L2")
    ax.legend();fig.tight_layout();fig.savefig(run_dir/"training.png",dpi=140);plt.close(fig)

    per=base.l2_per_timestep_np(pred,truth)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(time_grid,per.mean(axis=0))
    if name=="UNO":
        ax.axvline(time_grid[UNO_HISTORY-1],color="gray",linestyle="--",
                   label="observed prefix ends")
        ax.legend()
    ax.set(xlabel="time",ylabel="mean spatial physical L2",title=name)
    fig.tight_layout();fig.savefig(run_dir/"error_vs_time.png",dpi=140);plt.close(fig)
    np.savez_compressed(run_dir/"error_vs_time.npz",time=time_grid,per_test=per)

def score(pred, truth, name):
    common_start=UNO_HISTORY
    tail=base.abs_l2_full_traj_np(pred[...,common_start:],truth[...,common_start:])
    final=base.abs_l2_final_np(pred,truth)
    if name=="UNO":
        full_mean=full_sd=None
    else:
        full=base.abs_l2_full_traj_np(pred,truth)
        full_mean=float(np.mean(full));full_sd=float(np.std(full))
    return {
        "full_l2_mean":full_mean,
        "full_l2_sd_over_test":full_sd,
        "final_l2_mean":float(np.mean(final)),
        "final_l2_sd_over_test":float(np.std(final)),
        "tail_from_t10_l2_mean":float(np.mean(tail)),
        "tail_from_t10_l2_sd_over_test":float(np.std(tail)),
        "observed_true_frames":UNO_HISTORY if name=="UNO" else 1,
        "prediction_start_index":UNO_HISTORY if name=="UNO" else 1,
        "metrics_definition":"Physical annulus L2; tail integral starts t10. UNO full metric omitted because initial 10 frames are observed."
    }

def write_summary(rows):
    """Keep previous N25 measurements alongside new N50/100/200 experiments.

    Original N25 models used small fixed batches and 200 epochs; new batches
    and early stopping are different optimization settings. These must be
    identifiable in the result manifest, not conflated as one homogeneous run.
    """
    all_rows={}
    old_path=OUT/"results.json"
    if old_path.is_file():
        for record in json.loads(old_path.read_text()):
            key=(record["model"],int(record["N_train"]),int(record["seed"]))
            record.setdefault("training_protocol_version","v1_original_batches")
            all_rows[key]=record
    for record in rows:
        key=(record["model"],int(record["N_train"]),int(record["seed"]))
        all_rows[key]=record
    ordered=sorted(all_rows.values(),key=lambda r:(r["N_train"],r["model"],r["seed"]))
    if not ordered:
        return
    atomic_json(old_path,ordered)
    keys=list(dict.fromkeys(k for row in ordered for k in row))
    tmp=OUT/"results.csv.tmp"
    with tmp.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(ordered)
    tmp.replace(OUT/"results.csv")


def run_one(name,ntrain,seed,pool,val,test,tgrid,arch_sha,base_sha):
    settings=SETTINGS[name]
    source_dataset=NativeDataset(pool[:ntrain],tgrid,name)
    val_dataset=NativeDataset(val,tgrid,name)
    normalizer=StateNormalizer(pool[:ntrain])
    chosen_batch=select_batch_size(name,source_dataset,normalizer,
                                   base.r_grid_np,base.theta_grid_np,seed)
    # Re-seed after throughput measurements for clean deterministic initialization.
    seed_everything(seed)
    generator=torch.Generator().manual_seed(seed+ntrain)
    train_loader=DataLoader(source_dataset,batch_size=chosen_batch,shuffle=True,
                            pin_memory=base.device.type=="cuda",generator=generator,num_workers=0)
    val_loader=DataLoader(val_dataset,batch_size=chosen_batch,shuffle=False,
                          pin_memory=base.device.type=="cuda",num_workers=0)
    cfg={
        "version":VERSION,"model":name,"N_train":ntrain,"seed":seed,
        "architecture_sha":arch_sha,"base_sha":base_sha,
        "epochs":EPOCHS,"lr":settings["lr"],"weight_decay":settings["wd"],
        "batch_size":chosen_batch,"history":settings["history"],
        "batch_selection":"auto_measured_throughput" if AUTOTUNE_BATCH else "fixed",
        "early_stop_patience":EARLY_STOP_PATIENCE,
        "early_stop_min_epochs":EARLY_STOP_MIN_EPOCHS,
        "early_stop_relative_delta":EARLY_STOP_REL_DELTA,
        "validation_interval":VAL_INTERVAL,
        "train_items":len(source_dataset),"val_items":len(val_dataset),
        "test_trajectories":len(test),
        "optimizer":"AdamW","scheduler":"warmup_cosine_10epochs",
        "optimizer_loss":"normalized_MSE","train_noise_std":settings["noise"],
        "data_normalization":"training-only scalar mean/std",
    }
    config_hash=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:12]
    target=RESULTS_DIR/name/f"N{ntrain}_seed{seed}_{config_hash}"
    target.mkdir(parents=True,exist_ok=True)
    final=target/"result.json"
    if final.is_file() and (target/"training_checkpoint.pt").exists() \
            and os.environ.get("NATIVE_FORCE_RETRAIN","0")!="1":
        print(f"[CACHE] {name} N={ntrain} seed={seed}: {final}",flush=True)
        return json.loads(final.read_text())
    print(f"[START] {name} N={ntrain} seed={seed} examples={len(source_dataset)}, "
          f"updates/epoch={(len(source_dataset)+settings['batch']-1)//settings['batch']}",flush=True)
    model=make_native_model(name,base.r_grid_np,base.theta_grid_np).to(base.device)
    nparam=base.count_params(model)
    nreal=real_dof(model)
    try:
        history,best,train_sec,epochs_done,best_epoch,early_stop=train_or_load(
            model,name,seed,ntrain,train_loader,val_loader,normalizer,target,cfg)
        if base.device.type=="cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        # Warm up convolution algorithms + GPU initialization before measuring
        # steady-state full-test inference times and allocated peak memory.
        _=native_predict_test(model,name,test[:1],normalizer,tgrid)
        if base.device.type=="cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0=time.perf_counter()
        pred=native_predict_test(model,name,test,normalizer,tgrid)
        if base.device.type=="cuda":torch.cuda.synchronize()
        infer=time.perf_counter()-t0
        peak=(torch.cuda.max_memory_allocated()/1024**2
              if base.device.type=="cuda" else None)
        metrics=score(pred,test,name)
        record={
            "model":name,"N_train":ntrain,"seed":seed,
            "protocol":("direct_time_u0_t" if name=="CNO"
                        else "autoregressive_H10_to_H1" if name=="UNO"
                        else "autoregressive_H1_to_H1"),
            "parameters":nparam,"real_scalar_dof":nreal,
            "train_examples":len(source_dataset),
            "optimizer_steps":epochs_done*len(train_loader),
            "training_protocol_version":VERSION,
            "epochs_completed":epochs_done,"max_epochs":EPOCHS,
            "early_stopped":early_stop,"best_epoch":best_epoch,
            "effective_batch_size":chosen_batch,
            "best_val_physical_field_l2":best,"training_seconds":train_sec,
            "full_test_set_inference_seconds":infer,
            "peak_gpu_allocated_MiB":peak,
            **metrics
        }
        plot_run(pred,test,history,tgrid,target,name)
        atomic_json(target/"configuration.json",cfg)
        atomic_json(final,record)
        if os.environ.get("NATIVE_SAVE_ROLLOUTS","0")=="1":
            np.savez_compressed(target/"test_prediction.npz",prediction=pred)
        print(f"[COMPLETE] {name} N={ntrain} final={record['final_l2_mean']:.6e}, "
              f"full={record['full_l2_mean']}, tail={record['tail_from_t10_l2_mean']:.6e}, "
              f"train={train_sec:.2f}s, inference={infer:.2f}s, "
              f"batch={chosen_batch}, epochs={epochs_done}",flush=True)
        return record
    except Exception:
        (target/"error_traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        del model
        gc.collect()
        if base.device.type=="cuda":torch.cuda.empty_cache()

def smoke():
    print(f"[SMOKE] Native protocols {SELECTED} on annulus grid",flush=True)
    r=np.linspace(.5,1.,64,dtype=np.float32)
    th=np.arange(128,dtype=np.float32)*2*np.pi/128
    for name in SELECTED:
        seed_everything(1234)
        model=make_native_model(name,r,th).to(base.device)
        count=2
        x=torch.randn((count,64,128,UNO_HISTORY) if name=="UNO" else
                      (count,64,128),device=base.device)
        t=torch.tensor([.2,.8],dtype=torch.float32,device=base.device)
        y=predict_normalized(model,name,x,t)
        assert y.shape==(count,64,128),(name,tuple(y.shape))
        assert torch.isfinite(y).all(),name
        y.square().mean().backward()
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                raise AssertionError(f"{name}: nonfinite gradients")
        print(f"[SMOKE] PASS {name}, physical prediction shape={tuple(y.shape)}, "
              f"parameters={base.count_params(model):,}",flush=True)
        del model,x,y
        gc.collect()
        if base.device.type=="cuda":torch.cuda.empty_cache()

def main():
    arch_sha=hashlib.sha256(Path(__import__("reference_architectures_2d").__file__).read_bytes()).hexdigest()
    base_sha=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
    U,r,theta,t,sample_ids=base.load_annulus_dataset(base.DATA_PATH)
    U,r,theta=base.downsample_polar_grid(U,r,theta,base.TARGET_NR,base.TARGET_NTHETA)
    base.set_geometry_from_arrays(U.shape,r,theta,t)
    pool,tr_idx,val,v_idx,test,t_idx=base.make_train_val_test_splits(U)
    assert (base.Nr,base.Ntheta,base.Nt)==(64,128,101)
    if base.RUN_EXACT_HASH_OVERLAP_CHECK:
        base.check_exact_hash_overlap(pool,val,"train","validation")
        base.check_exact_hash_overlap(pool,test,"train","test")
        base.check_exact_hash_overlap(val,test,"validation","test")
    print(f"Native baseline study: {SELECTED}, sizes={SIZES}, seeds={SEEDS}, "
          f"epochs={EPOCHS}. Dataset split train/val/test=400/30/70",flush=True)
    print("WARNING: UNO_H10 uses 10 observed truth frames; score is NOT u0-only comparable.",flush=True)
    atomic_json(OUT/"manifest.json",{
        "version":VERSION,"models":SELECTED,"sizes":SIZES,"seeds":SEEDS,
        "autotune_batch":AUTOTUNE_BATCH,
        "batch_candidates":BATCH_CANDIDATES,
        "max_gpu_fraction":MAX_GPU_FRACTION,
        "early_stop_patience":EARLY_STOP_PATIENCE,
        "early_stop_min_epochs":EARLY_STOP_MIN_EPOCHS,
        "early_stop_relative_delta":EARLY_STOP_REL_DELTA,
        "note":"N25 results in results.csv may come from v1 fixed-batch full-200-epoch runs; compare configurations.",
        "epochs":EPOCHS,"base_sha":base_sha,"architecture_sha":arch_sha,
        "sample_dataset_bytes":base.DATA_PATH.stat().st_size,
        "protocols":{
            "FFNO":"one-step H1->H1, autoregressive test rollout",
            "CNO":"direct-time (u0,t)->u(t), independent queries",
            "UNet":"one-step H1->H1, autoregressive test rollout",
            "UNO":"H10->H1 autoregressive with first 10 test fields observed; not u0-only comparable"
        },
        "metrics":"same annulus physical L2; UNO full L2 omitted",
        "original_K30_results":"stored separately; untouched",
    })
    rows=[]
    for ntrain in SIZES:
        for seed in SEEDS:
            for name in SELECTED:
                rows.append(run_one(name,ntrain,seed,pool,val,test,t,arch_sha,base_sha))
                write_summary(rows)
    print("Completed native-method comparisons:",len(rows),"result file:",OUT/"results.csv",flush=True)

if __name__=="__main__":
    if os.environ.get("NATIVE_SMOKE_ONLY","0")=="1":
        smoke()
    else:
        main()
