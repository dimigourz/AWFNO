#!/usr/bin/env python3
"""Independent, auditable 2D Kolmogorov Re=500 neural-operator comparison.

Do not import older Burgers/Boussinesq/Navier-Stokes model implementations.
All methods receive ONE observed initial vorticity frame; their temporal
prediction formulations are explicitly different and labelled.

Targets: 128x128 vorticity, 101 frames over physical forecast time 0..0.25.
Models: AWFNO-GLL(K20), TB-AWFNO-GLL(K20, train-only POD temporal basis),
         WFNO(K20), parameter-matched WFNO(K20), FNO(direct 100-frame).
All quantitative figures are computed with dx*dy and dt physical quadrature.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, sys, time, shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

VERSION = 'ns-re500-original-method-comparison-clean-v1'
EXPECTED_DATA_SHA = '7030bc43e6331cdf472182ae0ed4f2684c18da8d336ca9ed92c37baf0bfc8a06'
PARENT_SHA = '82786a4293ac29bf23756c2cc39188e3b7fbb84bd26ef47061415fe76103ebb3'
MODELS = ('AWFNO', 'TB-AWFNO', 'WFNO', 'WFNO-PM', 'FNO')
SIZES = (25, 50, 100, 200)
NX = 128
NT = 101
DT = 0.0025
LX = 2 * math.pi
DX = LX / NX
K_DEFAULT = 20


def emit(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)


def hash_config(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:14]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_source(path, verify_hash=False):
    path = Path(path)
    assert path.is_file(), f'Missing dataset: {path}'
    meta_path = path.parent / 'metadata.json'
    assert meta_path.is_file(), f'Missing numerical verification metadata: {meta_path}'
    meta = json.loads(meta_path.read_text())
    assert meta['status'] == 'PASS'
    assert meta['sha256'] == EXPECTED_DATA_SHA
    assert meta['parent_sha256'] == PARENT_SHA
    assert meta['shape'] == [300, 128, 128, 101]
    if verify_hash:
        assert sha256_file(path) == EXPECTED_DATA_SHA, 'Dataset SHA256 mismatch'
    try:
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        data = torch.load(path, map_location='cpu', weights_only=True)
    U = data['U']
    assert U.shape == (300, NX, NX, NT) and U.dtype == torch.float32
    assert data['parent_dataset_sha256'] == PARENT_SHA
    assert math.isclose(float(data['forecast_start']), 0.25, abs_tol=1e-7)
    assert math.isclose(float(data['T_forecast']), 0.25, abs_tol=1e-7)
    t = np.asarray(data['times'], dtype=np.float64)
    assert np.allclose(t, np.arange(NT) * DT, atol=2e-7)
    for key, lo, hi in (('train_indices', 0, 200), ('val_indices', 200, 230), ('test_indices', 230, 300)):
        assert torch.equal(data[key], torch.arange(lo, hi)), f'Split mismatch for {key}'
    x = np.asarray(data['x'], dtype=np.float64)
    y = np.asarray(data['y'], dtype=np.float64)
    assert x.shape == (NX,) and y.shape == (NX,)
    assert np.allclose(x, np.arange(NX) * DX, atol=1e-6)
    assert np.allclose(y, np.arange(NX) * DX, atol=1e-6)
    return U


def train_stats(U, indexes):
    # Training-only statistics; streaming to avoid >1 GiB temporary arrays.
    summ = sqsum = 0.0
    count = 0
    for idx in indexes:
        w = U[int(idx)].numpy()
        summ += float(np.sum(w, dtype=np.float64))
        sqsum += float(np.sum(w.astype(np.float64) ** 2))
        count += w.size
    avg = summ / count
    std = math.sqrt(max(sqsum / count - avg * avg, 1e-10))
    return float(avg), float(std)


class FramePairs(Dataset):
    """Window samples: (u(t), u(t+1),...,u(t+K)); full source trajectories split by ID."""
    def __init__(self, U, indexes, K, stride):
        self.U = U
        self.ids = [int(x) for x in indexes]
        self.K = int(K)
        self.starts = list(range(0, NT-K, stride))
        assert self.starts and self.starts[-1] + K < NT
    def __len__(self):
        return len(self.ids) * len(self.starts)
    def __getitem__(self, ix):
        i = self.ids[ix // len(self.starts)]
        s = self.starts[ix % len(self.starts)]
        v = self.U[i]
        # (1,H,W), (K,H,W); no future-state inputs at inference.
        return v[...,s].unsqueeze(0), v[...,s+1:s+1+self.K].permute(2,0,1).contiguous()


class FullTrajectories(Dataset):
    """Full-FNO training: (u0 -> 100 future fields), one sample per trajectory."""
    def __init__(self, U, indexes):
        self.U = U
        self.ids = [int(x) for x in indexes]
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, ix):
        u = self.U[self.ids[ix]]
        return u[...,0].unsqueeze(0), u[...,1:].permute(2,0,1).contiguous()


def periodic_features(device, dtype=torch.float32):
    axis = 2 * torch.pi * torch.arange(NX, device=device, dtype=dtype) / NX
    x, y = torch.meshgrid(axis, axis, indexing='ij')
    return torch.stack((x.sin(), x.cos(), y.sin(), y.cos(),
                        (4*y).sin(), (4*y).cos()), dim=0).unsqueeze(0)


class Fourier2d(nn.Module):
    """2D truncated rFFT with BOTH positive and negative x frequency blocks."""
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        assert 1 <= modes <= NX//2
        self.modes = modes
        self.a = nn.Parameter(torch.randn(in_ch, out_ch, modes, modes, dtype=torch.cfloat)
                              / math.sqrt(in_ch*out_ch))
        self.b = nn.Parameter(torch.randn(in_ch, out_ch, modes, modes, dtype=torch.cfloat)
                              / math.sqrt(in_ch*out_ch))
    def forward(self, x):
        b, _, h, w = x.shape
        m = self.modes
        assert 2*m <= h and m <= w//2 + 1
        X = torch.fft.rfft2(x, norm='ortho')
        Y = torch.zeros((b, self.a.shape[1],h,w//2+1),device=x.device,dtype=X.dtype)
        Y[:,:,:m,:m] = torch.einsum('bixy,ioxy->boxy', X[:,:,:m,:m],self.a)
        Y[:,:,-m:,:m] = torch.einsum('bixy,ioxy->boxy', X[:,:,-m:,:m],self.b)
        return torch.fft.irfft2(Y,s=(h,w),norm='ortho')


class MixedBlock(nn.Module):
    def __init__(self, width, modes, radius=2, order='GLL'):
        super().__init__()
        assert order in ('LGL','GLL','LLG','GGG','LLL')
        self.ops=nn.ModuleList([
            Fourier2d(width,width,modes) if x=='G' else nn.Conv2d(
                width,width,2*radius+1,padding=radius,padding_mode='circular')
            for x in order
        ])
    def forward(self,x):
        for op in self.ops:
            x = x + F.gelu(op(x))
        return x


class FourierBlock(nn.Module):
    """Windowed FNO block: one spectral map + pointwise path per depth."""
    def __init__(self,width,modes):
        super().__init__()
        self.spectral=Fourier2d(width,width,modes)
        self.pointwise=nn.Conv2d(width,width,1)
    def forward(self,x):
        return F.gelu(self.spectral(x)+self.pointwise(x))


class OperatorModel(nn.Module):
    """All variants have exactly ONE initial observed field at test time."""
    def __init__(self, method, width, modes, depth, K, rank=12, order='GLL', radius=2):
        super().__init__()
        assert method in MODELS
        self.method, self.width, self.K, self.rank = method, width, K, rank
        self.register_buffer('grid',periodic_features('cpu'),persistent=False)
        self.lift=nn.Conv2d(1+6,width,1)
        mixed=method in ('AWFNO','TB-AWFNO')
        self.blocks=nn.ModuleList([
            MixedBlock(width,modes,radius,order) if mixed else FourierBlock(width,modes)
            for _ in range(depth)
        ])
        self.hidden=nn.Conv2d(width,128,1)
        outc=rank if method=='TB-AWFNO' else (NT-1 if method=='FNO' else K)
        self.final=nn.Conv2d(128,outc,1)
        if method=='TB-AWFNO':
            assert 1 <= rank <= K
            self.register_buffer('temporal_basis',torch.zeros(K,rank))
    def forward(self, u0):
        assert u0.ndim == 4 and u0.shape[1] == 1
        x=torch.cat((u0,self.grid.to(device=u0.device,dtype=u0.dtype).expand(u0.shape[0],-1,-1,-1)),dim=1)
        x=self.lift(x)
        for layer in self.blocks:
            x=layer(x)
        z=self.final(F.gelu(self.hidden(x)))
        if self.method=='TB-AWFNO':
            # basis[k,r] fit on TRAIN trajectories only. Predict *increments*.
            increment=torch.einsum('brxy,kr->bkxy',z,self.temporal_basis)
            return u0+increment
        return z


def real_dof(model):
    return int(sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad))


def param_count(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def match_wfno_width(target, modes, depth, K, tolerance=.05):
    candidates=[]
    for width in range(12,97):
        m=OperatorModel('WFNO-PM',width,modes,depth,K)
        n=real_dof(m)
        candidates.append((abs(n-target)/target,width,n))
        del m
    best=min(candidates)
    if best[0]>tolerance:
        raise RuntimeError(f'Unable to match real parameter count within {tolerance:.0%}: {best}')
    return best[1],best[2],best[0]


def compute_train_basis(U, train_ids, K, rank, stride=10, spatial_stride=4):
    """Training-only POD for residual windows, with deterministic uniform spatial quadrature.

    R has shape (sampled space-time locations, K); its covariance R^T R
    yields orthonormal temporal modes phi_j(t) for TB-AWFNO reconstruction.
    """
    cov=np.zeros((K,K),dtype=np.float64)
    rows=0
    for i in train_ids:
        w=U[int(i)].numpy()[::spatial_stride,::spatial_stride,:]
        for start in range(0,NT-K,stride):
            delta=w[...,start+1:start+1+K]-w[...,start,None]
            X=delta.reshape(-1,K).astype(np.float64,copy=False)
            cov+=X.T@X
            rows+=len(X)
    vals,vecs=np.linalg.eigh(cov)
    take=np.argsort(vals)[::-1]
    vals=np.maximum(vals[take],0)
    basis=vecs[:,take[:rank]].astype(np.float32)
    # Deterministic sign: largest magnitude component is positive.
    for col in range(rank):
        v=basis[:,col]
        if v[np.argmax(np.abs(v))]<0:basis[:,col]*=-1
    energy=float(vals[:rank].sum()/max(vals.sum(),1e-30))
    assert np.allclose(basis.T@basis,np.eye(rank),atol=3e-5)
    return torch.from_numpy(basis.copy()),energy,rows


def make_model(method, args, pm_width):
    width=pm_width if method=='WFNO-PM' else args.width
    return OperatorModel(method,width,args.modes,args.depth,args.window_k,
                         args.basis_rank,args.order,args.radius)


def load_normalized_stats(U, ids, output):
    # Reuse TRAIN-only normalization across the five methods at each N.
    path=output/f'train_normalization_N{len(ids)}.json'
    if path.is_file():
        ob=json.loads(path.read_text())
        assert ob['dataset_sha256']==EXPECTED_DATA_SHA and ob['ntrain']==len(ids)
        return float(ob['mean']),float(ob['std'])
    mean,std=train_stats(U,ids)
    assert std>0 and np.isfinite(std)
    atomic_json(path,{'dataset_sha256':EXPECTED_DATA_SHA,'ntrain':len(ids),
                      'mean':mean,'std':std,'fit_split':'train_only'})
    return mean,std


def prepare_batch(batch, device, mean, std):
    inp, future=batch
    return ((inp.to(device,non_blocking=True)-mean)/std,
            (future.to(device,non_blocking=True)-mean)/std)


def model_predictions(model, U, ids, device, mean, std, batch_size=3):
    """Full *free* rollouts. No true post-t0 observations are passed into models."""
    model.eval()
    predicted=[]; truth=[]
    with torch.inference_mode():
        for ix in range(0,len(ids),batch_size):
            selection=list(ids[ix:ix+batch_size])
            source=torch.stack([U[int(i)] for i in selection],dim=0)
            initial=((source[...,0].unsqueeze(1).to(device)-mean)/std)
            if model.method=='FNO':
                frames=model(initial)
                assert frames.shape[1]==NT-1
                parts=[frames]
            else:
                current=initial
                parts=[]
                for start in range(0,NT-1,model.K):
                    y=model(current)
                    length=min(model.K,NT-1-start)
                    parts.append(y[:,:length])
                    current=y[:,length-1:length].detach()
            output=torch.cat(parts,dim=1)
            assert output.shape==(len(selection),NT-1,NX,NX)
            full=torch.cat((initial,output),dim=1)*std+mean
            # (B,T,H,W) for both truth and prediction
            predicted.append(full.float().cpu().numpy())
            truth.append(source.permute(0,3,1,2).float().numpy())
    return np.concatenate(predicted,axis=0),np.concatenate(truth,axis=0)


def l2_metrics(pred,truth):
    """Absolute physical L2(Ω×T) with trapezoidal time quadrature; mean per trajectory."""
    assert pred.shape==truth.shape and pred.ndim==4
    error=(pred.astype(np.float64)-truth.astype(np.float64))
    l2=np.sqrt(DX*DX*np.sum(error**2,axis=(2,3)))
    weights=np.ones(NT,dtype=np.float64)
    weights[0]=weights[-1]=0.5
    full=np.sqrt(DT*np.sum((l2**2)*weights[None,:],axis=1))
    final=l2[:,-1]
    return {'full':full,'final':final,'per_timestep':l2}


def moment(xs):
    a=np.asarray(xs,dtype=np.float64)
    return dict(mean=float(a.mean()),std=float(a.std(ddof=0)),median=float(np.median(a)))


def spectral_region_metrics(pred,truth):
    """Final-time low/high Fourier error and high-gradient-region relative error."""
    final_p=pred[:,-1].astype(np.float64)
    final_t=truth[:,-1].astype(np.float64)
    k=np.fft.fftfreq(NX)*NX
    band=np.maximum(np.abs(k[:,None]),np.abs(k[None,:]))
    E=np.fft.fft2(final_p-final_t,axes=(-2,-1),norm='ortho')
    U=np.fft.fft2(final_t,axes=(-2,-1),norm='ortho')
    results={}
    for name,mask in (('low',band<16),('high',(band>=16)&(band<43))):
        en=np.sum(np.abs(E[:,mask])**2,axis=1)
        tn=np.sum(np.abs(U[:,mask])**2,axis=1)
        results[f'{name}_frequency_absolute_l2']=moment(np.sqrt(DX*DX*en))
        results[f'{name}_frequency_relative_l2']=moment(np.sqrt(en/np.maximum(tn,1e-20)))
    gx=(np.roll(final_t,-1,axis=1)-np.roll(final_t,1,axis=1))/(2*DX)
    gy=(np.roll(final_t,-1,axis=2)-np.roll(final_t,1,axis=2))/(2*DX)
    grad=np.hypot(gx,gy)
    mask=grad>=np.quantile(grad,0.9,axis=(1,2),keepdims=True)
    e2=(final_p-final_t)**2
    rel=np.sqrt(np.sum(e2*mask,axis=(1,2)) /
                np.maximum(np.sum(final_t**2*mask,axis=(1,2)),1e-30))
    results['sharp_region_relative_l2']=moment(rel)
    return results


def plot_case(output,pred,truth,case,name,frame_ids=(0,25,50,75,100)):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,len(frame_ids),figsize=(17,9),constrained_layout=True)
    gt=truth[case];pr=pred[case]
    vmax=float(np.max(np.abs(gt)))+1e-10
    emax=max(float(np.percentile(np.abs(pr-gt),99)),1e-6)
    for j,t in enumerate(frame_ids):
        axs[0,j].imshow(gt[t].T,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        axs[1,j].imshow(pr[t].T,origin='lower',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        axs[2,j].imshow(np.abs(pr[t]-gt[t]).T,origin='lower',cmap='magma',vmin=0,vmax=emax)
        axs[0,j].set_title(f't={t*DT:.4f}')
        for row in range(3):
            axs[row,j].set_xticks([]);axs[row,j].set_yticks([])
    for i,caption in enumerate(('True ω','Predicted ω','|Error|')):
        axs[i,0].set_ylabel(caption)
    fig.suptitle(name)
    fig.savefig(output,dpi=120)
    plt.close(fig)


def plot_training(folder,history):
    import matplotlib.pyplot as plt
    if not history:return
    f,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    epochs=[h['epoch'] for h in history]
    ax.plot(epochs,[h['train_mse'] for h in history],label='train window MSE')
    vals=[h for h in history if np.isfinite(h['val_full_l2'])]
    if vals:
        ax2=ax.twinx()
        ax2.plot([h['epoch'] for h in vals], [h['val_full_l2'] for h in vals],'-o',color='C1',label='val full L2')
        ax2.set_ylabel('Absolute validation full L2')
    ax.set_xlabel('Epoch');ax.set_ylabel('Mean normalized train-window MSE')
    ax.grid(alpha=.2)
    f.savefig(folder/'training_history.png',dpi=130)
    plt.close(f)


def plot_error_curve(folder,metrics,comparison='test'):
    import matplotlib.pyplot as plt
    per=metrics['per_timestep']
    mean=per.mean(0);std=per.std(0)
    t=np.arange(NT)*DT
    f,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    ax.plot(t,mean,label='Model',lw=2)
    ax.fill_between(t,np.maximum(0,mean-std),mean+std,alpha=.2,label='±1 std')
    ax.set(title=f'{comparison}: free-rollout absolute physical L2',xlabel='Forecast time',ylabel='L2(Ω)')
    ax.grid(alpha=.2);ax.legend()
    f.savefig(folder/f'{comparison}_error_vs_time.png',dpi=140)
    plt.close(f)
    np.savez_compressed(folder/f'{comparison}_curves.npz',t=t,mean=mean,std=std)


def elapsed_s(device,start):
    if device.type=='cuda':torch.cuda.synchronize(device)
    return time.perf_counter()-start


def run_epoch(model,loader,device,mean,std,optimizer,seed,epoch):
    model.train(); total=0.; cnt=0
    # A deterministic epoch shuffle is restored exactly on resume.
    generator=torch.Generator().manual_seed(seed+1000003*epoch)
    order=torch.randperm(len(loader.dataset),generator=generator).tolist()
    for beginning in range(0,len(order),loader.batch_size):
        idxs=order[beginning:beginning+loader.batch_size]
        batch=[loader.dataset[j] for j in idxs]
        x=torch.stack([b[0] for b in batch]);y=torch.stack([b[1] for b in batch])
        xn,yn=prepare_batch((x,y),device,mean,std)
        optimizer.zero_grad(set_to_none=True)
        predicted=model(xn)
        assert predicted.shape==yn.shape, f'Target mismatch {predicted.shape} != {yn.shape}'
        loss=F.mse_loss(predicted,yn)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(grad)):raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        total+=float(loss.detach().item())*len(idxs)
        cnt+=len(idxs)
    return total/cnt,math.ceil(len(order)/loader.batch_size)


def record_config(args,model,n,method,pm_width,pm_gap):
    return dict(version=VERSION,dataset_sha256=EXPECTED_DATA_SHA,
                n_train=n,seed=args.seed,method=method,epochs=args.max_epochs,
                early_stop_patience=args.patience,early_stop_min_epochs=args.min_epochs,
                val_every=args.val_every,window_stride=args.window_stride,
                k=args.window_k,basis_rank=args.basis_rank,
                order=args.order,depth=args.depth,modes=args.modes,radius=args.radius,
                width=model.width,reference_width=args.width,
                batch_size=args.batch_size,eval_batch_size=args.eval_batch,
                lr=args.lr,weight_decay=args.wd,pm_width=pm_width,
                pm_real_dof_relative_gap=pm_gap,
                normalized_targets=True,train_only_normalization=True,
                selection='best_validation_physical_absolute_L2_Omega_times_T',
                test_sealed_until_all_models_trained=True)


def model_folder(root,config):
    return root/'runs'/config['method']/f"N{config['n_train']}_seed{config['seed']}_{hash_config(config)}"


def save_checkpoint(folder,model,opt,config,epoch,best,best_epoch,best_state,history,
                    mean,std,basis_energy,optimizer_steps,training_seconds):
    tmp=folder/'checkpoint.tmp'
    payload=dict(epoch=epoch,state=model.state_dict(),optimizer=opt.state_dict(),
                 best=best,best_epoch=best_epoch,best_state=best_state,
                 history=history,config=config,mean=mean,std=std,basis_energy=basis_energy,
                 steps=optimizer_steps,elapsed=training_seconds,
                 rng_torch=torch.get_rng_state(),rng_numpy=np.random.get_state(),
                 rng_python=random.getstate(),
                 rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    torch.save(payload,tmp)
    tmp.replace(folder/'checkpoint.pt')


def fit_one(U,args,n,method,pm_width,pm_gap,device):
    # Avoid loading any test tensors until ALL models have been trained and selected.
    ids=np.arange(n)
    valid=np.arange(200,230)
    model=make_model(method,args,pm_width).to(device)
    config=record_config(args,model,n,method,pm_width,pm_gap)
    folder=model_folder(args.output,config);folder.mkdir(parents=True,exist_ok=True)
    done=folder/'training_result.json'
    if done.exists() and not args.force:
        emit(f'[CACHED] {method} N={n} from {done}')
        return json.loads(done.read_text())
    avg,std=load_normalized_stats(U,ids,args.output)
    if method=='TB-AWFNO':
        basis,energy,rows=compute_train_basis(U,ids,args.window_k,args.basis_rank,args.window_stride)
        model.temporal_basis.copy_(basis.to(device))
        emit(f'[TB BASIS] N={n} rank={args.basis_rank} captured energy={100*energy:.4f}% from {rows:,} training rows')
    else:energy=float('nan')
    dataset=(FullTrajectories(U,ids) if method=='FNO'
             else FramePairs(U,ids,args.window_k,args.window_stride))
    # Use Dataset's own deterministic epoch sampling for exact resumability.
    loader=DataLoader(dataset,batch_size=args.batch_size)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.wd)
    # Fixed LR for unambiguous epoch-resume, early stopping via validation.
    history=[];best=float('inf');best_epoch=0;best_state=None;last_epoch=0
    steps=0;prev_elapsed=0.;bad=0
    ckpt=folder/'checkpoint.pt'
    if ckpt.exists() and not args.force:
        state=torch.load(ckpt,map_location='cpu',weights_only=False)
        if state['config']!=config:raise ValueError('Checkpoint config mismatch')
        model.load_state_dict(state['state'])
        optimizer.load_state_dict(state['optimizer'])
        best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state']
        history=state['history'];last_epoch=state['epoch'];steps=state['steps']
        prev_elapsed=state['elapsed']
        torch.set_rng_state(state['rng_torch'])
        np.random.set_state(state['rng_numpy']);random.setstate(state['rng_python'])
        if state['rng_cuda'] is not None and device.type=='cuda':
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        bad=0
        for h in reversed(history):
            if np.isfinite(h['val_full_l2']):
                if h['epoch'] != best_epoch:bad+=1
                else:break
        emit(f'[RESUME] {method} N={n} epoch {last_epoch}/{args.max_epochs}')
    emit(f'[TRAIN] {method} N={n}: {len(dataset)} samples/epoch, '
         f'{param_count(model):,} PyTorch params / {real_dof(model):,} real DoF, '
         f'epochs <= {args.max_epochs}, training batch={args.batch_size}')
    start=time.perf_counter()
    if last_epoch==args.max_epochs and best_state is None:
        raise RuntimeError('Reached training cap but checkpoint has no best validation state')
    for epoch in range(last_epoch+1,args.max_epochs+1):
        mse,nsteps=run_epoch(model,loader,device,avg,std,optimizer,args.seed,epoch)
        steps+=nsteps
        val_full=float('nan');val_final=float('nan')
        if epoch==1 or epoch%args.val_every==0 or epoch==args.max_epochs:
            emit(f'[VALIDATE] {method} N={n} epoch {epoch}: 30 trajectories x 101 frames')
            predicted,true=model_predictions(model,U,valid,device,avg,std,args.eval_batch)
            vm=l2_metrics(predicted,true)
            val_full=moment(vm['full'])['mean'];val_final=moment(vm['final'])['mean']
            if val_full<best-args.min_delta:
                best=val_full;best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                bad=0
            else:bad+=1
            del predicted,true
        history.append(dict(epoch=epoch,train_mse=mse,val_full_l2=val_full,
                            val_final_l2=val_final,optimizer_steps=steps))
        duration=prev_elapsed+elapsed_s(device,start)
        emit(f'[EPOCH] {method} N={n} epoch {epoch}/{args.max_epochs}, '
             f'loss={mse:.5e}, val_full={val_full:.6f}, '
             f'best={best:.6f}@{best_epoch}, '
             f'optimizer_steps={steps}, elapsed={duration:.1f}s')
        if epoch%args.checkpoint_every==0 or epoch==args.max_epochs or (
                epoch>=args.min_epochs and bad>=args.patience):
            save_checkpoint(folder,model,optimizer,config,epoch,best,best_epoch,best_state,
                            history,avg,std,energy,steps,duration)
        if (epoch>=args.min_epochs and bad>=args.patience):
            emit(f'[EARLY STOP] {method} N={n} after {bad} non-improving validation checks')
            break
        if time.perf_counter()-args.session_start>args.session_minutes*60:
            # Save current epoch even if not a checkpoint multiple.
            save_checkpoint(folder,model,optimizer,config,epoch,best,best_epoch,best_state,
                            history,avg,std,energy,steps,duration)
            emit('[BUDGET] Session budget reached. Rerun training to resume automatically.')
            raise SessionBudgetExceeded
    if best_state is None:raise RuntimeError('No validation checkpoint')
    model.load_state_dict(best_state)
    chosen_pred,chosen_true=model_predictions(model,U,valid,device,avg,std,args.eval_batch)
    chosen_metrics=l2_metrics(chosen_pred,chosen_true)
    plot_training(folder,history)
    plot_error_curve(folder,chosen_metrics,'validation')
    plot_case(folder/'validation_vorticity_0.png',chosen_pred,chosen_true,0,f'{method} N{n} VALIDATION example 0')
    plot_case(folder/'validation_vorticity_15.png',chosen_pred,chosen_true,15,f'{method} N{n} VALIDATION example 15')
    torch.save({'state_dict':best_state,'config':config,'mean':avg,'std':std,
                'basis_energy':energy},folder/'best_validation.pt')
    duration=prev_elapsed+elapsed_s(device,start)
    row=dict(config_hash=hash_config(config),version=VERSION,method=method,N_train=n,
             seed=args.seed,parameters=param_count(model),real_scalar_dof=real_dof(model),
             width=model.width,K=(0 if method=='FNO' else args.window_k),
             observed_input_frames=1,prediction_protocol='direct_100_frame' if method=='FNO'
             else 'windowed_basis_increment' if method=='TB-AWFNO' else 'windowed_K20',
             train_samples_per_epoch=len(dataset),optimizer_steps=steps,
             epochs_completed=history[-1]['epoch'],best_epoch=best_epoch,
             best_validation_full_l2=float(moment(chosen_metrics['full'])['mean']),
             best_validation_final_l2=float(moment(chosen_metrics['final'])['mean']),
             train_seconds=float(duration),basis_energy_fraction=None if method!='TB-AWFNO' else energy,
             pm_real_dof_relative_gap=None if method!='WFNO-PM' else pm_gap,
             checkpoint=str(folder/'best_validation.pt'))
    atomic_json(done,row)
    del model,optimizer,chosen_pred,chosen_true
    if device.type=='cuda':torch.cuda.empty_cache()
    return row


class SessionBudgetExceeded(Exception):
    pass


def all_expected(args):
    return [(n,m) for n in args.sizes for m in args.methods]


def expected_configs(args,pm_width,pm_gap):
    result=[]
    for n,method in all_expected(args):
        m=make_model(method,args,pm_width)
        cfg=record_config(args,m,n,method,pm_width,pm_gap)
        result.append(dict(method=method,N_train=n,hash=hash_config(cfg),path=str(model_folder(args.output,cfg))))
    return result


def summarize_tables(root,rows,from_test=False):
    import pandas as pd
    if not rows:return
    df=pd.DataFrame(rows).sort_values(['N_train','method']).reset_index(drop=True)
    df.to_csv(root/('results.csv' if from_test else 'training_selection.csv'),index=False)
    df.to_json(root/('results.json' if from_test else 'training_selection.json'),orient='records',indent=2)
    if from_test:
        columns=['N_train','method','K','prediction_protocol','observed_input_frames',
                 'parameters','real_scalar_dof','full_l2_mean','full_l2_std','full_l2_median',
                 'final_l2_mean','final_l2_std','final_l2_median',
                 'train_seconds','inference_seconds_70','inference_seconds_per_trajectory',
                 'peak_inference_gpu_mib','best_epoch','optimizer_steps']
        df[[c for c in columns if c in df.columns]].to_csv(root/'publication_table.csv',index=False)
        print('\n=== Boussinesq/Burgers-style model-by-training-size table (TEST) ===',flush=True)
        print(df[[c for c in columns if c in df.columns]].to_string(index=False),flush=True)
    else:
        columns=['N_train','method','parameters','real_scalar_dof',
                 'best_validation_full_l2','best_validation_final_l2','train_seconds',
                 'epochs_completed','optimizer_steps']
        print('\n=== TRAIN/VALIDATION ONLY (TEST SEALED) ===',flush=True)
        print(df[[c for c in columns if c in df.columns]].to_string(index=False),flush=True)


def plot_size_summary(root,rows):
    import matplotlib.pyplot as plt
    if not rows:return
    by={(int(r['N_train']),r['method']):r for r in rows}
    for name in ('full','final'):
        f,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        for method in MODELS:
            nlist=sorted({n for n,m in by if m==method})
            if not nlist:continue
            values=[by[(n,method)][f'{name}_l2_mean'] for n in nlist]
            ax.plot(nlist,values,'-o',label=method)
        ax.set_xlabel('Number of training trajectories')
        ax.set_ylabel(f'Mean test absolute physical {name} L2')
        ax.set_title(f'Test {name}-trajectory L2 vs training-set size')
        ax.grid(alpha=.2);ax.legend()
        f.savefig(root/f'all_models_{name}_l2_vs_n.png',dpi=140)
        plt.close(f)


def plot_cross_model_figures(root, rows):
    """Burgers/Boussinesq-format plots from completed TEST metrics, no retraining."""
    import matplotlib.pyplot as plt
    lookup={(int(r['N_train']),r['method']):r for r in rows}
    for metric,title,name in (
        ('train_seconds','Training wall time','all_models_training_time_vs_n.png'),
        ('inference_seconds_70','Full 70-case test inference time','all_models_inference_time_vs_n.png'),
        ('sharp_region_relative_l2_mean','Sharp-gradient-region error at final time',
         'all_models_sharp_region_error_vs_n.png'),
        ('high_frequency_relative_l2_mean','High-frequency relative error at final time',
         'all_models_high_frequency_error_vs_n.png')):
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        for model in MODELS:
            nlist=sorted(n for n in SIZES if (n,model) in lookup)
            if nlist:
                ax.plot(nlist,[lookup[(n,model)][metric] for n in nlist],'-o',label=model)
        ax.set(xlabel='Training trajectories',ylabel=title,title=title+' vs N')
        ax.grid(alpha=.2);ax.legend()
        fig.savefig(root/name,dpi=140)
        plt.close(fig)
    for n in SIZES:
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        count=0
        for model in MODELS:
            if (n,model) not in lookup:continue
            row=lookup[(n,model)]
            folder=Path(row['checkpoint']).parent
            p=folder/'test_curves.npz'
            if not p.is_file():continue
            arr=np.load(p)
            ax.plot(arr['t'],arr['mean'],label=model,lw=1.8)
            count+=1
        ax.set(title=f'Test mean physical L2 vs time | N={n} | {count} methods',
               xlabel='Forecast time',ylabel='Absolute physical L2(Ω)')
        ax.grid(alpha=.2);ax.legend()
        fig.savefig(root/f'all_models_error_vs_time_N{n}.png',dpi=150)
        plt.close(fig)
    if all((n,'AWFNO') in lookup and (n,'WFNO-PM') in lookup for n in SIZES):
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        for method in ('AWFNO','WFNO-PM'):
            vals=[lookup[(n,method)]['full_l2_mean'] for n in SIZES]
            ax.plot(SIZES,vals,'-o',label=method)
        ax.set(xlabel='Training trajectories',ylabel='Mean full L2(Ω×T)',
               title='Parameter-matched AWFNO versus WFNO')
        ax.grid(alpha=.2);ax.legend()
        fig.savefig(root/'matched_WFNO_vs_AWFNO_full_l2.png',dpi=150)
        plt.close(fig)


def evaluate_test(U,args,expected,device):
    assert all((Path(j['path'])/'training_result.json').is_file() for j in expected),(
        'Refusing test evaluation until ALL predeclared training runs complete')
    locked=expected
    lock_path=args.output/'LOCKED_TEST_PLAN.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text())!=locked:
            raise RuntimeError('Test plan was already locked to different configurations')
    else:atomic_json(lock_path,locked)
    test_ids=np.arange(230,300)
    test_rows=[]
    for j,record in enumerate(locked,1):
        folder=Path(record['path'])
        finish=folder/'test_result.json'
        if finish.exists() and not args.force_test:
            emit(f'[CACHED TEST] {j}/{len(locked)}: {record["method"]} N{record["N_train"]}')
            test_rows.append(json.loads(finish.read_text()));continue
        training=json.loads((folder/'training_result.json').read_text())
        weight=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=False)
        config=weight['config']
        assert hash_config(config)==record['hash']
        m=OperatorModel(config['method'],config['width'],config['modes'],config['depth'],
                        config['k'],config['basis_rank'],config['order'],config['radius']).to(device)
        m.load_state_dict(weight['state_dict'])
        m.eval()
        emit(f'[TEST] {j}/{len(locked)}: {record["method"]} N={record["N_train"]} '
             f'(one initial observation, 70 trajectories)')
        if device.type=='cuda':
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        pred,true=model_predictions(m,U,test_ids,device,weight['mean'],weight['std'],args.eval_batch)
        sec=elapsed_s(device,started)
        peak=float(torch.cuda.max_memory_allocated(device)/2**20) if device.type=='cuda' else 0.0
        metric=l2_metrics(pred,true)
        diagnostic=spectral_region_metrics(pred,true)
        row=dict(training,
                 full_l2_mean=moment(metric['full'])['mean'],
                 full_l2_std=moment(metric['full'])['std'],
                 full_l2_median=moment(metric['full'])['median'],
                 final_l2_mean=moment(metric['final'])['mean'],
                 final_l2_std=moment(metric['final'])['std'],
                 final_l2_median=moment(metric['final'])['median'],
                 inference_seconds_70=sec,
                 inference_seconds_per_trajectory=sec/len(test_ids),
                 peak_inference_gpu_mib=peak,
                 **{k+'_'+z:vv for k,v in diagnostic.items() for z,vv in v.items()})
        plot_error_curve(folder,metric,'test')
        plot_case(folder/'test_vorticity_0.png',pred,true,0,f'{record["method"]} N{record["N_train"]} TEST case 230')
        plot_case(folder/'test_vorticity_35.png',pred,true,35,f'{record["method"]} N{record["N_train"]} TEST case 265')
        atomic_json(finish,row)
        test_rows.append(row)
        summarize_tables(args.output,test_rows,from_test=True)
        plot_size_summary(args.output,test_rows)
        plot_cross_model_figures(args.output,test_rows)
        emit(f'[TEST COMPLETE] {j}/{len(locked)} {record["method"]} N{record["N_train"]} '
             f'full={row["full_l2_mean"]:.5f}, final={row["final_l2_mean"]:.5f}, '
             f'inference_70={sec:.2f}s')
        del pred,true,m
        if device.type=='cuda':torch.cuda.empty_cache()
    return test_rows


def smoke_test(args,device):
    """CPU/GPU test of every method, gradients, temporal basis and quadrature."""
    seed_all(args.seed)
    target=OperatorModel('AWFNO',args.width,args.modes,args.depth,args.window_k,
                         args.basis_rank,args.order,args.radius)
    ref_real=real_dof(target)
    pm_width,pm_n,gap=match_wfno_width(ref_real,args.modes,args.depth,args.window_k)
    emit(f'[PARAMETER MATCH] AWFNO real DOF={ref_real:,}; '
         f'WFNO-PM width={pm_width} real DOF={pm_n:,}; relative gap={100*gap:.2f}%')
    del target
    X=torch.randn(2,1,NX,NX,device=device)
    target_seq=torch.randn(2,args.window_k,NX,NX,device=device)
    basis=torch.linalg.qr(torch.randn(args.window_k,args.basis_rank,device=device),mode='reduced').Q
    for method in MODELS:
        model=make_model(method,args,pm_width).to(device)
        if method=='TB-AWFNO':model.temporal_basis.copy_(basis)
        target=target_seq if method!='FNO' else torch.randn(2,NT-1,NX,NX,device=device)
        out=model(X)
        assert out.shape==target.shape and bool(torch.isfinite(out).all()),method
        loss=F.mse_loss(out,target)
        loss.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        emit(f'[SMOKE] {method} | shape={tuple(out.shape)} | '
             f'params={param_count(model):,} | real DoF={real_dof(model):,} | gradients PASS')
        del model,out,loss
        if device.type=='cuda':torch.cuda.empty_cache()
    # Independently test the controlled component/order ablation kernels too.
    for order in ('LGL','LLG','GGG','LLL'):
        ablation=OperatorModel('AWFNO',args.width,args.modes,args.depth,
                              args.window_k,args.basis_rank,order,args.radius).to(device)
        out=ablation(X[:1])
        assert out.shape==(1,args.window_k,NX,NX) and bool(torch.isfinite(out).all()),order
        out.mean().backward()
        assert any(p.grad is not None for p in ablation.parameters())
        emit(f'[SMOKE ABLATION] {order} | params={param_count(ablation):,} | gradients PASS')
        del ablation,out
        if device.type=='cuda':torch.cuda.empty_cache()
    clean=np.ones((1,NT,NX,NX),dtype=np.float32)
    m=l2_metrics(clean,clean)
    assert m['full'].max()==0. and m['final'].max()==0.
    altered=clean.copy();altered[:,1:]+=1
    q=l2_metrics(altered,clean)
    expected=math.sqrt(LX*LX*(NT-1.5)*DT)
    assert np.allclose(q['full'],expected,rtol=1e-6)
    assert np.allclose(q['final'],LX,rtol=1e-6)
    emit('[METRIC] Physical dxdy/dt quadrature analytical tests PASS')
    return pm_width,gap


def supplementary_ablation_plan(args):
    """Fresh controlled ablations from THIS clean implementation (not the older 22)."""
    from copy import copy
    base=dict(method='AWFNO',ntrain=100,order='GLL',window_k=20,
              modes=16,depth=4,radius=2,basis_rank=12,seed=args.seed)
    jobs=[]
    def add(family,**changes):
        cfg={**base,**changes}
        cfg['family']=family
        jobs.append(cfg)
    for order in ('LGL','LLG','GGG','LLL'):
        add('ordering_and_components',order=order)
    for K in (2,5,10):
        add('temporal_window_K',window_k=K,basis_rank=min(12,K))
    for modes in (8,24,32):
        add('Fourier_modes',modes=modes)
    for radius in (1,3):
        add('local_radius',radius=radius)
    for depth in (2,6):
        add('network_depth',depth=depth)
    for rank in (8,16):
        add('temporal_basis_rank',method='TB-AWFNO',basis_rank=rank)
    for seed in (2026,7727):
        for order in ('GLL','LGL'):
            add('repeat_seed',order=order,seed=seed)
    assert len(jobs)==20
    assert len({tuple(j[x] for x in ('method','ntrain','order','window_k',
                        'modes','depth','radius','basis_rank','seed')) for j in jobs})==len(jobs)
    return jobs


def do_supplementary_ablations(U,args,pm_width,pm_gap,device):
    from copy import copy
    output=args.output/'independent_ablations'
    output.mkdir(parents=True,exist_ok=True)
    jobs=supplementary_ablation_plan(args)
    manifest=output/'predeclared_controlled_studies.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=jobs:
            raise RuntimeError('Controlled ablation plan has changed')
    else:atomic_json(manifest,jobs)
    results=[]
    for i,job in enumerate(jobs,1):
        if time.perf_counter()-args.session_start>args.session_minutes*60:
            emit('[ABLATON SESSION BUDGET] Resume Colab to continue saved studies')
            break
        cfg=copy(args)
        cfg.output=output
        for key in ('seed','order','window_k','modes','depth','radius','basis_rank'):
            setattr(cfg,key,job[key])
        emit(f'[CLEAN ABLATION {i}/{len(jobs)}] {job["family"]}: '
             f'{job["method"]} order={job["order"]} K{job["window_k"]} '
             f'M{job["modes"]} D{job["depth"]} r{job["radius"]} '
             f'rank{job["basis_rank"]} seed{job["seed"]}')
        try:
            row=fit_one(U,cfg,job['ntrain'],job['method'],pm_width,pm_gap,device)
        except SessionBudgetExceeded:
            break
        results.append(dict(row,family=job['family'],order=job['order'],
                            modes=job['modes'],radius=job['radius'],
                            depth=job['depth'],basis_rank=job['basis_rank']))
    # Rebuild the cumulative table, not merely this session's completed runs.
    results=[]
    for job in jobs:
        cfg=copy(args)
        cfg.output=output
        for key in ('seed','order','window_k','modes','depth','radius','basis_rank'):
            setattr(cfg,key,job[key])
        m=make_model(job['method'],cfg,pm_width)
        config=record_config(cfg,m,job['ntrain'],job['method'],pm_width,pm_gap)
        file=model_folder(output,config)/'training_result.json'
        if file.exists():
            row=json.loads(file.read_text())
            results.append(dict(row,family=job['family'],order=job['order'],
                                modes=job['modes'],radius=job['radius'],
                                depth=job['depth'],basis_rank=job['basis_rank']))
    if results:
        import pandas as pd
        df=pd.DataFrame(results)
        df.to_csv(output/'clean_ablation_results.csv',index=False)
        df.to_json(output/'clean_ablation_results.json',orient='records',indent=2)
        # Join the reference AWFNO/TB-AWFNO N100 configurations for readable tables.
        reference_rows=[]
        for method in ('AWFNO','TB-AWFNO'):
            mm=make_model(method,args,pm_width)
            cc=record_config(args,mm,100,method,pm_width,pm_gap)
            fp=model_folder(args.output,cc)/'training_result.json'
            if fp.is_file():
                reference_rows.append(dict(json.loads(fp.read_text()),
                                           family='reference',order='GLL',
                                           modes=16,radius=2,depth=4,basis_rank=12))
        pd.DataFrame(reference_rows+results).to_csv(output/'clean_ablation_table_with_reference.csv',index=False)
        import matplotlib.pyplot as plt
        kinds=['ordering_and_components','temporal_window_K','Fourier_modes',
               'local_radius','network_depth','temporal_basis_rank','repeat_seed']
        baseline_rows={}
        for base_method in ('AWFNO','TB-AWFNO'):
            m=make_model(base_method,args,pm_width)
            c=record_config(args,m,100,base_method,pm_width,pm_gap)
            file=model_folder(args.output,c)/'training_result.json'
            if file.is_file():baseline_rows[base_method]=json.loads(file.read_text())
        fig,axs=plt.subplots(4,2,figsize=(15,16),constrained_layout=True)
        for ax,kind in zip(axs.flat,kinds):
            subset=[q for q in results if q['family']==kind]
            bmethod='TB-AWFNO' if kind=='temporal_basis_rank' else 'AWFNO'
            if bmethod in baseline_rows:
                reference=baseline_rows[bmethod]
                subset=[dict(reference,order='GLL',modes=16,radius=2,depth=4,
                             basis_rank=12)]+subset
            if kind=='repeat_seed':
                subset.extend(q for q in results
                              if q['family']=='ordering_and_components' and q['order']=='LGL')
            if kind=='ordering_and_components': labels=[q['order'] for q in subset]
            elif kind=='temporal_window_K':labels=[f"K{q['K']}" for q in subset]
            elif kind=='Fourier_modes':labels=[f"M{q['modes']}" for q in subset]
            elif kind=='local_radius':labels=[f"r{q['radius']}" for q in subset]
            elif kind=='network_depth':labels=[f"D{q['depth']}" for q in subset]
            elif kind=='temporal_basis_rank':labels=[f"rank{q['basis_rank']}" for q in subset]
            else: labels=[f"{q['order']} s{q['seed']}" for q in subset]
            ax.bar(labels,[q['best_validation_full_l2'] for q in subset])
            ax.set(title=kind,ylabel='Mean validation full physical L2')
            ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
        axs.flat[-1].axis('off')
        fig.savefig(output/'clean_all_ablation_comparison.png',dpi=140)
        plt.close(fig)
    emit(f'[INDEPENDENT ABLATION STATUS] {len(results)}/{len(jobs)} completed; validation only.')
    return len(results)==len(jobs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--methods',nargs='+',default=list(MODELS),choices=MODELS)
    p.add_argument('--sizes',type=int,nargs='+',default=list(SIZES))
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--width',type=int,default=32)
    p.add_argument('--modes',type=int,default=16)
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--radius',type=int,default=2)
    p.add_argument('--order',choices=('GLL','LGL','LLG'),default='GLL')
    p.add_argument('--window-k',type=int,default=20)
    p.add_argument('--basis-rank',type=int,default=12)
    p.add_argument('--window-stride',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--eval-batch',type=int,default=3)
    p.add_argument('--max-epochs',type=int,default=45)
    p.add_argument('--min-epochs',type=int,default=12)
    p.add_argument('--patience',type=int,default=6)
    p.add_argument('--val-every',type=int,default=3)
    p.add_argument('--checkpoint-every',type=int,default=3)
    p.add_argument('--min-delta',type=float,default=1e-4)
    p.add_argument('--lr',type=float,default=0.0007)
    p.add_argument('--wd',type=float,default=1e-6)
    p.add_argument('--session-minutes',type=float,default=180)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--train-only',action='store_true')
    p.add_argument('--suite',choices=('full','comparison'),default='full')
    p.add_argument('--evaluate-test-only',action='store_true')
    p.add_argument('--verify-file-hash',action='store_true')
    p.add_argument('--force',action='store_true')
    p.add_argument('--force-test',action='store_true')
    args=p.parse_args()
    if args.train_only and args.evaluate_test_only:raise ValueError('Conflicting modes')
    if args.window_stride<1 or args.batch_size<1 or args.val_every<1:raise ValueError('Invalid batch/stride')
    if not (1<=args.basis_rank<=args.window_k<NT):raise ValueError('Invalid K/basis rank')
    if args.min_epochs>args.max_epochs:raise ValueError('Invalid min epochs')
    if not (set(args.methods)==set(MODELS) and set(args.sizes)==set(SIZES)) and not args.smoke_only:
        emit('[NOTE] Partial matrix requested: all requested runs must finish before requested test evaluation')
    args.output.mkdir(parents=True,exist_ok=True)
    args.session_start=time.perf_counter()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emit(f'[DEVICE] {device} | PyTorch {torch.__version__} | implementation {VERSION}')
    pm_width,pm_gap=smoke_test(args,device)
    if args.smoke_only:
        emit('[SMOKE ONLY] Did not open dataset, train or inspect test trajectories.')
        return
    U=load_source(args.dataset,verify_hash=args.verify_file_hash)
    emit(f'[DATASET] {tuple(U.shape)} | data SHA={EXPECTED_DATA_SHA} | '
         f'sealed 70 test trajectories')
    plan=expected_configs(args,pm_width,pm_gap)
    planfile=args.output/'predeclared_model_comparison.json'
    if planfile.exists():
        if json.loads(planfile.read_text())!=plan:
            raise RuntimeError('Existing plan differs. Use new output folder')
    else:atomic_json(planfile,plan)
    comparisons_complete=False
    ablations_complete=(args.suite=='comparison')
    if not args.evaluate_test_only:
        finished=[]
        for i,job in enumerate(plan,1):
            if time.perf_counter()-args.session_start>args.session_minutes*60:
                emit('[SESSION BUDGET] Rerun notebook to resume at next model')
                break
            emit(f'[EXPERIMENT {i}/{len(plan)}] {job["method"]} N{job["N_train"]}')
            try:
                row=fit_one(U,args,job['N_train'],job['method'],pm_width,pm_gap,device)
                finished.append(row)
            except SessionBudgetExceeded:
                break
            summarize_tables(args.output,finished,from_test=False)
        # Rebuild summary from all saved runs, including completed prior sessions.
        finished=[json.loads((Path(j['path'])/'training_result.json').read_text())
                  for j in plan if (Path(j['path'])/'training_result.json').exists()]
        summarize_tables(args.output,finished,from_test=False)
        comparisons_complete=(len(finished)==len(plan))
        emit(f'[TRAINING STATUS] {len(finished)}/{len(plan)} method comparisons completed; test not accessed')
        if comparisons_complete and args.suite=='full':
            ablations_complete=do_supplementary_ablations(U,args,pm_width,pm_gap,device)
        if args.train_only:return
    else:
        comparisons_complete=all((Path(j['path'])/'training_result.json').exists() for j in plan)
        if args.suite=='full':
            # Independent ablations are required for the predeclared full suite.
            folder=args.output/'independent_ablations'
            data=folder/'clean_ablation_results.json'
            ablations_complete=(data.is_file() and len(json.loads(data.read_text()))==20)
    if comparisons_complete and ablations_complete:
        emit('[TEST LOCK] All clean comparison and ablation configurations complete. Evaluating 70 sealed test trajectories.')
        evaluate_test(U,args,plan,device)
    else:
        emit('[TEST SEALED] Incomplete clean full suite. Rerun same notebook to resume.')


if __name__=='__main__':
    main()
