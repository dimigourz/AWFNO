#!/usr/bin/env python3
"""Publication-oriented annular adaptations of existing reference architectures.

F-FNO: FourierFlow/Cahn–Hilliard-style depth-24 factorized Fourier layer,
       Fourier multipliers shared across depth, residual backcast FFN.
CNO: official-simplified CNO 2D encoder/decoder, anti-aliased bicubic
     LeakyReLU stages, residual subnetworks, skipped scales, rectangular grid.
U-Net: four-level Cahn–Hilliard baseline with group norm and transpose upconv.
U-NO: library neuralop.models.UNO, rather than an ad-hoc spectral U-Net.

Temporal head is K=30 for ALL models because the paper's annular comparison
fixes the same input-to-output mapping. Architectures are *adaptations*, never
represented as source-identical copies of the original authors' training runs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def polar_coordinates(r, theta):
    r = torch.as_tensor(r, dtype=torch.float32)
    theta = torch.as_tensor(theta, dtype=torch.float32)
    rr = (r-r.min()) / (r.max()-r.min()+1.e-12)
    rr,tt = torch.meshgrid(rr,theta,indexing="ij")
    return torch.stack((rr,torch.sin(tt),torch.cos(tt)),dim=0).unsqueeze(0)


class StateAndCoordinates(nn.Module):
    def __init__(self, r, theta):
        super().__init__()
        self.register_buffer("grid",polar_coordinates(r,theta),persistent=False)
    def forward(self,u):
        assert u.ndim==3
        return torch.cat((u[:,None],self.grid.expand(u.shape[0],-1,-1,-1)),dim=1)


class WNLinear(nn.Module):
    """FourierFlow-style weight-normalized linear projection."""
    def __init__(self, ci, co):
        super().__init__()
        self.layer = nn.utils.parametrizations.weight_norm(nn.Linear(ci, co))
    def forward(self, x):
        return self.layer(x)


class ReferenceFFNOSpectralLayer(nn.Module):
    """FourierFlow 2D separable Fourier layer, with a pointwise backcast network."""
    def __init__(self,width,factor=4):
        super().__init__()
        self.backcast = nn.Sequential(WNLinear(width,width*factor),nn.ReLU(),
                                      WNLinear(width*factor,width))
    def forward(self,x,weight_r,weight_theta):
        # x: (B,R,Theta,C). Spectral weights are shared across all layers.
        yc=x.permute(0,3,1,2).contiguous()
        B,C,R,T=yc.shape
        fr=torch.fft.rfft(yc,dim=-2,norm="ortho")
        nr=min(R//2+1,weight_r.shape[2])
        yr=torch.zeros((B,C,R//2+1,T),dtype=fr.dtype,device=fr.device)
        yr[:,:,:nr,:]=torch.einsum("birt,ior->bort",fr[:,:,:nr,:],weight_r[:,:,:nr])
        yr=torch.fft.irfft(yr,n=R,dim=-2,norm="ortho")
        ft=torch.fft.rfft(yc,dim=-1,norm="ortho")
        nt=min(T//2+1,weight_theta.shape[2])
        yt=torch.zeros((B,C,R,T//2+1),dtype=ft.dtype,device=ft.device)
        yt[:,:,:,:nt]=torch.einsum("birm,iom->borm",ft[:,:,:,:nt],weight_theta[:,:,:nt])
        yt=torch.fft.irfft(yt,n=T,dim=-1,norm="ortho")
        return self.backcast((yr+yt).permute(0,2,3,1).contiguous())


class FFNO2D(nn.Module):
    """2D F-FNO close to FourierFlow and existing Cahn–Hilliard FFNO2D_AR.

    In contrast to Cahn–Hilliard's one-step head, its output has K channels.
    """
    def __init__(self,k,r,theta,width=64,depth=24,modes_r=16,modes_theta=16):
        super().__init__()
        self.coords=StateAndCoordinates(r,theta)
        self.in_proj=WNLinear(4,width)
        # Weight sharing across all 24 Fourier layers is deliberate and matches
        # the reference FourierFlow-style residual factorized architecture.
        self.weight_r=nn.Parameter(torch.view_as_complex(
            torch.nn.init.xavier_normal_(torch.empty(width,width,modes_r,2),gain=.1).contiguous()))
        self.weight_theta=nn.Parameter(torch.view_as_complex(
            torch.nn.init.xavier_normal_(torch.empty(width,width,modes_theta,2),gain=.1).contiguous()))
        self.layers=nn.ModuleList([ReferenceFFNOSpectralLayer(width) for _ in range(depth)])
        self.head=nn.Sequential(WNLinear(width,128),WNLinear(128,k))
    def forward(self,u):
        with torch.autocast(device_type=u.device.type,enabled=False):
            x=self.coords(u.float()).permute(0,2,3,1)
            x=self.in_proj(x)
            backcast=None
            for layer in self.layers:
                backcast=layer(x,self.weight_r,self.weight_theta)
                x=x+backcast
            return self.head(backcast)


def polar_pad(x,n):
    if n<1:return x
    # Periodic azimuth, nonperiodic radial ends. Radial replication follows
    # the actual annulus mesh rather than artificially periodicizing radius.
    x=F.pad(x,(n,n,0,0),mode="circular")
    return F.pad(x,(0,0,n,n),mode="replicate")


class PolarConv(nn.Module):
    def __init__(self,ci,co,kernel_size=3):
        super().__init__()
        self.radius=kernel_size//2
        self.conv=nn.Conv2d(ci,co,kernel_size)
    def forward(self,x):
        return self.conv(polar_pad(x,self.radius))


class CNOLReLU(nn.Module):
    """Bicubic, antialiased upsampling -> activation -> antialiased resampling.

    Mirrors the CNO2d_simplified CNO_LReLU scheme used for Cahn–Hilliard.
    """
    def __init__(self,in_size,out_size):
        super().__init__()
        self.in_size=in_size
        self.out_size=out_size
        self.act=nn.LeakyReLU()
    def forward(self,x):
        z=F.interpolate(x,size=(2*self.in_size[0],2*self.in_size[1]),
                        mode="bicubic",align_corners=False,antialias=True)
        z=self.act(z)
        return F.interpolate(z,size=self.out_size,mode="bicubic",
                             align_corners=False,antialias=True)


class CNOBlock(nn.Module):
    def __init__(self,ci,co,size_in,size_out,use_bn=True):
        super().__init__()
        self.conv=PolarConv(ci,co,3)
        self.bn=nn.BatchNorm2d(co) if use_bn else nn.Identity()
        self.act=CNOLReLU(size_in,size_out)
    def forward(self,x):return self.act(self.bn(self.conv(x)))


class CNOLiftProject(nn.Module):
    def __init__(self,ci,co,size,latent=64):
        super().__init__()
        self.hidden=CNOBlock(ci,latent,size,size,False)
        self.project=PolarConv(latent,co,3)
    def forward(self,x):return self.project(self.hidden(x))


class CNOResidualBlock(nn.Module):
    def __init__(self,c,size,use_bn=True):
        super().__init__()
        self.c1=PolarConv(c,c,3);self.c2=PolarConv(c,c,3)
        self.bn1=nn.BatchNorm2d(c) if use_bn else nn.Identity()
        self.bn2=nn.BatchNorm2d(c) if use_bn else nn.Identity()
        self.act=CNOLReLU(size,size)
    def forward(self,x):
        return x+self.bn2(self.c2(self.act(self.bn1(self.c1(x)))))


class CNOResNet(nn.Module):
    def __init__(self,c,size,n,use_bn=True):
        super().__init__()
        self.blocks=nn.Sequential(*[CNOResidualBlock(c,size,use_bn) for _ in range(n)])
    def forward(self,x):return self.blocks(x)


class CNO2D(nn.Module):
    """Reference-style simplified CNO encoder/decoder, rectangular annulus.

    Carries 3 levels, 3 residual blocks/scale + 3 bottleneck blocks,
    skipped scales and CNO antialiased nonlinearities as in Cahn–Hilliard.
    The output is K frames rather than the paper's direct-time scalar.
    """
    def __init__(self,k,r,theta,channel_multiplier=32,n_layers=3,res_blocks=3,neck_blocks=3):
        super().__init__()
        nr,nt=len(r),len(theta)
        if nr%(2**n_layers)!=0 or nt%(2**n_layers)!=0:
            raise ValueError("CNO input dimensions must be divisible by 2**n_layers")
        self.coords=StateAndCoordinates(r,theta)
        self.n_layers=n_layers
        sizes=[(nr//(2**i),nt//(2**i)) for i in range(n_layers+1)]
        enc=[channel_multiplier//2]+[(2**i)*channel_multiplier for i in range(n_layers)]
        decoder_in=list(reversed(enc[1:]))
        decoder_out=list(reversed(enc[:-1]))
        for i in range(1,n_layers):decoder_in[i]*=2
        self.lift=CNOLiftProject(4,enc[0],sizes[0])
        self.project=CNOLiftProject(enc[0]+decoder_out[-1],k,sizes[0])
        self.encoder=nn.ModuleList([
            CNOBlock(enc[i],enc[i+1],sizes[i],sizes[i+1],True)
            for i in range(n_layers)])
        self.skip_expansion=nn.ModuleList([
            CNOBlock(enc[i],enc[i],sizes[i],sizes[i],True)
            for i in range(n_layers+1)])
        self.decoder=nn.ModuleList([
            CNOBlock(decoder_in[i],decoder_out[i],sizes[n_layers-i],sizes[n_layers-i-1],True)
            for i in range(n_layers)])
        self.res_nets=nn.ModuleList([
            CNOResNet(enc[i],sizes[i],res_blocks,True)
            for i in range(n_layers)])
        self.neck=CNOResNet(enc[n_layers],sizes[n_layers],neck_blocks,True)
    def forward(self,u):
        x=self.lift(self.coords(u))
        skipped=[]
        for i in range(self.n_layers):
            skipped.append(self.res_nets[i](x))
            x=self.encoder[i](x)
        x=self.neck(x)
        for i in range(self.n_layers):
            if i==0:
                x=self.skip_expansion[self.n_layers](x)
            else:
                x=torch.cat((x,self.skip_expansion[self.n_layers-i](skipped[-i])),dim=1)
            x=self.decoder[i](x)
        x=torch.cat((x,self.skip_expansion[0](skipped[0])),dim=1)
        return self.project(x).permute(0,2,3,1).contiguous()


def _group_count(c):
    for n in (8,4,2,1):
        if c%n==0:return n
    return 1


class UNetBlock(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.block=nn.Sequential(
            PolarConv(ci,co),nn.GroupNorm(_group_count(co),co),nn.GELU(),
            PolarConv(co,co),nn.GroupNorm(_group_count(co),co),nn.GELU())
    def forward(self,x):return self.block(x)


class UNet2D(nn.Module):
    """Four-level U-Net based on the Cahn–Hilliard baseline (K-output head)."""
    def __init__(self,k,r,theta,channels=32):
        super().__init__()
        self.coords=StateAndCoordinates(r,theta)
        n=[channels,2*channels,4*channels,8*channels,16*channels]
        self.encoder=nn.ModuleList([UNetBlock(ci,co)
            for ci,co in zip([4]+n[:3],n[:4])])
        self.pool=nn.MaxPool2d(2)
        self.bottleneck=UNetBlock(n[3],n[4])
        self.ups=nn.ModuleList([nn.ConvTranspose2d(n[i+1],n[i],2,2)
                                 for i in (3,2,1,0)])
        self.decoder=nn.ModuleList([UNetBlock(2*n[i],n[i])
                                    for i in (3,2,1,0)])
        self.head=nn.Conv2d(n[0],k,1)
    def forward(self,u):
        x=self.coords(u)
        skips=[]
        for enc in self.encoder:
            x=enc(x);skips.append(x);x=self.pool(x)
        x=self.bottleneck(x)
        for i,(up,dec) in enumerate(zip(self.ups,self.decoder)):
            x=up(x)
            x=torch.cat((x,skips[-1-i]),dim=1)
            x=dec(x)
        return self.head(x).permute(0,2,3,1).contiguous()


class UNO2D(nn.Module):
    """Official neuraloperator UNO class with annular coordinate channels."""
    def __init__(self,k,r,theta,width=32):
        super().__init__()
        from neuralop.models import UNO as NeuralopUNO
        self.coords=StateAndCoordinates(r,theta)
        self.network=NeuralopUNO(
            # neuraloperator 2.0.0 defaults to channel_mlp_skip='soft-gating',
            # which raises when 32 -> 64 -> 128 channel widths change.
            # Use learnable linear projections on channel-changing skips.
            fno_skip="linear",
            channel_mlp_skip="linear",
            in_channels=4,out_channels=k,hidden_channels=width,
            lifting_channels=64,projection_channels=64,
            positional_embedding=None,
            n_layers=5,
            uno_out_channels=[width,2*width,4*width,2*width,width],
            uno_n_modes=[[16,24],[12,20],[6,12],[12,20],[16,24]],
            uno_scalings=[[1.,1.],[.5,.5],[.5,.5],[2.,2.],[2.,2.]],
            horizontal_skips_map={4:0,3:1})
    def forward(self,u):
        return self.network(self.coords(u)).permute(0,2,3,1).contiguous()


def make_reference_model(name,k,r,theta,width=32):
    if name=="FFNO":return FFNO2D(k,r,theta)
    if name=="CNO":return CNO2D(k,r,theta)
    if name=="UNet":return UNet2D(k,r,theta)
    if name=="UNO":return UNO2D(k,r,theta,width=width)
    raise ValueError(name)
