from __future__ import annotations
import torch
from torch import nn
from cldm.model import EmbeddingNetwork,StepEmbedding,DilatedResidualBlock

class JointDenoiser(nn.Module):
    def __init__(self,farms=5,channels=64,layers=3,steps=50):
        super().__init__();self.err=nn.Conv1d(farms,channels,1);self.cond=nn.Conv1d(farms,channels,1);self.step=StepEmbedding(steps,channels)
        self.blocks=nn.ModuleList(DilatedResidualBlock(channels,2**i) for i in range(layers));self.out=nn.Sequential(nn.LeakyReLU(.2),nn.Conv1d(channels,channels,1),nn.LeakyReLU(.2),nn.Conv1d(channels,farms,1))
    def forward(self,e,t,f):
        h=self.err(e.transpose(1,2))-self.cond(f.transpose(1,2))+self.step(t);sk=[]
        for b in self.blocks:h,s=b(h);sk.append(s)
        return self.out(torch.stack(sk).sum(0)).transpose(1,2)

class JointCLDM(nn.Module):
    def __init__(self,farms=5,channels=64,layers=3,steps=50,beta_start=1e-4,beta_end=.05):
        super().__init__();self.farms=farms;self.steps=steps
        self.embeddings=nn.ModuleList(EmbeddingNetwork(4,channels,3) for _ in range(farms));self.denoiser=JointDenoiser(farms,channels,layers,steps)
        b=torch.linspace(beta_start,beta_end,steps);a=1-b;ab=torch.cumprod(a,0);prev=torch.cat([torch.ones(1),ab[:-1]]);pv=b*(1-prev)/(1-ab)
        for n,v in [('betas',b),('alphas',a),('alpha_bars',ab),('posterior_variance',pv.clamp_min(1e-20))]:self.register_buffer(n,v)
    def forecast(self,x):return torch.stack([m(x[:,:,i,:]) for i,m in enumerate(self.embeddings)],dim=2)
    def loss(self,x,y,step=None,noise=None):
        with torch.no_grad():f=self.forecast(x)
        clean=y-f
        if step is None:step=torch.randint(0,self.steps,(len(y),),device=y.device)
        if noise is None:noise=torch.randn_like(clean)
        ab=self.alpha_bars[step,None,None];noisy=ab.sqrt()*clean+(1-ab).sqrt()*noise
        return nn.functional.mse_loss(self.denoiser(noisy,step,f),noise)
    @torch.no_grad()
    def sample(self,x,n):
        f=self.forecast(x);b,t,k=f.shape;f=f[:,None].expand(b,n,t,k).reshape(-1,t,k);xx=x[:,None].expand(b,n,*x.shape[1:]).reshape(-1,*x.shape[1:]);e=torch.randn_like(f)
        for i in reversed(range(self.steps)):
            s=torch.full((len(e),),i,device=e.device,dtype=torch.long);p=self.denoiser(e,s,f);a=self.alphas[i];ab=self.alpha_bars[i];e=(e-(1-a)/(1-ab).sqrt()*p)/a.sqrt()
            if i:e=e+self.posterior_variance[i].sqrt()*torch.randn_like(e)
        return (f+e).clamp(0,1).reshape(b,n,t,k)
