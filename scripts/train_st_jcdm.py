from __future__ import annotations
import argparse,json,random,sys,torch,numpy as np
from pathlib import Path
from torch.utils.data import DataLoader,TensorDataset
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src')]
from stde_cdm import STJCDM,load_joint

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=500);p.add_argument('--output',type=Path);a=p.parse_args()
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 d=load_joint(R/'data/wind_data_all_zone.csv');m=STJCDM().to(dev);loader=DataLoader(TensorDataset(torch.from_numpy(d.x_train),torch.from_numpy(d.y_train)),50,shuffle=True,generator=torch.Generator().manual_seed(a.seed));xv=torch.from_numpy(d.x_validation).to(dev);yv=torch.from_numpy(d.y_validation).to(dev);out=a.output or R/f'outputs/checkpoints/st_jcdm_z1-5_seed{a.seed}';out.mkdir(parents=True,exist_ok=True);history={'embedding':[],'diffusion':[]}
 opt=torch.optim.Adam(m.encoder.parameters(),1e-3);best=float('inf')
 for ep in range(1,a.epochs+1):
  m.train();vals=[]
  for x,y in loader:
   x,y=x.to(dev),y.to(dev);opt.zero_grad(set_to_none=True);loss,_=m.embedding_loss(x,y);loss.backward();opt.step();vals.append(float(loss.detach()))
  m.eval()
  with torch.no_grad():v,parts=m.embedding_loss(xv,yv);v=float(v)
  row={'epoch':ep,'train':float(np.mean(vals)),'validation':v,'point':float(parts['point']),'aggregate':float(parts['aggregate'])};history['embedding'].append(row)
  if v<best:best=v;torch.save(m.encoder.state_dict(),out/'best_encoder.pt')
  if ep==1 or ep%50==0:print('embed',json.dumps(row),flush=True)
 m.encoder.load_state_dict(torch.load(out/'best_encoder.pt',weights_only=True));m.encoder.requires_grad_(False);opt=torch.optim.Adam(m.denoiser.parameters(),1e-3);best_diff=float('inf');g=torch.Generator(device=dev).manual_seed(12345);steps=torch.randint(0,50,(50,),device=dev,generator=g);noise=torch.randn(yv.shape,device=dev,generator=g)
 for ep in range(1,a.epochs+1):
  m.train();vals=[]
  for x,y in loader:
   x,y=x.to(dev),y.to(dev);opt.zero_grad(set_to_none=True);loss=m.diffusion_loss(x,y);loss.backward();opt.step();vals.append(float(loss.detach()))
  m.eval()
  with torch.no_grad():v=float(m.diffusion_loss(xv,yv,steps,noise))
  row={'epoch':ep,'train':float(np.mean(vals)),'validation':v};history['diffusion'].append(row)
  if v<best_diff:best_diff=v;torch.save(m.denoiser.state_dict(),out/'best_denoiser.pt')
  if ep==1 or ep%50==0:print('diff',json.dumps(row),flush=True)
 m.denoiser.load_state_dict(torch.load(out/'best_denoiser.pt',weights_only=True));torch.save({'state_dict':m.state_dict(),'config':{'farms':5,'channels':64,'layers':3,'steps':50,'beta_start':1e-4,'beta_end':.05},'seed':a.seed,'best_embedding':best,'best_diffusion':best_diff,'train_dates':d.train_dates,'validation_dates':d.validation_dates,'test_dates':d.test_dates},out/'st_jcdm.pt');(out/'history.json').write_text(json.dumps(history,indent=2));print(f'saved={out / "st_jcdm.pt"} embed={best:.6f} diff={best_diff:.6f}',flush=True)
if __name__=='__main__':main()
