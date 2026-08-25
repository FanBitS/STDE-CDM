from __future__ import annotations
import argparse,json,sys,time,torch,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src')]
from cldm.metrics import scenario_metrics
from stde_cdm import (JointVAE,JointWGANGenerator,JointDDPM,build_joint_umnn,
                       JointCLDM,STJCDM,load_joint)

def metrics(s,y):return scenario_metrics(s.reshape(len(y),s.shape[1],-1),y.reshape(len(y),-1))
def inverse_scaled(generated,ck,n=200):
 s=generated.reshape(50,n,-1)*ck['y_std'][None,None,:]+ck['y_mean'][None,None,:]
 return np.clip(s.reshape(50,n,24,5),0,1)
def scaled_condition(values,ck):
 x=(values.reshape(len(values),-1)-ck['x_mean'])/ck['x_std']
 return torch.from_numpy(x.astype(np.float32))
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--seeds',type=int,nargs='+',default=list(range(10)),
                help='Training seeds to evaluate; all paper seeds are packaged')
 p.add_argument('--scenarios',type=int,default=200)
 p.add_argument('--output',type=Path,default=R/'outputs/metrics/all_joint_models_locked_test.json')
 a=p.parse_args()
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');n=a.scenarios
 d=load_joint(R/'data/wind_data_all_zone.csv');raw=torch.from_numpy(d.x_test).to(dev);rows=[]
 for seed in a.seeds:
  started=time.perf_counter();models={}
  vae_path=R/f'artifacts/checkpoints/joint_vae_seed{seed}.pt';ck=torch.load(vae_path,map_location=dev,weights_only=False);m=JointVAE(latent_size=ck['config']['latent_size'],hidden_size=ck['config']['hidden_size'],hidden_layers=ck['config']['hidden_layers']).to(dev);m.load_state_dict(ck['state_dict']);m.eval();g=torch.Generator(device=dev).manual_seed(81000+seed);models['Joint-VAE']=inverse_scaled(m.sample(scaled_condition(d.x_test,ck).to(dev),n,g).cpu().numpy(),ck,n)
  wgan_path=R/f'artifacts/checkpoints/joint_wgan_gp_seed{seed}.pt';ck=torch.load(wgan_path,map_location=dev,weights_only=False);m=JointWGANGenerator(latent_size=ck['config']['latent_size'],width=ck['config']['width'],layers=ck['config']['layers']).to(dev);m.load_state_dict(ck['generator_state_dict']);m.eval();g=torch.Generator(device=dev).manual_seed(82000+seed);models['Joint-WGAN-GP']=inverse_scaled(m.sample(scaled_condition(d.x_test,ck).to(dev),n,g).cpu().numpy(),ck,n)
  ddpm_path=R/f'artifacts/checkpoints/joint_ddpm_seed{seed}.pt';ck=torch.load(ddpm_path,map_location=dev,weights_only=False);config={k:v for k,v in ck['config'].items() if k not in {'epochs','batch_size','seed'}};m=JointDDPM(**config).to(dev);m.load_state_dict(ck['state_dict']);m.eval();g=torch.Generator(device=dev).manual_seed(83000+seed);models['Joint-DDPM']=m.sample(raw,n,g).cpu().numpy()
  umnn_path=R/f'artifacts/checkpoints/joint_umnn_seed{seed}.pt';ck=torch.load(umnn_path,map_location=dev,weights_only=False);m=build_joint_umnn().to(dev);m.load_state_dict(ck['state_dict']);m.eval();condition=scaled_condition(d.x_test,ck).to(dev);generated=[];torch.manual_seed(84000+seed)
  with torch.no_grad():
   for day in range(50):generated.append(m.invert(torch.randn(n,120,device=dev),condition[day:day+1].expand(n,-1)).cpu().numpy())
  models['Joint-UMNN']=inverse_scaled(np.stack(generated),ck,n)
  bp=R/f'artifacts/checkpoints/joint_cldm_seed{seed}.pt';bc=torch.load(bp,map_location=dev,weights_only=False);base=JointCLDM().to(dev);base.load_state_dict(bc['state_dict']);base.eval();torch.manual_seed(85000+seed);bs=base.sample(raw,n).cpu().numpy();models['Joint-CLDM']=bs
  st_path=R/f'artifacts/checkpoints/stde_spatiotemporal_seed{seed}.pt';sc=torch.load(st_path,map_location=dev,weights_only=False);st=STJCDM(**sc['config']).to(dev);st.load_state_dict(sc['state_dict']);st.eval();g=torch.Generator(device=dev).manual_seed(85000+seed);ss=st.sample(raw,n,g).cpu().numpy();models['STDE-CDM']=.6*bs+.4*ss
  row={'seed':seed,'elapsed_seconds':time.perf_counter()-started,'models':{name:metrics(s,d.y_test) for name,s in models.items()}};rows.append(row);print(seed,json.dumps(row),flush=True)
 names=['Joint-WGAN-GP','Joint-VAE','Joint-UMNN','Joint-DDPM','Joint-CLDM','STDE-CDM'];keys=['MAE','RMSE','CRPS','PS','ES','VS'];summary={}
 for name in names:
  summary[name]={k:{'mean':float(np.mean([r['models'][name][k] for r in rows])),'std':float(np.std([r['models'][name][k] for r in rows],ddof=1 if len(rows)>1 else 0))} for k in keys}
 out={'protocol':f'locked TEST, {len(a.seeds)} seeds, {n} joint scenarios/day; no retraining or selection','seeds':a.seeds,'rows':rows,'summary':summary}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
