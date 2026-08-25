from pathlib import Path
import sys,torch,numpy as np
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src')]
from stde_cdm import JointCLDM,load_joint
def test_joint_data():
 d=load_joint(R/'data/wind_data_all_zone.csv');assert d.x_train.shape==(631,24,5,4);assert d.y_test.shape==(50,24,5);assert d.test_dates[1]==np.datetime64('2012-07-06T01:00:00')
def test_joint_model():
 m=JointCLDM(channels=8,layers=2,steps=4);x=torch.randn(2,24,5,4);y=torch.rand(2,24,5);assert torch.isfinite(m.loss(x,y));assert m.sample(x,3).shape==(2,3,24,5)
