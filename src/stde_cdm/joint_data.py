from dataclasses import dataclass
from pathlib import Path
import numpy as np
from cldm import load_gefcom_zone_unified

@dataclass
class JointSplits:
    x_train:np.ndarray;y_train:np.ndarray;x_validation:np.ndarray;y_validation:np.ndarray;x_test:np.ndarray;y_test:np.ndarray
    train_dates:np.ndarray;validation_dates:np.ndarray;test_dates:np.ndarray

def load_joint(path: str|Path,zones=(1,2,3,4,5),seed=0)->JointSplits:
    parts=[load_gefcom_zone_unified(path,z,seed=seed) for z in zones]
    for name in ['train_dates','validation_dates','test_dates']:
        assert all(np.array_equal(getattr(parts[0],name),getattr(p,name)) for p in parts[1:])
    def sx(name):return np.stack([getattr(p,name) for p in parts],axis=2)
    def sy(name):return np.stack([getattr(p,name) for p in parts],axis=2)
    return JointSplits(sx('x_train'),sy('y_train'),sx('x_validation'),sy('y_validation'),sx('x_test'),sy('y_test'),parts[0].train_dates,parts[0].validation_dates,parts[0].test_dates)
