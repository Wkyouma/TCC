import glob
import os
import pandas as pd

POSE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "pose")

for p in glob.glob(os.path.join(POSE_DIR, "adl-*-cam0-rgb.csv")):
    d = pd.read_csv(p)
    i = d.index[d.detectado == 1]
    d.loc[i[0]:].to_csv(p, index=False)
