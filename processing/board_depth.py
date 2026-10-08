"""Read supported depth at detected checkerboard corners; never fill zeros."""
import numpy as np


def corner_depths(depth_mm, corners):
    values=[]
    for x,y in np.asarray(corners).reshape(-1,2):
        x=int(round(x));y=int(round(y))
        if not (0<=x<depth_mm.shape[1] and 0<=y<depth_mm.shape[0]):
            values.append(np.nan);continue
        patch=depth_mm[max(0,y-2):y+3,max(0,x-2):x+3].astype(float)/1000
        valid=patch[(patch>=.4)&(patch<=5)]
        values.append(float(np.median(valid)) if len(valid)>=8 else np.nan)
    return np.asarray(values)
