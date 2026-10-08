"""Software exposure normalisation and diagnostics, not sensor calibration."""
import cv2
import numpy as np


def light_metrics(bgr):
    gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    return {"median_luma":float(np.median(gray)),
            "dark_fraction":float(np.mean(gray < 30)),
            "clipped_fraction":float(np.mean(gray > 245)),
            "laplacian_variance":float(cv2.Laplacian(gray,cv2.CV_64F).var())}


def normalise_gray(gray):
    median = float(np.median(gray))
    # Bound gain: lifting a black frame cannot create usable image detail.
    gamma = float(np.clip(np.log(110/255)/np.log(max(5,min(250,median))/255),.65,1.3))
    lut = np.clip((np.arange(256)/255.)**gamma*255,0,255).astype(np.uint8)
    enhanced = cv2.LUT(gray,lut)
    return cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(enhanced)
