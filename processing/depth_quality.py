"""Conservative spatial depth screening; does not fill holes or classify materials."""
import cv2
import numpy as np


def screen_depth(depth_m):
    """Exclude unsupported pixels and sharp depth boundaries before fusion.

    Input/output are metres. Thresholds are heuristics, not sensor confidence.
    Neighbourhoods never wrap between image edges. Raw observations are kept by
    the recorder; this operates on a separate processing copy.
    """
    depth = np.asarray(depth_m,dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError('Expected a two-dimensional depth image')
    valid = np.isfinite(depth) & (depth >= .4) & (depth <= 5.)
    safe = np.where(valid,depth,0).astype(np.float32)
    kernel = np.ones((3,3),np.uint8)
    low = cv2.erode(np.where(valid,safe,6.).astype(np.float32),kernel,
                    borderType=cv2.BORDER_REPLICATE)
    high = cv2.dilate(safe,kernel,borderType=cv2.BORDER_REPLICATE)
    support = cv2.boxFilter(valid.astype(np.float32),-1,(3,3),normalize=False,
                            borderType=cv2.BORDER_REPLICATE)
    # Preserve smooth sloped surfaces; remove both sides of a large depth step.
    discontinuity = (high-low) > (.08+.02*safe)
    keep = valid & (support >= 5) & ~discontinuity
    result = np.where(keep,safe,0).astype(np.float32)
    return result, {'input_valid_pixels':int(valid.sum()),
                    'retained_pixels':int(keep.sum()),
                    'excluded_pixels':int((valid & ~keep).sum())}
