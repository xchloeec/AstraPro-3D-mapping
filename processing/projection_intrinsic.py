"""Resolve replay projection without mixing poses fitted with different intrinsics."""
from pathlib import Path
import numpy as np
import open3d as o3d


def load_projection(dataset, override=None, trajectory=None):
    dataset=Path(dataset)
    original_path=dataset/'intrinsic.json'
    original=o3d.io.read_pinhole_camera_intrinsic(str(original_path))
    snapshot=Path(trajectory).parent/'intrinsic_used.json' if trajectory else None
    has_snapshot=snapshot is not None and snapshot.is_file()
    path=Path(override).resolve() if override else snapshot if has_snapshot else original_path
    if not path.is_file():raise FileNotFoundError(path)
    intrinsic=o3d.io.read_pinhole_camera_intrinsic(str(path))
    k=intrinsic.intrinsic_matrix
    if ((intrinsic.width,intrinsic.height)!=(original.width,original.height) or
        not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or
        not np.allclose(k[2],[0,0,1])):
        raise ValueError('Invalid or resolution-mismatched replay projection')
    if trajectory:
        fitted=o3d.io.read_pinhole_camera_intrinsic(str(snapshot)) if has_snapshot else original
        if not np.allclose(k,fitted.intrinsic_matrix,rtol=0,atol=1e-6):
            raise ValueError('Selected poses use different intrinsics; recompute trajectory before fusion')
    return intrinsic,path.resolve()
