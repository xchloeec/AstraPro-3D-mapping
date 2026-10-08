"""Feature-seeded RGB-D registration with independent depth consistency checks."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
import open3d as o3d
from .photometric import normalise_gray
from .depth_quality import screen_depth


def rotation_degrees(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))


def rigid_fit(source, target):
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-source_mean).T @ (target-target_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_mean - rotation @ source_mean
    return transform


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def robust_rigid(source, target, threshold=.06, iterations=250):
    """Reject wrong feature matches rather than using all matches in a fit."""
    if len(source) < 3:
        return None, np.zeros(len(source), dtype=bool)
    rng = np.random.default_rng(42)
    best = np.zeros(len(source), dtype=bool)
    for _ in range(iterations):
        chosen = rng.choice(len(source), 3, replace=False)
        a, b = source[chosen], target[chosen]
        if np.linalg.norm(np.cross(a[1]-a[0], a[2]-a[0])) < 1e-5:
            continue
        transform = rigid_fit(a, b)
        mask = np.linalg.norm(transform_points(source, transform)-target, axis=1) < threshold
        if mask.sum() > best.sum():
            best = mask
    if best.sum() < 3:
        return None, best
    transform = rigid_fit(source[best], target[best])
    best = np.linalg.norm(transform_points(source, transform)-target, axis=1) < threshold
    if best.sum() >= 3:
        transform = rigid_fit(source[best], target[best])
    return transform, best


@dataclass
class RegistrationFrame:
    rgbd: object
    cloud: object
    depth: np.ndarray
    keypoints: object
    descriptors: object
    intrinsic: object


def make_frame(rgbd, intrinsic):
    """Keep full image detail for feature matching; downsample geometry for ICP."""
    rgb = np.asarray(rgbd.color)
    depth = np.asarray(rgbd.depth)
    width = min(640, rgb.shape[1])
    height = round(rgb.shape[0] * width / rgb.shape[1])
    sx, sy = width / rgb.shape[1], height / rgb.shape[0]
    k = intrinsic.intrinsic_matrix
    scaled = o3d.camera.PinholeCameraIntrinsic(width, height, k[0,0]*sx, k[1,1]*sy,
                                             (k[0,2]+.5)*sx-.5, (k[1,2]+.5)*sy-.5)
    small_rgb = cv2.resize(rgb, (width,height), interpolation=cv2.INTER_AREA)
    small_depth = cv2.resize(depth, (width,height), interpolation=cv2.INTER_NEAREST)
    small_depth, _ = screen_depth(small_depth)
    small_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(small_rgb)),
        o3d.geometry.Image(np.ascontiguousarray(small_depth)),
        depth_scale=1, depth_trunc=5, convert_rgb_to_intensity=False,
    )
    gray = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2GRAY)
    gray = normalise_gray(gray)
    # Do not spend the feature budget on distant walls without usable depth.
    mask = cv2.erode((small_depth > 0).astype(np.uint8)*255,np.ones((3,3),np.uint8))
    kp, des = cv2.ORB_create(5000, fastThreshold=10).detectAndCompute(gray, mask)
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(small_rgbd, scaled).voxel_down_sample(.04)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.12, max_nn=30))
    return RegistrationFrame(small_rgbd, cloud, small_depth, kp, des, scaled)


def matches_between(source, target):
    if source.descriptors is None or target.descriptors is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    def ratio(a,b):
        return {pair[0].queryIdx: pair[0] for pair in matcher.knnMatch(a,b,k=2)
                if len(pair)==2 and pair[0].distance < .72*pair[1].distance}
    forward = ratio(source.descriptors,target.descriptors)
    reverse = ratio(target.descriptors,source.descriptors)
    return [m for m in forward.values() if m.trainIdx in reverse and reverse[m.trainIdx].trainIdx == m.queryIdx]


def backproject(frame, pixel):
    u,v = pixel
    x,y = round(u),round(v)
    if not (0 <= y < frame.depth.shape[0] and 0 <= x < frame.depth.shape[1]):
        return None
    z = float(frame.depth[y,x])
    if not .4 <= z <= 5:
        return None
    k = frame.intrinsic.intrinsic_matrix
    return [(u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1], z]


def register_pair(source, target, loop=False):
    """Return measured source->target only when visual AND depth checks pass."""
    details = {"accepted": False, "reason": "insufficient_features"}
    matches = matches_between(source,target)
    a,b,pixels = [],[],[]
    for match in matches:
        p = source.keypoints[match.queryIdx].pt
        q = target.keypoints[match.trainIdx].pt
        x,y = backproject(source,p),backproject(target,q)
        if x is not None and y is not None:
            a.append(x); b.append(y); pixels.append(p)
    details["depth_matches"] = len(a)
    minimum = 25 if loop else 15
    if len(a) < minimum:
        return False, np.eye(4), np.eye(6), details
    a,b = np.asarray(a),np.asarray(b)
    transform, inliers = robust_rigid(a,b)
    details["inliers"] = int(inliers.sum())
    if transform is None or inliers.sum() < minimum or inliers.mean() < (.45 if loop else .30):
        details["reason"] = "inconsistent_feature_depth"
        return False, np.eye(4), np.eye(6), details
    support = np.asarray(pixels,dtype=np.float32)[inliers]
    area = cv2.contourArea(cv2.convexHull(support)) / (source.depth.shape[0]*source.depth.shape[1])
    details["feature_support_fraction"] = float(area)
    if area < (.06 if loop else .025):
        details["reason"] = "features_too_localised"
        return False, transform, np.eye(6), details
    reg = o3d.pipelines.registration
    refined = reg.registration_icp(
        source.cloud, target.cloud, .08, transform,
        reg.TransformationEstimationPointToPlane(),
        reg.ICPConvergenceCriteria(max_iteration=20),
    )
    delta = np.linalg.inv(transform) @ refined.transformation
    selected = refined.transformation
    refined_residual = np.linalg.norm(transform_points(a[inliers],selected)-b[inliers],axis=1)
    # ICP on broad planes can slide away from the feature-supported solution.
    # Keep the independently measured seed if refinement breaks that support;
    # it must still pass the SAME bidirectional geometric checks below.
    if (np.linalg.norm(delta[:3,3]) >= .08 or rotation_degrees(delta[:3,:3]) >= 5
            or np.median(refined_residual) > .045):
        selected = transform
        delta = np.eye(4)
        details["method"] = "feature_depth_seed"
    else:
        details["method"] = "feature_depth_icp"
    reverse = reg.evaluate_registration(target.cloud, source.cloud, .06, np.linalg.inv(selected))
    forward = reg.evaluate_registration(source.cloud, target.cloud, .06, selected)
    residual = np.linalg.norm(transform_points(a[inliers],selected)-b[inliers],axis=1)
    details.update(fitness=float(min(forward.fitness,reverse.fitness)),
                   rmse_m=float(max(forward.inlier_rmse,reverse.inlier_rmse)),
                   feature_median_error_m=float(np.median(residual)))
    accepted = (
        np.isfinite(selected).all()
        and np.linalg.norm(delta[:3,3]) < .08
        and rotation_degrees(delta[:3,:3]) < 5
        and details["fitness"] >= (.5 if loop else .35)
        and details["rmse_m"] <= .035
        and details["feature_median_error_m"] <= .045
    )
    if not loop:
        accepted = accepted and np.linalg.norm(selected[:3,3]) <= .30 and rotation_degrees(selected[:3,:3]) <= 25
    details["accepted"] = bool(accepted)
    details["reason"] = "validated" if accepted else "geometry_validation_failed"
    information = reg.get_information_matrix_from_point_clouds(source.cloud,target.cloud,.06,selected)
    return bool(accepted), selected, information, details
