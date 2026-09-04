"""Create categorized, timestamped evidence files for project reports."""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class EvidenceManager:
    """Own the folder and naming rules for repeatable test evidence."""

    STAGE_FOLDERS = {
        "depth": "stage_01_depth",
        "rgb": "stage_02_rgb",
        "point_cloud": "stage_03_point_cloud",
        "rgbd_alignment": "stage_04_rgbd_alignment",
        "coloured_point_cloud": "stage_05_coloured_point_cloud",
        "point_cloud_filter": "stage_06_point_cloud_filter",
        "multiframe_capture": "stage_07_multiframe_capture",
        "multiframe_point_clouds": "stage_07b_multiframe_point_clouds",
        "open3d_dataset": "stage_08_open3d_dataset",
    }

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.evidence_root = self.project_root / "report_evidence"

    def new_path(self, stage: str, description: str, extension: str) -> Path:
        """Return a timestamped path inside the requested stage folder."""
        if stage not in self.STAGE_FOLDERS:
            raise ValueError(f"Unknown evidence stage: {stage}")

        stage_folder = self.evidence_root / self.STAGE_FOLDERS[stage]
        stage_folder.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return stage_folder / f"{timestamp}_{description}.{extension.lstrip('.')}"

    def save_cv_image(
        self, stage: str, description: str, image: np.ndarray
    ) -> Path:
        """Save an OpenCV image as PNG and return its report path."""
        output_path = self.new_path(stage, description, "png")
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Could not save report image: {output_path}")
        return output_path

    def save_text(self, stage: str, description: str, content: str) -> Path:
        """Save a UTF-8 diagnostic/report text file."""
        output_path = self.new_path(stage, description, "txt")
        output_path.write_text(content, encoding="utf-8")
        return output_path
