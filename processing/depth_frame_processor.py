"""Processing and visualization for one depth frame."""

from __future__ import annotations

import cv2
import numpy as np


class DepthFrameProcessor:
    """Convert raw millimetre measurements into a diagnostic preview."""

    def __init__(self, display_limit_mm: int = 5000) -> None:
        self.display_limit_mm = display_limit_mm

    def create_preview(self, depth_mm: np.ndarray) -> np.ndarray:
        """Return a colourized copy of a raw depth frame."""
        clipped_depth = np.clip(depth_mm, 0, self.display_limit_mm)
        depth_8bit = (
            clipped_depth * (255.0 / self.display_limit_mm)
        ).astype(np.uint8)

        colour_depth = cv2.applyColorMap(
            255 - depth_8bit,
            cv2.COLORMAP_TURBO,
        )

        status_text = self._build_status_text(depth_mm)
        if status_text is not None:
            cv2.putText(
                colour_depth, status_text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
                cv2.LINE_AA,
            )

        return colour_depth

    def _build_status_text(self, depth_mm: np.ndarray) -> str | None:
        """Build the resolution/range label from valid non-zero pixels.

        The leading underscore marks this as an internal helper method. Other
        objects normally call create_preview(), not this method directly.
        """
        valid_depth = depth_mm[depth_mm > 0]
        if valid_depth.size == 0:
            return None

        height, width = depth_mm.shape
        nearest_mm = int(valid_depth.min())
        farthest_mm = int(valid_depth.max())
        return f"{width}x{height}  range: {nearest_mm}-{farthest_mm} mm"
