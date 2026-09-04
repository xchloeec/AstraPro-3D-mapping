"""OOP controller for the Stage 4 registered RGB-D test."""

import cv2
import numpy as np

from camera import OpenNIError, RGBCameraError, RGBDCamera
from processing import DepthFrameProcessor
from reporting import EvidenceManager


class RGBDAlignmentApplication:
    """Acquire, visualize and document registered RGB and depth frames."""

    def __init__(self, display_limit_mm: int = 5000) -> None:
        self.camera = RGBDCamera()
        self.depth_processor = DepthFrameProcessor(display_limit_mm)
        self.evidence = EvidenceManager()
        self.window_name = "Astra Pro - registered RGB-D alignment"

    def run(self) -> int:
        try:
            with self.camera:
                print(f"OpenNI runtime: {self.camera.runtime_path}")
                print(f"Depth-to-colour registration: {self.camera.registration_enabled}")
                print(
                    "Depth mirroring: "
                    f"{self.camera.depth_camera.stream.get_mirroring_enabled()}"
                )
                print(f"Depth/colour synchronization: {self.camera.synchronization_enabled}")
                if not self.camera.synchronization_enabled:
                    print(
                        "NOTE: Astra Pro RGB uses its separate UVC interface; "
                        "the two frames are not hardware synchronized."
                    )
                print("Press Q or Escape to stop.")

                frame_number = 0
                evidence_saved = False
                while True:
                    depth_mm, colour_bgr = self.camera.read()
                    preview = self._build_preview(depth_mm, colour_bgr)
                    frame_number += 1

                    if frame_number >= 30 and not evidence_saved:
                        path = self.evidence.save_cv_image(
                            "rgbd_alignment", "registered_rgbd_preview", preview
                        )
                        print(f"RGB-D alignment evidence saved: {path}")
                        evidence_saved = True

                    cv2.imshow(self.window_name, preview)
                    if self._stop_requested():
                        break

            return 0
        except (OpenNIError, RGBCameraError, ValueError) as error:
            print(f"RGB-D alignment test failed: {error}")
            return 1
        finally:
            cv2.destroyAllWindows()
            print("RGB-D alignment test stopped safely.")

    def _build_preview(
        self, depth_mm: np.ndarray, colour_bgr: np.ndarray
    ) -> np.ndarray:
        """Create colour, depth and overlay panels at one display size."""
        depth_preview = self.depth_processor.create_preview(depth_mm)

        # Registration changes the depth coordinate system to the colour view,
        # but stream resolutions may still differ. Resize only for displaying
        # panels; this resized array is never used as calibrated 3-D data.
        height, width = depth_mm.shape
        colour_display = cv2.resize(colour_bgr, (width, height))
        if depth_preview.shape[:2] != (height, width):
            depth_preview = cv2.resize(depth_preview, (width, height))

        valid_depth_mask = depth_mm > 0
        overlay = colour_display.copy()
        overlay[valid_depth_mask] = cv2.addWeighted(
            colour_display, 0.55, depth_preview, 0.45, 0
        )[valid_depth_mask]

        self._label(colour_display, "OpenNI RGB")
        self._label(depth_preview, "Registered depth")
        self._label(overlay, "RGB + depth overlay")
        return np.hstack((colour_display, depth_preview, overlay))

    @staticmethod
    def _label(image: np.ndarray, text: str) -> None:
        cv2.putText(
            image,
            text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _stop_requested() -> bool:
        pressed_key = cv2.waitKey(1) & 0xFF
        return pressed_key == ord("q") or pressed_key == 27
