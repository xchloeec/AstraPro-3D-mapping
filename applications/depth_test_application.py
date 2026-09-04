"""OOP controller for the Stage 1 depth-camera test."""

import cv2

from camera import DepthCamera, OpenNIError
from processing import DepthFrameProcessor
from reporting import EvidenceManager


class DepthTestApplication:
    """Coordinate depth acquisition, processing, display and user input.

    This class uses composition: it has a DepthCamera and a
    DepthFrameProcessor. It does not inherit from them because an application
    is not a type of camera or a type of frame processor.
    """

    def __init__(self, display_limit_mm: int = 5000) -> None:
        self.camera = DepthCamera()
        self.processor = DepthFrameProcessor(display_limit_mm)
        self.evidence = EvidenceManager()
        self.window_name = "Astra Pro - depth test"

    def run(self) -> int:
        """Run the preview until Q/Escape is pressed or an error occurs."""
        try:
            with self.camera:
                print(f"OpenNI runtime: {self.camera.runtime_path}")
                print("Depth stream started. Press Q or Escape to stop.")

                frame_number = 0
                evidence_saved = False
                while True:
                    depth_mm = self.camera.read()
                    preview = self.processor.create_preview(depth_mm)
                    frame_number += 1

                    if frame_number >= 30 and not evidence_saved:
                        evidence_path = self.evidence.save_cv_image(
                            "depth", "depth_preview", preview
                        )
                        print(f"Depth evidence saved: {evidence_path}")
                        evidence_saved = True

                    cv2.imshow(self.window_name, preview)

                    if self._stop_requested():
                        break

        except OpenNIError as error:
            print(f"Depth camera test failed: {error}")
            return 1
        finally:
            cv2.destroyAllWindows()
            print("Depth camera test stopped safely.")

        return 0

    @staticmethod
    def _stop_requested() -> bool:
        """Return True when the user presses Q or Escape."""
        pressed_key = cv2.waitKey(1) & 0xFF
        return pressed_key == ord("q") or pressed_key == 27
