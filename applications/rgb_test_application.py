"""OOP controller for testing the Astra Pro RGB stream separately."""

import cv2

from camera import RGBCamera, RGBCameraError
from reporting import EvidenceManager


class RGBTestApplication:
    """Coordinate RGB acquisition, preview and keyboard input."""

    def __init__(self, camera_index: int = 0) -> None:
        self.camera = RGBCamera(camera_index)
        self.evidence = EvidenceManager()
        self.window_name = "Astra Pro - RGB test"

    def run(self) -> int:
        try:
            with self.camera:
                width, height, fps = self.camera.get_stream_information()
                print(
                    f"Opened camera index {self.camera.camera_index}: "
                    f"{width}x{height} at {fps:.1f} FPS using "
                    f"{self.camera.active_backend_name}"
                )
                print("RGB stream started. Press Q or Escape to stop.")
                frame_number = 0
                evidence_saved = False
                while True:
                    rgb_frame = self.camera.read()
                    frame_number += 1

                    status_text = (
                        f"camera index: {self.camera.camera_index}  "
                        f"{rgb_frame.shape[1]}x{rgb_frame.shape[0]}"
                    )
                    cv2.putText(
                        rgb_frame, status_text, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA,
                    )

                    if frame_number >= 30 and not evidence_saved:
                        evidence_path = self.evidence.save_cv_image(
                            "rgb", "astra_rgb_preview", rgb_frame
                        )
                        print(f"RGB evidence saved: {evidence_path}")
                        evidence_saved = True

                    cv2.imshow(self.window_name, rgb_frame)

                    pressed_key = cv2.waitKey(1) & 0xFF
                    if pressed_key == ord("q") or pressed_key == 27:
                        break
        except RGBCameraError as error:
            print(f"RGB camera test failed: {error}")
            return 1
        finally:
            cv2.destroyAllWindows()

        return 0
