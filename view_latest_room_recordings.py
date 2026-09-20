"""Play the two most recent Stage 9 RGB recordings side by side.

This viewer checks the captured scanning route.  It does not reconstruct or
modify the RGB-D dataset, and it does not require the camera to be connected.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class LatestRoomRecordingViewer:
    """Load matching frame numbers from the newest two recording folders."""

    def __init__(self, playback_fps: float = 10.0) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.recordings_root = self.project_root / "output" / "room_recordings"
        self.playback_fps = playback_fps
        self.paused = False

    def run(self) -> int:
        # Failed camera starts can leave newer but empty room_* folders.  They
        # are not recordings, so filter them out before choosing the latest two.
        recordings = sorted(
            (
                folder
                for folder in self.recordings_root.glob("room_*")
                if any((folder / "color").glob("*.jpg"))
            ),
            key=lambda path: path.name,
            reverse=True,
        )[:2]
        if len(recordings) < 2:
            print("At least two room recording folders are required.")
            return 1

        frame_lists = [sorted((folder / "color").glob("*.jpg")) for folder in recordings]
        frame_count = min(len(frames) for frames in frame_lists)
        delay_ms = max(1, round(1000.0 / self.playback_fps))
        index = 0

        print(f"Left:  {recordings[0]} ({len(frame_lists[0])} RGB frames)")
        print(f"Right: {recordings[1]} ({len(frame_lists[1])} RGB frames)")
        print("Space: pause/resume | A/D: previous/next frame | Q/Esc: close")

        while index < frame_count:
            images = [cv2.imread(str(frames[index])) for frames in frame_lists]
            if any(image is None for image in images):
                print(f"Could not read RGB frame {index}.")
                return 1

            panels = [
                self._make_panel(image, folder.name, index, frame_count)
                for image, folder in zip(images, recordings)
            ]
            comparison = np.hstack(panels)
            cv2.imshow("Latest two Stage 9 room recordings", comparison)

            key = cv2.waitKey(0 if self.paused else delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                self.paused = not self.paused
                continue
            if key in (ord("a"), 81):
                index = max(0, index - 1)
                self.paused = True
                continue
            if key in (ord("d"), 83):
                index = min(frame_count - 1, index + 1)
                self.paused = True
                continue
            if not self.paused:
                index += 1

        cv2.destroyAllWindows()
        return 0

    @staticmethod
    def _make_panel(
        image: np.ndarray,
        folder_name: str,
        index: int,
        frame_count: int,
    ) -> np.ndarray:
        panel_width, panel_height = 640, 480
        panel = cv2.resize(image, (panel_width, panel_height))
        cv2.rectangle(panel, (0, 0), (panel_width, 62), (0, 0, 0), -1)
        cv2.putText(
            panel,
            folder_name,
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"RGB frame {index + 1}/{frame_count}",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 230, 80),
            2,
            cv2.LINE_AA,
        )
        return panel


if __name__ == "__main__":
    raise SystemExit(LatestRoomRecordingViewer().run())
