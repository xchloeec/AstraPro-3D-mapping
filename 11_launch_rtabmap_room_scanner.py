"""Stage 11 - launch RTAB-Map for live RGB-D room mapping.

Why this stage exists
---------------------
Stages 9 and 10 generated point clouds with our own Python/Open3D pipeline.
They could combine frames, but small pose errors accumulated while the camera
was moving.  RTAB-Map is a complete SLAM system: it estimates camera motion,
recognises previously visited places (loop closure), and optimises the whole
camera trajectory before producing the final room model.

This file deliberately does not capture the camera itself.  Its job is to:
1. check that RTAB-Map is installed;
2. create a separate folder for this scanning session;
3. launch RTAB-Map with a new database path;
4. keep the database and a small information file together for later export.

The first time RTAB-Map opens, select OpenNI2 in its Source preferences.  That
selection is stored by RTAB-Map, so later launches normally remember it.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RtabmapSession:
    """Paths belonging to one room-scanning experiment."""

    folder: Path
    database: Path
    notes: Path


class RtabmapRoomScanner:
    """Prepare one session and hand camera mapping over to RTAB-Map."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.rtabmap_install_bin = Path(r"C:\Program Files\RTABMap\bin")
        # Run a copy outside Program Files. Windows always searches beside an
        # executable before PATH; the installed executable therefore forced
        # RTAB-Map's bundled OpenNI2.dll to load. The small launcher copy has no
        # DLL beside it, allowing our tested Orbbec runtime to be selected.
        self.rtabmap_exe = (
            project_root
            / "external"
            / "rtabmap_orbbec_runtime"
            / "RTABMap.exe"
        )
        # This is the same Orbbec OpenNI2 runtime already proven by Stages 1-5.
        # RTAB-Map's bundled OpenNI2 folder does not contain orbbec.dll, so it
        # cannot discover the Astra unless we explicitly select this runtime.
        self.openni2_redist = Path(
            r"C:\1 Swinburne\Sem 6\openni\OpenNI"
            r"\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows"
            r"\Win64-Release\sdk\libs"
        )
        self.sessions_root = project_root / "output" / "rtabmap_sessions"

    def create_session(self) -> RtabmapSession:
        """Make a timestamped folder so an older scan is never overwritten."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.sessions_root / f"room_{timestamp}"
        folder.mkdir(parents=True, exist_ok=False)
        return RtabmapSession(
            folder=folder,
            database=folder / "room_mapping.db",
            notes=folder / "session_info.txt",
        )

    def check_installation(self) -> None:
        """Fail with a useful message instead of a Windows file-not-found error."""
        if not self.rtabmap_exe.is_file():
            raise FileNotFoundError(
                "RTAB-Map was not found at:\n"
                f"  {self.rtabmap_exe}\n"
                "Install RTAB-Map or update 'rtabmap_exe' in this program."
            )
        if not (self.openni2_redist / "OpenNI2.dll").is_file():
            raise FileNotFoundError(
                "The tested Orbbec OpenNI2 runtime was not found at:\n"
                f"  {self.openni2_redist}\n"
                "Stage 11 needs this folder because it contains the Astra driver."
            )
        if not (
            self.openni2_redist / "OpenNI2" / "Drivers" / "orbbec.dll"
        ).is_file():
            raise FileNotFoundError(
                "The Astra OpenNI2 driver 'orbbec.dll' is missing from:\n"
                f"  {self.openni2_redist / 'OpenNI2' / 'Drivers'}"
            )

    @staticmethod
    def write_session_notes(session: RtabmapSession) -> None:
        """Record what the database contains for future reports."""
        session.notes.write_text(
            "Stage 11 - RTAB-Map live RGB-D room scan\n"
            "=========================================\n\n"
            f"Database: {session.database}\n\n"
            "The .db file is the editable SLAM project. It can contain RGB and "
            "depth frames, estimated camera poses, visual features, loop-closure "
            "constraints, and map settings. Export PLY only after inspecting and "
            "optimising the map.\n",
            encoding="utf-8",
        )

    def run(self) -> int:
        self.check_installation()
        session = self.create_session()
        self.write_session_notes(session)

        print("Stage 11: RTAB-Map live room scanner")
        print("====================================")
        print(f"Session folder: {session.folder}")
        print(f"SLAM database:  {session.database}")
        print(f"OpenNI runtime: {self.openni2_redist}")
        print()
        print("In RTAB-Map:")
        print("  1. If asked, create/open the database shown above.")
        print("  2. Edit > Preferences > Source: select RGB-D and OpenNI2.")
        print("  3. Confirm the live colour and depth previews are aligned.")
        print("  4. Press Start, then move slowly with overlapping views.")
        print("  5. Press Stop before closing RTAB-Map.")
        print()

        # RTAB-Map's GUI creates a new database through File > New database.
        # We therefore open the application without pretending that the not-yet
        # created SQLite file already exists. A list is used instead of a shell
        # command string so Windows paths are handled safely.
        # OPENNI2_REDIST tells OpenNI2 where both OpenNI2.dll and the matching
        # OpenNI2/Drivers/orbbec.dll are located. Prepending the directory to
        # PATH also lets Windows resolve the matching runtime dependencies.
        child_environment = os.environ.copy()
        child_environment["OPENNI2_REDIST"] = str(self.openni2_redist)
        child_environment["PATH"] = (
            str(self.openni2_redist)
            + os.pathsep
            + str(self.rtabmap_install_bin)
            + os.pathsep
            + child_environment.get("PATH", "")
        )
        # The copied executable still uses Qt plugins from the official
        # installation; only the OpenNI2 runtime selection is changed.
        child_environment["QT_PLUGIN_PATH"] = str(
            self.rtabmap_install_bin / "plugins"
        )

        completed = subprocess.run(
            [str(self.rtabmap_exe)],
            check=False,
            env=child_environment,
        )

        if session.database.exists():
            size_mib = session.database.stat().st_size / (1024 * 1024)
            print(f"RTAB-Map closed. Database saved ({size_mib:.1f} MiB):")
            print(f"  {session.database}")
        else:
            print("RTAB-Map closed, but the expected database was not created.")
            print("Use File > Save database as... and select:")
            print(f"  {session.database}")

        return completed.returncode


if __name__ == "__main__":
    project = Path(__file__).resolve().parent
    raise SystemExit(RtabmapRoomScanner(project).run())
