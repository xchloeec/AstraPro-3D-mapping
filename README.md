# Astra Pro 3D mapping

The project is intentionally split into independent stages. Stage 1 only tests
the OpenNI depth stream; RGB, point-cloud generation, registration, and mapping
will be added after this test works.

## Stage 1: depth test

1. Close NiViewer, OrbbecViewer, Windows Camera, and other camera applications.
2. Open this directory as a project in PyCharm.
3. Create or select a **64-bit Python 3.11** interpreter.
4. In PyCharm's terminal, run:

   ```powershell
   python -m pip install -r requirements.txt
   python 01_test_depth_camera.py
   ```

5. A colourized depth window should appear. Press **Q** or **Escape** to quit.

`01_test_depth_camera.py` is the learning version: all OpenNI steps are shown
through the OOP objects used by the project.

## OOP structure

```text
01_test_depth_camera.py
    -> DepthTestApplication
        -> DepthCamera
        -> DepthFrameProcessor

astra_3d_test.py
    -> RGBTestApplication
        -> RGBCamera
```

Each hardware object owns its connection and stream. Each application object
coordinates the relevant hardware and processing objects. Future point-cloud
and mapping stages will follow the same structure.

## Stage 2: RGB test

Close Windows Camera and other camera applications, then test camera indices
until the Astra Pro RGB view appears:

```powershell
python 02_test_rgb_camera.py --camera-index 0
python 02_test_rgb_camera.py --camera-index 1
```

## Stage 3: one depth-only XYZ point cloud

Close NiViewer and other depth-camera applications, then run:

```powershell
python 03_test_point_cloud.py
```

The program captures one stabilized depth frame, reads the active stream FOV,
calculates the pinhole intrinsics, back-projects valid pixels into XYZ metres,
saves `output/depth_point_cloud.ply`, and opens it in an Open3D viewer. The
display colours represent depth only; RGB alignment has not been added yet.

The program currently auto-detects the installed OpenNI runtime at:

```text
C:\1 Swinburne\Sem 6\openni\OpenNI\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs
```

If OpenNI is moved later, set `OPENNI2_REDIST` to the folder containing
`OpenNI2.dll`.
# Stage 11: RTAB-Map live room scanning

Run `11_launch_rtabmap_room_scanner.py` to create a new timestamped mapping
session and open RTAB-Map. Each session is stored under
`output/rtabmap_sessions/` as an editable `room_mapping.db` SLAM database.

On the first run, open **Edit > Preferences > Source**, choose an **RGB-D**
source and the **OpenNI2** driver, then verify the RGB and depth previews before
pressing **Start**. Move slowly and keep previously observed objects in view so
visual odometry has enough overlap. Revisit an earlier area to give RTAB-Map a
chance to detect a loop closure and correct accumulated drift.

The database is kept instead of immediately reducing the scan to PLY because it
contains the original frames, camera trajectory and graph constraints needed to
inspect or improve the reconstruction. Export the final point cloud/mesh after
the scan has been checked.
