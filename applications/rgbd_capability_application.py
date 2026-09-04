"""Stage 4 diagnostic before choosing an RGB-D alignment method."""

import math

from camera import DepthCamera, OpenNIError
from reporting import EvidenceManager


class RGBDCapabilityApplication:
    """Inspect and document Astra Pro RGB-D features exposed by OpenNI2."""

    def __init__(self) -> None:
        self.camera = DepthCamera()
        self.evidence = EvidenceManager()

    def run(self) -> int:
        try:
            with self.camera:
                capabilities = self.camera.inspect_rgbd_capabilities()
                horizontal_fov, vertical_fov = self.camera.get_field_of_view()

            report_lines = [
                "Astra Pro RGB-D capability check",
                "================================",
                f"OpenNI runtime: {self.camera.runtime_path}",
                f"Depth horizontal FOV: {math.degrees(horizontal_fov):.2f} deg",
                f"Depth vertical FOV: {math.degrees(vertical_fov):.2f} deg",
                "",
            ]
            for capability, supported in capabilities.items():
                report_lines.append(f"{capability}: {supported}")

            if capabilities["depth_to_colour_registration"]:
                route = (
                    "Use OpenNI Depth-to-Color registration for depth and "
                    "OpenCV/UVC for Astra Pro RGB."
                )
            else:
                route = (
                    "Use OpenCV RGB plus explicit intrinsic/extrinsic "
                    "calibration; do not align by resizing images."
                )

            report_lines.extend(["", f"Recommended Stage 4 route: {route}"])
            report = "\n".join(report_lines)
            print(report)

            evidence_path = self.evidence.save_text(
                "rgbd_alignment", "openni_rgbd_capabilities", report
            )
            print(f"Capability evidence saved: {evidence_path}")
            return 0

        except OpenNIError as error:
            print(f"RGB-D capability check failed: {error}")
            return 1
