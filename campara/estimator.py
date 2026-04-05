"""
Interactive camera parameter estimator using OpenCV sliders.

Given a single image and an initial camera parameter file, lets you
interactively adjust rotation (theta_x, theta_y, theta_z), focal length,
and vertical translation (Tz) while visualising the ground-plane grid
projected onto the image. Press 'q' to save and quit.
"""

import copy

import cv2
import numpy as np

from .mapper import readCamParaFile


def _xy2uv(x, y, Ki, Ko):
    uv = Ki @ Ko @ np.array([x, y, 0, 1])
    uv /= uv[2]
    return int(uv[0]), int(uv[1])


class CameraParaEstimator:
    """Interactive GUI for estimating / refining camera parameters."""

    # Slider range constants
    _THETA_RANGE = 500       # ±25 degrees  (value-250)/10
    _THETA_Z_RANGE = 1000    # ±100 degrees (value-500)/5
    _FOCAL_RANGE = 500
    _TZ_RANGE = 500

    def __init__(self, image_path: str, cam_para_path: str):
        self.image_path = image_path
        self.cam_para_path = cam_para_path

        self.Ki, self.Ko, ok = readCamParaFile(cam_para_path)
        if not ok:
            raise FileNotFoundError(f"Cannot read camera parameter file: {cam_para_path}")

        self._theta_x = 0.0
        self._theta_y = 0.0
        self._theta_z = 0.0
        self._focal   = 0
        self._tz      = 0.0

        self._value_display = np.zeros((360, 320), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Slider callbacks
    # ------------------------------------------------------------------
    def _on_theta_x(self, v): self._theta_x = (v - 250) / 10.0;  self._refresh_hud()
    def _on_theta_y(self, v): self._theta_y = (v - 250) / 10.0;  self._refresh_hud()
    def _on_theta_z(self, v): self._theta_z = (v - 500) / 5.0;   self._refresh_hud()
    def _on_focal(self,   v): self._focal   = (v - 100) * 5;      self._refresh_hud()
    def _on_tz(self,      v): self._tz      = (v - 30)  * 0.04;   self._refresh_hud()

    def _refresh_hud(self):
        d = self._value_display
        d.fill(0)
        for i, (label, val) in enumerate([
            ("theta_x", f"{self._theta_x:.2f} deg"),
            ("theta_y", f"{self._theta_y:.2f} deg"),
            ("theta_z", f"{self._theta_z:.2f} deg"),
            ("focal",   f"{self._focal} px"),
            ("Tz",      f"{self._tz:.2f} m"),
        ]):
            cv2.putText(d, f"{label}: {val}", (10, 60 + i * 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        cv2.imshow("Values", d)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def _build_matrices(self):
        Ki = copy.copy(self.Ki)
        Ko = copy.copy(self.Ko)

        tx = self._theta_x / 180 * np.pi
        ty = self._theta_y / 180 * np.pi
        tz = self._theta_z / 180 * np.pi

        Rx = np.array([[1,0,0],[0,np.cos(tx),-np.sin(tx)],[0,np.sin(tx),np.cos(tx)]])
        Ry = np.array([[np.cos(ty),0,np.sin(ty)],[0,1,0],[-np.sin(ty),0,np.cos(ty)]])
        Rz = np.array([[np.cos(tz),-np.sin(tz),0],[np.sin(tz),np.cos(tz),0],[0,0,1]])

        Ko[:3, :3] = Ko[:3, :3] @ Rx @ Ry @ Rz
        Ko[2, 3]  += self._tz
        Ki[0, 0]  += self._focal
        Ki[1, 1]  += self._focal
        return Ki, Ko

    def run(self, grid_x=(0, 10, 0.5), grid_y=(-5, 5, 0.5), display_scale=0.5):
        """Start the interactive GUI loop.

        Args:
            grid_x: (start, stop, step) for the ground-plane X axis (meters).
            grid_y: (start, stop, step) for the ground-plane Y axis (meters).
            display_scale: Scale factor applied to the displayed image.
        """
        img_orig = cv2.imread(self.image_path)
        if img_orig is None:
            raise FileNotFoundError(f"Cannot read image: {self.image_path}")

        h, w = img_orig.shape[:2]

        cv2.namedWindow("CamParaSettings")
        cv2.namedWindow("Values")

        cv2.createTrackbar("theta_x", "CamParaSettings", 250, self._THETA_RANGE,  self._on_theta_x)
        cv2.createTrackbar("theta_y", "CamParaSettings", 250, self._THETA_RANGE,  self._on_theta_y)
        cv2.createTrackbar("theta_z", "CamParaSettings", 500, self._THETA_Z_RANGE, self._on_theta_z)
        cv2.createTrackbar("focal",   "CamParaSettings", 100, self._FOCAL_RANGE,  self._on_focal)
        cv2.createTrackbar("Tz",      "CamParaSettings", 30,  self._TZ_RANGE,     self._on_tz)

        self._refresh_hud()

        while True:
            Ki, Ko = self._build_matrices()
            img = img_orig.copy()

            for x in np.arange(*grid_x):
                for y in np.arange(*grid_y):
                    u, v = _xy2uv(x, y, Ki, Ko)
                    cv2.circle(img, (u, v), 3, (0, 255, 0), -1)

            disp = cv2.resize(img, (int(w * display_scale), int(h * display_scale)))
            cv2.imshow("img", disp)

            if cv2.waitKey(50) == ord("q"):
                break

        cv2.destroyAllWindows()
        self._save(Ko, Ki)
        print(f"Camera parameters saved to: {self.cam_para_path}")

    def _save(self, Ko, Ki):
        R = Ko[:3, :3]
        T = Ko[:3, 3]
        with open(self.cam_para_path, "w") as f:
            f.write("RotationMatrices\n")
            for row in R:
                f.write(" ".join(str(v) for v in row) + " \n")
            f.write("\nTranslationVectors\n")
            f.write(" ".join(str(int(v * 1000)) for v in T) + " \n")
            f.write("\nIntrinsicMatrix\n")
            K3 = Ki[:3, :3]
            for row in K3:
                f.write(" ".join(str(int(v)) for v in row) + " \n")
