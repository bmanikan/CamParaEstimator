"""
Interactive camera parameter estimator — matches the original UCMCTrack UI exactly.

Two windows:
  - CamParaSettings : five trackbars (theta_x, theta_y, theta_z, focal, Tz)
  - img             : live preview with green ground-plane grid projected on image
  - Values          : small readout panel showing current values in degrees/pixels

Press 'q' to save and quit.
"""

import copy

import cv2
import numpy as np

from .mapper import readCamParaFile


# ── helpers ────────────────────────────────────────────────────────────────

def _xy2uv(x: float, y: float, Ki: np.ndarray, Ko: np.ndarray):
    uv = Ki @ Ko @ np.array([x, y, 0, 1])
    uv /= uv[2]
    return int(uv[0]), int(uv[1])


def _rotation(tx, ty, tz):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(tx), -np.sin(tx)],
                   [0, np.sin(tx),  np.cos(tx)]])
    Ry = np.array([[ np.cos(ty), 0, np.sin(ty)],
                   [0,           1, 0          ],
                   [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0],
                   [np.sin(tz),  np.cos(tz), 0],
                   [0,           0,          1]])
    return Rx, Ry, Rz


# ── slider value decoders (match original exactly) ─────────────────────────

def _theta(v):   return (v - 250) / 10.0      # ±25 deg
def _theta_z(v): return (v - 500) / 5.0       # ±100 deg
def _focal(v):   return (v - 100) * 5         # focal delta in pixels
def _tz(v):      return (v - 30)  * 0.04      # Tz delta in meters


# ── main class ─────────────────────────────────────────────────────────────

class CameraParaEstimator:
    """Interactive GUI for estimating / refining camera parameters from a single image."""

    def __init__(self, image_path: str, cam_para_path: str):
        self.image_path    = image_path
        self.cam_para_path = cam_para_path

        self.Ki, self.Ko, ok = readCamParaFile(cam_para_path)
        if not ok:
            raise FileNotFoundError(f"Cannot read camera parameter file: {cam_para_path}")

        # Current slider-decoded values
        self._theta_x = 0.0
        self._theta_y = 0.0
        self._theta_z = 0.0
        self._focal   = 0
        self._tz      = 0.0

        # Values display panel (300 × 400 greyscale — same as original)
        self._value_display = np.zeros((400, 300), dtype=np.uint8)

    # ── slider callbacks ───────────────────────────────────────────────────

    def _on_theta_x(self, v):
        self._theta_x = _theta(v)
        self._refresh_values()

    def _on_theta_y(self, v):
        self._theta_y = _theta(v)
        self._refresh_values()

    def _on_theta_z(self, v):
        self._theta_z = _theta_z(v)
        self._refresh_values()

    def _on_focal(self, v):
        self._focal = _focal(v)
        self._refresh_values()

    def _on_tz(self, v):
        self._tz = _tz(v)
        self._refresh_values()

    def _refresh_values(self):
        d = self._value_display
        d.fill(0)
        entries = [
            ("theta_x", f"{self._theta_x:.2f}"),
            ("theta_y", f"{self._theta_y:.2f}"),
            ("theta_z", f"{self._theta_z:.2f}"),
            ("focal",   f"{self._focal}"),
            ("Tz",      f"{self._tz:.2f}"),
        ]
        for i, (label, val) in enumerate(entries):
            cv2.putText(d, f"{label}: {val}",
                        (10, 60 + i * 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imshow("Values", d)

    # ── matrix helpers ─────────────────────────────────────────────────────

    def _build_matrices(self):
        Ki = copy.copy(self.Ki)
        Ko = copy.copy(self.Ko)

        tx = self._theta_x / 180.0 * np.pi
        ty = self._theta_y / 180.0 * np.pi
        tz = self._theta_z / 180.0 * np.pi

        Rx, Ry, Rz = _rotation(tx, ty, tz)
        Ko[:3, :3] = Ko[:3, :3] @ Rx @ Ry @ Rz
        Ko[2, 3]  += self._tz
        Ki[0, 0]  += self._focal
        Ki[1, 1]  += self._focal
        return Ki, Ko

    # ── public API ─────────────────────────────────────────────────────────

    def run(self,
            grid_x=(0, 10, 0.5),
            grid_y=(-5, 5, 0.5),
            display_scale=0.5):
        """
        Start the interactive calibration loop.

        Args:
            grid_x        : (start, stop, step) sweep range on ground X axis (metres).
            grid_y        : (start, stop, step) sweep range on ground Y axis (metres).
            display_scale : Scale factor applied to the preview image (default 0.5).
        """
        img_orig = cv2.imread(self.image_path)
        if img_orig is None:
            raise FileNotFoundError(f"Cannot read image: {self.image_path}")

        h, w = img_orig.shape[:2]

        # ── create windows ─────────────────────────────────────────────────
        cv2.namedWindow("CamParaSettings")
        cv2.namedWindow("Values")

        # ── trackbars (initial values match original defaults) ─────────────
        cv2.createTrackbar("theta_x", "CamParaSettings", 250, 500,  self._on_theta_x)
        cv2.createTrackbar("theta_y", "CamParaSettings", 250, 500,  self._on_theta_y)
        cv2.createTrackbar("theta_z", "CamParaSettings", 500, 1000, self._on_theta_z)
        cv2.createTrackbar("focal",   "CamParaSettings", 100, 500,  self._on_focal)
        cv2.createTrackbar("Tz",      "CamParaSettings", 30,  500,  self._on_tz)

        self._refresh_values()

        # ── main loop ──────────────────────────────────────────────────────
        while True:
            Ki, Ko = self._build_matrices()
            img = img_orig.copy()

            # Draw ground-plane grid (green filled circles, same as original)
            for x in np.arange(*grid_x):
                for y in np.arange(*grid_y):
                    u, v = _xy2uv(x, y, Ki, Ko)
                    cv2.circle(img, (u, v), 3, (0, 255, 0), -1)

            disp = cv2.resize(img, (int(w * display_scale),
                                    int(h * display_scale)))
            cv2.imshow("img", disp)

            if cv2.waitKey(50) == ord("q"):
                break

        cv2.destroyAllWindows()
        self._save(Ko, Ki)
        print(f"Saved → {self.cam_para_path}")

    # ── file I/O ───────────────────────────────────────────────────────────

    def _save(self, Ko: np.ndarray, Ki: np.ndarray):
        R = Ko[:3, :3]
        T = Ko[:3,  3]
        K3 = Ki[:3, :3]

        with open(self.cam_para_path, "w") as f:
            f.write("RotationMatrices\n")
            for row in R:
                f.write(" ".join(str(v) for v in row) + " \n")

            f.write("\nTranslationVectors\n")
            f.write(" ".join(str(int(v * 1000)) for v in T) + " \n")

            f.write("\nIntrinsicMatrix\n")
            for row in K3:
                f.write(" ".join(str(int(v)) for v in row) + " \n")
