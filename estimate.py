#!/usr/bin/env python3
"""
CamParaEstimator — CLI entry point.

Usage:
    python estimate.py --img examples/demo.jpg --cam_para examples/cam_para.txt
"""

import argparse
from campara import CameraParaEstimator


def main():
    parser = argparse.ArgumentParser(
        description="Interactively estimate camera parameters from a single image."
    )
    parser.add_argument("--img",       required=True, help="Path to input image")
    parser.add_argument("--cam_para",  required=True, help="Path to camera parameter file (will be updated on quit)")
    parser.add_argument("--scale",     type=float, default=0.5, help="Display scale factor (default: 0.5)")
    args = parser.parse_args()

    estimator = CameraParaEstimator(args.img, args.cam_para)
    estimator.run(display_scale=args.scale)


if __name__ == "__main__":
    main()
