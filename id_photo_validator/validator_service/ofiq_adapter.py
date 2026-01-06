"""
Thin ctypes wrapper around the C shim (`ofiq_c_api.dll`) that exposes two
functions:
  - ofiq_init(config_dir, config_file)
  - ofiq_score_rgb(data, width, height, stride, out_score)

The C shim was added under `external/OFIQ-Project/ofiq_c_api.cpp` and built to:
`external/OFIQ-Project/install_x86_64/Release/bin/ofiq_c_api.dll`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import ctypes
import logging
import os

import numpy as np


@dataclass
class OFIQScorer:
    """
    Load the OFIQ C shim and expose a simple `score` API that returns the
    UnifiedQualityScore for a single RGB image (H x W x 3, uint8).
    """

    dll_path: Optional[Path] = None
    config_dir: Optional[Path] = None
    config_file: str = "ofiq_config.jaxn"

    def __post_init__(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        default_bin = (
            project_root
            / "external"
            / "OFIQ-Project"
            / "install_x86_64"
            / "Release"
            / "bin"
        )
        default_data = project_root / "external" / "OFIQ-Project" / "data"

        self.dll_path = self.dll_path or (default_bin / "ofiq_c_api.dll")
        self.config_dir = self.config_dir or default_data

        if not self.dll_path.exists():
            raise ImportError(f"ofiq_c_api.dll not found at {self.dll_path}")
        if not (self.config_dir / self.config_file).exists():
            raise ImportError(f"OFIQ config not found at {self.config_dir / self.config_file}")

        # Add DLL directory first so dependencies (ofiq_lib.dll, onnxruntime.dll) resolve.
        os.environ["PATH"] = f"{self.dll_path.parent};{os.environ.get('PATH', '')}"

        try:
            self._lib = ctypes.WinDLL(str(self.dll_path))
        except Exception as exc:  # pragma: no cover - environment-specific
            raise ImportError(f"Failed to load {self.dll_path}: {exc}") from exc

        # Define signatures
        self._lib.ofiq_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.ofiq_init.restype = ctypes.c_int

        self._lib.ofiq_score_rgb.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._lib.ofiq_score_rgb.restype = ctypes.c_int

        rc = self._lib.ofiq_init(
            str(self.config_dir).encode("utf-8"),
            str(self.config_file).encode("utf-8"),
        )
        if rc != 0:
            raise RuntimeError(f"ofiq_init failed with code {rc} (config_dir={self.config_dir})")
        logging.info("Initialized OFIQ (config_dir=%s)", self.config_dir)

    def score(self, image_rgb: np.ndarray) -> Optional[float]:
        """
        Compute OFIQ UnifiedQualityScore for a single RGB image.
        Returns None if scoring fails.
        """
        if image_rgb is None:
            return None
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("OFIQ expects an RGB image with shape (H, W, 3)")
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)

        # OFIQ expects BGR planar bytes; reverse channels and ensure contiguous layout.
        bgr = np.ascontiguousarray(image_rgb[..., ::-1])
        height, width, _ = bgr.shape
        stride = int(bgr.strides[0])

        score_out = ctypes.c_double()
        rc = self._lib.ofiq_score_rgb(
            bgr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int(width),
            ctypes.c_int(height),
            ctypes.c_int(stride),
            ctypes.byref(score_out),
        )
        if rc != 0:
            logging.warning("OFIQ scoring failed with code %s", rc)
            return None
        return float(score_out.value)
