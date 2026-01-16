# Non-diagnostic, deterministic image checks (usability only).

import io
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from typing import Dict, List, Any
import logging

logger = logging.getLogger("sentinel.services.image_quality")

class ImageQualityService:
    """Deterministic service for assessing image usability risks (Resolution, Blur, Contrast)."""

    @staticmethod
    def load_image(image_bytes: bytes) -> Image.Image:
        return Image.open(io.BytesIO(image_bytes))

    @staticmethod
    def compute_quality(img: Image.Image) -> Dict[str, Any]:
        """Runs heuristic checks: Blur, Contrast, Resolution."""
        report = {
            "quality_issues": [],
            "workflow_risks": [],
            "visual_observations": [],
            "uncertainties": [],
            "meta": {}
        }

        try:
            width, height = img.size
            report["meta"] = {"width": width, "height": height, "format": img.format}

            # 1. Resolution
            if min(width, height) < 700:
                report["quality_issues"].append("Low Resolution: Small image dimensions may allow OCR errors.")

            # 2. Blur (Laplacian Approximation)
            gray = img.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)
            edge_var = edge_stat.var[0] if edge_stat.var else 0

            if edge_var < 100:
                report["quality_issues"].append("Blur Detected: Image content appears out of focus.")

            # 3. Contrast
            stat = ImageStat.Stat(gray)
            std_dev = stat.stddev[0]
            if std_dev < 20:
                report["quality_issues"].append("Low Contrast: Image is washed out or uniform.")
            elif std_dev < 40:
                report["workflow_risks"].append("Moderate Contrast: May be difficult for OCR.")

            # 4. Exposure
            mean_intensity = stat.mean[0]
            if mean_intensity < 10 or mean_intensity > 245:
                 report["quality_issues"].append("Exposure Issue: Image is too dark or too bright.")

            # 5. Orientation (Meta)
            if width > height:
                report["visual_observations"].append("Landscape orientation detected.")
            else:
                 report["visual_observations"].append("Portrait orientation detected.")

        except Exception as e:
            logger.error(f"Quality Check Failed: {e}")
            report["uncertainties"].append("Could not complete quality analysis on this file.")

        return report
