# This module is intentionally non-diagnostic and deterministic.
# Do not extend with AI-based vision analysis.
#
# Product Rationale:
# Vision-language models were intentionally excluded to ensure deterministic,
# safe, and non-diagnostic behavior. Image handling is limited to usability checks.

import io
import numpy as np
from PIL import Image, ImageStat, ImageFilter
from typing import Dict, List, Any, Union

def load_image(image_input: Union[bytes, str]) -> Image.Image:
    """Loads an image from bytes or path into a PIL Image."""
    if isinstance(image_input, str):
        return Image.open(image_input)
    elif isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input))
    else:
        raise ValueError("Input must be bytes or file path")

def compute_quality(image: Image.Image) -> Dict[str, Any]:
    """
    Computes heuristic quality metrics for a document image.
    Deterministic checks: resolution, blur, contrast, margins.
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    grayscale = image.convert("L")
    img_array = np.array(grayscale)

    issues = []
    risks = []
    observations = []
    uncertainties = []

    # 1. Resolution Check
    min_dim = min(width, height)
    if min_dim < 700:
        issues.append("Low resolution; may reduce legibility.")
    else:
        observations.append(f"Resolution adequate ({width}x{height})")

    # 2. Blur Check (Variance of Laplacian)
    # Using Pillow's Kernel filter to approximate Laplacian if cv2 missing
    try:
        # Standard 3x3 Laplacian kernel
        laplacian_kernel = (
            0, -1, 0,
            -1, 4, -1,
            0, -1, 0
        )
        edges = grayscale.filter(ImageFilter.Kernel((3, 3), laplacian_kernel, scale=1, offset=0))
        # Variance of the edge map
        edge_var = np.var(np.array(edges))

        # Threshold implies "blurriness". Document images usually have sharp edges (high var).
        # < 100 is often quite blurry for text.
        if edge_var < 100:
            issues.append("Image appears blurry; may reduce legibility.")
            uncertainties.append(f"Blur score: {edge_var:.1f} (Low)")
        else:
            observations.append("Sharpness check passed.")

    except Exception as e:
        uncertainties.append(f"Could not compute blur score: {str(e)}")

    # 3. Contrast Check (Std Dev of Grayscale)
    stats = ImageStat.Stat(grayscale)
    std_dev = stats.stddev[0]
    if std_dev < 20: # Flat gray/white image
        issues.append("Low contrast; content may be hard to read.")
    elif std_dev < 40:
        risks.append("Moderate contrast; verify text visibility.")

    # 4. Cropping / Empty Margins
    # Simple check: bright pixels near borders?
    # This is tricky without advanced segmentation, lets rely on simple "Is it mostly white?"
    if np.mean(img_array) > 250:
        issues.append("Image appears blank or overexposed.")
    elif np.mean(img_array) < 5:
        issues.append("Image appears too dark.")

    # 5. Orientation
    if width > height * 1.5:
        observations.append("Landscape orientation detected.")
    elif height > width * 1.5:
        observations.append("Portrait orientation detected.")

    # Success Condition
    if not issues and not risks:
        observations.append("Image appears readable and complete (quality check only).")

    return {
        "visual_observations": observations,
        "quality_issues": issues,
        "workflow_risks": risks,
        "uncertainties": uncertainties,
        "meta": {"width": width, "height": height, "format": image.format}
    }
