
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"

# Dominant colors observed in our dataset.
REFERENCE_COLORS = {
    "teal": (0, 142, 142),
    "orange": (255, 142, 70),
    "light_blue": (175, 216, 248),
    "yellow": (246, 189, 15),
    "green": (139, 186, 0),
    "red": (214, 70, 70),
    "black": (0, 0, 0),
}

references = np.array(
    list(REFERENCE_COLORS.values()),
    dtype=np.int16
)

names = list(REFERENCE_COLORS.keys())

color_totals = Counter()
unmatched_pixels = 0
total_pixels = 0

for index, mask_path in enumerate(
    sorted(MASK_DIR.glob("*.png")), start=1
):
    with Image.open(mask_path) as img:
        img.thumbnail((1000, 1000))
        pixels = np.asarray(img.convert("RGB")).reshape(-1, 3)

    # Calculate distance from every pixel to each reference color.
    # Convert to float32 before subtraction and squaring.
    pixel_values = pixels.astype(np.float32)

    distances = np.sqrt(
        np.sum(
            (pixel_values[:, None, :] -
            references.astype(np.float32)[None, :, :]) ** 2,
            axis=2
    )
)

    nearest = np.argmin(distances, axis=1)
    minimum_distance = np.min(distances, axis=1)

    # A pixel is matched if it is sufficiently close.
    matched = minimum_distance <= 35

    for class_index, name in enumerate(names):
        color_totals[name] += int(
            np.sum(matched & (nearest == class_index))
        )

    unmatched_pixels += int(np.sum(~matched))
    total_pixels += len(pixels)

    print(f"Processed {index}/121")

print("\n========== COLOR VARIATION ANALYSIS ==========")

for name in names:
    count = color_totals[name]
    percentage = count / total_pixels * 100
    print(f"{name:12}: {percentage:.3f}%")

unmatched_percentage = unmatched_pixels / total_pixels * 100

print(f"\nUnmatched pixels: {unmatched_percentage:.3f}%")
print("Total sampled pixels:", total_pixels)
