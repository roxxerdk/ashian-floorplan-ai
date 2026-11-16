
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"

PLAN_IDS = [1, 16, 100]


def inspect_mask(plan_id):

    mask_path = next(
        p for p in MASK_DIR.glob("*.png")
        if p.name.startswith(f"{plan_id} ")
    )

    with Image.open(mask_path) as img:
        img.thumbnail((1000, 1000))
        pixels = np.asarray(img.convert("RGB"))

    pixels = pixels.reshape(-1, 3)

    # Count exact RGB values in the resized sample.
    colors, counts = np.unique(
        pixels,
        axis=0,
        return_counts=True
    )

    order = np.argsort(counts)[::-1]

    print(f"\n========== PLAN {plan_id} ==========")
    print("Mask:", mask_path.name)

    for index in order[:12]:
        color = tuple(int(v) for v in colors[index])
        percentage = counts[index] / len(pixels) * 100

        print(
            f"RGB {color} | "
            f"{percentage:.3f}%"
        )


for plan_id in PLAN_IDS:
    inspect_mask(plan_id)
