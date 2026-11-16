from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "dataset"
OUTPUT = ROOT / "outputs" / "overlays"

PLAN_IDS = [1, 16, 40, 67, 85]


def validate(plan_id):
    image_path = DATA / "X" / f"{plan_id}.png"
    mask_path = DATA / "Y" / f"{plan_id} (2).png"

    with Image.open(image_path) as img:
        original = np.asarray(img.convert("RGB"))

    with Image.open(mask_path) as img:
        mask = np.asarray(img.convert("RGB"))

    if original.shape != mask.shape:
        print(f"Plan {plan_id}: SHAPE MISMATCH")
        return

    # Check whether the mask has any non-black pixels.
    foreground = np.any(mask != 0, axis=2)

    # Report the spatial extent of the mask foreground.
    ys, xs = np.where(foreground)

    if len(xs) == 0:
        print(f"Plan {plan_id}: mask contains no non-black pixels")
        return

    print(f"\nPLAN {plan_id}")
    print(f"Image shape: {original.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Non-black mask pixels: {len(xs)}")
    print(
        "Mask foreground bounding box:",
        (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    )
    print(
        "Foreground coverage:",
        round(len(xs) / foreground.size * 100, 2), "%"
    )
    print("Dimension check: PASS")


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for plan_id in PLAN_IDS:
        validate(plan_id)
