
from pathlib import Path
from collections import Counter
import json

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

color_counts = Counter()
total_pixels = 0

mask_paths = sorted(MASK_DIR.glob("*.png"))

for index, mask_path in enumerate(mask_paths, start=1):

    with Image.open(mask_path) as img:
        # Reduce image size for efficient palette inspection.
        img.thumbnail((1000, 1000))
        mask = np.asarray(img.convert("RGB"))

    # Pack RGB values into single integers.
    pixels = mask.astype(np.uint32)

    packed = (
        (pixels[:, :, 0] << 16)
        | (pixels[:, :, 1] << 8)
        | pixels[:, :, 2]
    ).ravel()

    colors, counts = np.unique(
        packed,
        return_counts=True
    )

    for color, count in zip(colors, counts):
        color_counts[int(color)] += int(count)

    total_pixels += packed.size

    print(
        f"Processed {index}/{len(mask_paths)}: "
        f"{mask_path.name}"
    )

palette = []

for color, count in color_counts.most_common():
    rgb = [
        (color >> 16) & 255,
        (color >> 8) & 255,
        color & 255
    ]

    palette.append({
        "rgb": rgb,
        "sampled_pixel_count": count,
        "percentage": round(count / total_pixels * 100, 5)
    })

report = {
    "total_sampled_pixels": total_pixels,
    "unique_colors": len(palette),
    "colors": palette
}

output_path = OUTPUT_DIR / "mask_palette.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\n========== MASK PALETTE ==========")
print("Masks processed:", len(mask_paths))
print("Sampled pixels:", total_pixels)
print("Unique colors:", len(palette))

print("\nTop 20 colors:")

for item in palette[:20]:
    print(
        f"RGB {tuple(item['rgb'])} | "
        f"{item['percentage']}%"
    )

print("\nReport saved:", output_path)
