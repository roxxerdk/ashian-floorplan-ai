from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = ROOT / "data" / "raw" / "dataset" / "Y"

PLAN_IDS = [1, 16, 40, 67, 85]
TOLERANCE = 35
MIN_AREA = 100

REFERENCE_COLORS = {
    "teal": (0, 142, 142),
    "orange": (255, 142, 70),
    "light_blue": (175, 216, 248),
    "yellow": (246, 189, 15),
    "green": (139, 186, 0),
    "red": (214, 70, 70),
}


def analyze_family(mask, color):
    reference = np.array(color, dtype=np.float32)

    distance = np.linalg.norm(
        mask.astype(np.float32) - reference,
        axis=2
    )

    binary = np.uint8(distance <= TOLERANCE)

    number, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    components = []

    for component_id in range(1, number):
        x, y, width, height, area = stats[component_id]

        if area < MIN_AREA:
            continue

        cx, cy = centroids[component_id]

        components.append({
            "area": int(area),
            "bbox": (int(x), int(y), int(width), int(height)),
            "centroid": (round(float(cx), 2), round(float(cy), 2))
        })

    components.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    return int(binary.sum()), components


def main():
    for plan_id in PLAN_IDS:
        mask_path = MASK_DIR / f"{plan_id} (2).png"

        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("RGB"))

        total_pixels = mask.shape[0] * mask.shape[1]

        print(f"\n{'=' * 55}")
        print(f"PLAN {plan_id} | Dimensions: {mask.shape[1]}x{mask.shape[0]}")
        print(f"{'=' * 55}")

        for name, color in REFERENCE_COLORS.items():
            pixel_count, components = analyze_family(mask, color)

            percentage = pixel_count / total_pixels * 100

            print(f"\n{name.upper()} {color}")
            print(f"Pixel coverage: {percentage:.3f}%")
            print(f"Components above {MIN_AREA} pixels: {len(components)}")

            for index, component in enumerate(components[:5], start=1):
                print(
                    f"  #{index} | "
                    f"Area: {component['area']:,} px | "
                    f"BBox: {component['bbox']} | "
                    f"Centroid: {component['centroid']}"
                )


if __name__ == "__main__":
    main()
