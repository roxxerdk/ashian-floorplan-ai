
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"

TARGET_COLOR = np.array([255, 142, 70], dtype=np.float32)
TOLERANCE = 35


def analyze_plan(plan_id):

    mask_path = next(
        (
            p for p in MASK_DIR.glob("*.png")
            if p.name.startswith(f"{plan_id} ")
        ),
        None
    )

    if mask_path is None:
        print(f"Plan {plan_id}: mask not found")
        return

    mask = cv2.imread(str(mask_path))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    distance = np.linalg.norm(
        mask.astype(np.float32) - TARGET_COLOR,
        axis=2
    )

    binary = np.uint8(distance <= TOLERANCE)

    number, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    regions = []

    for region_id in range(1, number):

        x, y, width, height, area = stats[region_id]
        cx, cy = centroids[region_id]

        regions.append({
            "id": region_id,
            "area": int(area),
            "bbox": (int(x), int(y), int(width), int(height)),
            "centroid": (round(float(cx), 2), round(float(cy), 2))
        })

    regions.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    print(f"\n========== PLAN {plan_id} ==========")
    print("Dimensions:", mask.shape)
    print("Orange connected regions:", number - 1)

    for region in regions[:20]:
        print(
            f"Region {region['id']} | "
            f"Area: {region['area']} px | "
            f"Centroid: {region['centroid']}"
        )


if __name__ == "__main__":
    analyze_plan(16)
    analyze_plan(100)
