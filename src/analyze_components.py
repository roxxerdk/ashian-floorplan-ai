
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"

TARGET_COLOR = np.array([0, 142, 142], dtype=np.float32)


def analyze_plan(plan_id):

    mask_path = next(
        (
            p for p in MASK_DIR.glob("*.png")
            if p.name.startswith(f"{plan_id} ")
        ),
        None
    )

    if mask_path is None:
        print(f"Mask not found for plan {plan_id}")
        return

    mask = cv2.imread(str(mask_path))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    distance = np.linalg.norm(
        mask.astype(np.float32) - TARGET_COLOR,
        axis=2
    )

    binary_mask = np.uint8(distance <= 35)

    number, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
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
            "centroid": (round(cx, 2), round(cy, 2))
        })

    regions.sort(
        key=lambda item: item["area"],
        reverse=True
    )

    print(f"\n========== PLAN {plan_id} ==========")
    print("Mask:", mask_path.name)
    print("Dimensions:", mask.shape)
    print("Connected regions:", number - 1)

    print("\nTop 15 largest regions:")

    for region in regions[:15]:
        print(
            f"Region {region['id']} | "
            f"Area: {region['area']} px | "
            f"Centroid: {region['centroid']}"
        )


if __name__ == "__main__":
    for plan_id in [2, 3]:
        analyze_plan(plan_id)
