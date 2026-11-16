
from pathlib import Path
from collections import Counter
import json
import re

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLOR = np.array([0, 142, 142], dtype=np.float32)
COLOR_TOLERANCE = 35


def get_plan_id(filename):
    match = re.match(r"(\d+)", filename)
    return int(match.group(1)) if match else None


def extract_plan_regions(mask_path):

    mask = cv2.imread(str(mask_path))

    if mask is None:
        raise ValueError(f"Could not read {mask_path}")

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    distance = np.linalg.norm(
        mask.astype(np.float32) - TARGET_COLOR,
        axis=2
    )

    binary_mask = np.uint8(distance <= COLOR_TOLERANCE)

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
            "region_id": int(region_id),
            "area_pixels": int(area),
            "bounding_box": {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height)
            },
            "centroid": {
                "x": round(float(cx), 2),
                "y": round(float(cy), 2)
            },
            "semantic_label": None,
            "validation_status": "unverified"
        })

    return {
        "plan_id": get_plan_id(mask_path.name),
        "source_mask": mask_path.name,
        "image_width": int(mask.shape[1]),
        "image_height": int(mask.shape[0]),
        "target_color_rgb": TARGET_COLOR.astype(int).tolist(),
        "color_tolerance": COLOR_TOLERANCE,
        "connected_region_count": len(regions),
        "regions": regions
    }


def main():

    mask_paths = sorted(
        MASK_DIR.glob("*.png"),
        key=lambda p: get_plan_id(p.name) or 0
    )

    all_plans = []
    region_counts = Counter()

    for index, mask_path in enumerate(mask_paths, start=1):

        plan_data = extract_plan_regions(mask_path)

        all_plans.append(plan_data)

        region_counts[
            plan_data["connected_region_count"]
        ] += 1

        print(
            f"Processed {index}/{len(mask_paths)} | "
            f"Plan {plan_data['plan_id']} | "
            f"Regions: {plan_data['connected_region_count']}"
        )

    output_path = OUTPUT_DIR / "extracted_regions.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_plans, file, indent=4)

    print("\n========== EXTRACTION SUMMARY ==========")
    print("Plans processed:", len(all_plans))
    print("Region count distribution:", dict(region_counts))
    print("Output saved:", output_path)


if __name__ == "__main__":
    main()
