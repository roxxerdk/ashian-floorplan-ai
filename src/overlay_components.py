
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "X"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLOR = np.array([0, 142, 142], dtype=np.float32)


def get_file(folder, plan_id):
    return next(
        (
            p for p in folder.glob("*.png")
            if p.name.startswith(f"{plan_id}.")
            or p.name.startswith(f"{plan_id} ")
        ),
        None
    )


def create_overlay(plan_id):
    image_path = get_file(IMAGE_DIR, plan_id)
    mask_path = get_file(MASK_DIR, plan_id)

    if image_path is None or mask_path is None:
        print(f"Files not found for plan {plan_id}")
        return

    original = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path))

    if original.shape[:2] != mask.shape[:2]:
        print(f"Dimension mismatch: plan {plan_id}")
        return

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    distance = np.linalg.norm(
        mask_rgb.astype(np.float32) - TARGET_COLOR,
        axis=2
    )

    binary = np.uint8(distance <= 35)

    number, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    overlay = original.copy()

    rng = np.random.default_rng(42)

    for region_id in range(1, number):
        component = np.uint8(labels == region_id) * 255

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        color = tuple(
            int(v) for v in rng.integers(40, 255, size=3)
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            color,
            thickness=5
        )

        cx, cy = centroids[region_id]

        cv2.circle(
            overlay,
            (int(cx), int(cy)),
            24,
            (0, 0, 0),
            -1
        )

        cv2.putText(
            overlay,
            str(region_id),
            (int(cx) - 12, int(cy) + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    output_path = (
        OUTPUT_DIR / f"components_overlay_plan_{plan_id}.png"
    )

    cv2.imwrite(str(output_path), overlay)

    print(
        f"Plan {plan_id}: {number - 1} components | "
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    create_overlay(2)
    create_overlay(3)
