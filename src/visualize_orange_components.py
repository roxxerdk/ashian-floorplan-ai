
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "X"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLOR = np.array([255, 142, 70], dtype=np.float32)
TOLERANCE = 35


def get_file(folder, plan_id):
    return next(
        (
            p for p in folder.glob("*.png")
            if p.name.startswith(f"{plan_id}.")
            or p.name.startswith(f"{plan_id} ")
        ),
        None
    )


def visualize_plan(plan_id):

    image_path = get_file(IMAGE_DIR, plan_id)
    mask_path = get_file(MASK_DIR, plan_id)

    if image_path is None or mask_path is None:
        print(f"Files missing for plan {plan_id}")
        return

    original = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path))

    if original.shape[:2] != mask.shape[:2]:
        print(f"Dimension mismatch for plan {plan_id}")
        return

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    distance = np.linalg.norm(
        mask_rgb.astype(np.float32) - TARGET_COLOR,
        axis=2
    )

    binary = np.uint8(distance <= TOLERANCE)

    number, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    overlay = original.copy()

    for region_id in range(1, number):

        component = np.uint8(labels == region_id) * 255

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        color = (
            (region_id * 67) % 255,
            (region_id * 113) % 255,
            (region_id * 173) % 255
        )

        cv2.drawContours(
            overlay,
            contours,
            -1,
            color,
            thickness=6
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
        OUTPUT_DIR / f"orange_components_plan_{plan_id}.png"
    )

    cv2.imwrite(str(output_path), overlay)

    print(
        f"Plan {plan_id}: {number - 1} orange components"
    )
    print("Saved:", output_path)


if __name__ == "__main__":
    visualize_plan(16)
    visualize_plan(100)

