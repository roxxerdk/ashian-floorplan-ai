from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "dataset"
OUTPUT = ROOT / "outputs" / "overlays"

PLAN_IDS = [1, 16, 40, 67, 85]
TOLERANCE = 35
MIN_AREA = 100

COLORS = {
    "teal": (0, 142, 142),
    "orange": (255, 142, 70),
    "yellow": (246, 189, 15),
    "green": (139, 186, 0),
    "red": (214, 70, 70),
    "light_blue": (175, 216, 248),
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for plan_id in PLAN_IDS:
        image_path = DATA / "X" / f"{plan_id}.png"
        mask_path = DATA / "Y" / f"{plan_id} (2).png"

        with Image.open(image_path) as img:
            original = np.array(img.convert("RGB"))

        with Image.open(mask_path) as img:
            mask = np.array(img.convert("RGB"))

        total_pixels = mask.shape[0] * mask.shape[1]
        family_counts = {}

        for name, color in COLORS.items():
            distance = np.linalg.norm(
                mask.astype(np.float32) -
                np.array(color, dtype=np.float32),
                axis=2
            )
            family_counts[name] = int(np.count_nonzero(distance <= TOLERANCE))

        dominant_name = max(family_counts, key=family_counts.get)
        dominant_color = COLORS[dominant_name]

        distance = np.linalg.norm(
            mask.astype(np.float32) -
            np.array(dominant_color, dtype=np.float32),
            axis=2
        )

        binary = np.uint8(distance <= TOLERANCE)

        number, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        overlay = cv2.cvtColor(
            original.copy(),
            cv2.COLOR_RGB2BGR
        )

        component_count = 0

        for component_id in range(1, number):
            x, y, width, height, area = stats[component_id]

            if area < MIN_AREA:
                continue

            component = np.uint8(labels == component_id) * 255

            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            component_count += 1

            cv2.drawContours(
                overlay,
                contours,
                -1,
                (0, 0, 255),
                5
            )

            cx, cy = centroids[component_id]

            cv2.putText(
                overlay,
                str(component_count),
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 0),
                4
            )

        output_path = (
            OUTPUT / f"dominant_components_plan_{plan_id}.png"
        )

        cv2.imwrite(str(output_path), overlay)

        print(
            f"Plan {plan_id} | "
            f"Dominant family: {dominant_name} | "
            f"Components: {component_count} | "
            f"Coverage: {family_counts[dominant_name] / total_pixels * 100:.2f}%"
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
