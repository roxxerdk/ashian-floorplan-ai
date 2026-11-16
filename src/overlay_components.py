
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pytesseract
from pytesseract import Output


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "X"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "overlays"

PLAN_IDS = [16, 40, 85]

COLOR_TOLERANCE = 35
MIN_COMPONENT_AREA = 100

OCR_LANGUAGE = "fra+eng"
OCR_CONFIG = "--psm 11"

# Observed RGB palette values.
# Semantic class mapping has not been verified.
PALETTE = {
    "Palette 1": (0, 142, 142),
    "Palette 2": (255, 142, 70),
    "Palette 3": (175, 216, 248),
    "Palette 4": (246, 189, 15),
    "Palette 5": (139, 186, 0),
    "Palette 6": (214, 70, 70),
}


def find_plan_file(folder, plan_id):
    matches = [
        path for path in folder.glob("*.png")
        if path.name.startswith(f"{plan_id} ")
        or path.name.startswith(f"{plan_id}.")
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Plan {plan_id}: expected one file in {folder}, "
            f"found {[p.name for p in matches]}"
        )

    return matches[0]


def crop_plan(image, mask):
    """Crop excess white margins using the original drawing."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Include faint drawing and dimension lines.
    foreground = gray < 250

    ys, xs = np.where(foreground)

    if len(xs) == 0:
        return image, mask

    height, width = gray.shape

    margin_x = max(20, int(width * 0.02))
    margin_y = max(20, int(height * 0.02))

    x1 = max(0, int(xs.min()) - margin_x)
    y1 = max(0, int(ys.min()) - margin_y)
    x2 = min(width, int(xs.max()) + margin_x + 1)
    y2 = min(height, int(ys.max()) + margin_y + 1)

    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def extract_components(mask):
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    results = []

    for palette_name, rgb in PALETTE.items():

        target = np.array(rgb, dtype=np.float32)

        distance = np.linalg.norm(
            mask_rgb.astype(np.float32) - target,
            axis=2
        )

        binary = np.uint8(distance <= COLOR_TOLERANCE)

        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        for component_id in range(1, count):

            area = int(stats[component_id, cv2.CC_STAT_AREA])

            if area < MIN_COMPONENT_AREA:
                continue

            component = np.uint8(labels == component_id) * 255

            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            results.append({
                "palette": palette_name,
                "rgb": rgb,
                "area": area,
                "contours": contours
            })

    return results


def detect_text(original):
    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    result = pytesseract.image_to_data(
        rgb,
        lang=OCR_LANGUAGE,
        config=OCR_CONFIG,
        output_type=Output.DICT
    )

    detections = []

    for i, raw_text in enumerate(result["text"]):

        text = raw_text.strip()

        if not text:
            continue

        try:
            confidence = float(result["conf"][i])
        except (ValueError, TypeError):
            continue

        if confidence < 35:
            continue

        x = int(result["left"][i])
        y = int(result["top"][i])
        w = int(result["width"][i])
        h = int(result["height"][i])

        if w <= 0 or h <= 0:
            continue

        detections.append({
            "text": text,
            "box": (x, y, x + w, y + h)
        })

    return detections


def create_overlay(plan_id):

    image_path = find_plan_file(IMAGE_DIR, plan_id)
    mask_path = find_plan_file(MASK_DIR, plan_id)

    original = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path))

    if original is None or mask is None:
        raise RuntimeError(f"Could not read plan {plan_id}")

    if original.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Dimension mismatch for plan {plan_id}"
        )

    # Crop both images using exactly the same coordinates.
    original, mask = crop_plan(original, mask)

    components = extract_components(mask)
    detections = detect_text(original)

    overlay = original.copy()

    # Draw geometric mask components.
    for component in components:

        rgb = component["rgb"]
        bgr = (rgb[2], rgb[1], rgb[0])

        cv2.drawContours(
            overlay,
            component["contours"],
            -1,
            bgr,
            thickness=4
        )

    # Draw OCR bounding boxes.
    for detection in detections:

        x1, y1, x2, y2 = detection["box"]

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    counts = Counter(
        component["palette"] for component in components
    )

    # Readable legend.
    legend_height = 320

    canvas = cv2.copyMakeBorder(
        overlay,
        legend_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(25, 25, 25)
    )

    cv2.putText(
        canvas,
        f"PLAN {plan_id} - MASK PALETTE AND OCR OVERLAY",
        (30, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )

    y = 85

    for palette_name, rgb in PALETTE.items():

        bgr = (rgb[2], rgb[1], rgb[0])

        cv2.line(
            canvas,
            (32, y - 7),
            (65, y - 7),
            bgr,
            9
        )

        cv2.putText(
            canvas,
            f"{palette_name} RGB {rgb}: "
            f"{counts[palette_name]} components",
            (85, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        y += 34

    cv2.putText(
        canvas,
        f"Green boxes: OCR detections ({len(detections)})",
        (30, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        canvas,
        "WARNING: Palette colours are observed segmentation values; "
        "semantic class mapping is unverified.",
        (30, y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1
    )

    output_path = OUTPUT_DIR / f"overlay_plan_{plan_id}.png"

    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to save {output_path}")

    print(f"\nPLAN {plan_id}")
    print(f"Original: {image_path.name}")
    print(f"Mask: {mask_path.name}")
    print(f"Cropped dimensions: {original.shape[1]} x {original.shape[0]}")

    for palette_name in PALETTE:
        print(
            f"  {palette_name}: {counts[palette_name]} components"
        )

    print(f"  OCR detections: {len(detections)}")
    print(f"  Saved: {output_path}")


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for plan_id in PLAN_IDS:
        try:
            create_overlay(plan_id)
        except Exception as exc:
            print(f"\nERROR Plan {plan_id}: {exc}")


if __name__ == "__main__":
    main()
