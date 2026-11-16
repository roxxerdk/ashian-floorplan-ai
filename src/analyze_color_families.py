
from email.mime import image
from pathlib import Path
from collections import Counter
import json

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "color_family_report.json"

THRESHOLD = 35

REFERENCE_COLORS = {
    "black": (0, 0, 0),
    "teal": (0, 142, 142),
    "orange": (255, 142, 70),
    "light_blue": (175, 216, 248),
    "yellow": (246, 189, 15),
    "green": (139, 186, 0),
    "red": (214, 70, 70),
}


def get_mask_files():
    return sorted(
        MASK_DIR.glob("*.png"),
        key=lambda p: int(p.stem.split()[0].split("(")[0])
    )


def analyze_mask(mask_path):
    with Image.open(mask_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    pixels = pixels.reshape(-1, 3)

    references = np.array(
    list(REFERENCE_COLORS.values()),
    dtype=np.float32
    )

    # Calculate Euclidean RGB distance to every reference colour.
    differences = pixels[:, None, :] - references[None, :, :]
    distances = np.sqrt(np.sum(differences ** 2, axis=2))

    nearest_indices = np.argmin(distances, axis=1)
    nearest_distances = np.min(distances, axis=1)

    # A pixel belongs to a family only if it is within the threshold.
    matched = nearest_distances <= THRESHOLD

    family_counts = Counter(
        nearest_indices[matched].tolist()
    )

    total_pixels = len(pixels)
    matched_pixels = int(np.sum(matched))
    unmatched_pixels = total_pixels - matched_pixels

    families = {}

    for index, name in enumerate(REFERENCE_COLORS):
        count = family_counts.get(index, 0)

        families[name] = {
            "reference_rgb": list(REFERENCE_COLORS[name]),
            "pixel_count": count,
            "percentage": round(count / total_pixels * 100, 5)
        }

    return {
        "filename": mask_path.name,
        "dimensions": list(Image.open(mask_path).size),
        "total_pixels": total_pixels,
        "families": families,
        "matched_pixels": matched_pixels,
        "matched_percentage": round(
            matched_pixels / total_pixels * 100, 5
        ),
        "unmatched_pixels": unmatched_pixels,
        "unmatched_percentage": round(
            unmatched_pixels / total_pixels * 100, 5
        )
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_files = get_mask_files()

    if not mask_files:
        raise FileNotFoundError(
            f"No PNG masks found in {MASK_DIR}"
        )

    results = []

    for mask_path in mask_files:
        result = analyze_mask(mask_path)
        results.append(result)

        print(
            f"{mask_path.name}: "
            f"matched {result['matched_percentage']}%, "
            f"unmatched {result['unmatched_percentage']}%"
        )

    report = {
        "total_masks": len(results),
        "threshold": THRESHOLD,
        "distance_metric": "Euclidean RGB",
        "reference_colors": {
            name: list(rgb)
            for name, rgb in REFERENCE_COLORS.items()
        },
        "masks": results
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nProcessed {len(results)} masks.")
    print(f"Report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
