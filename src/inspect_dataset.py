
from pathlib import Path
from collections import Counter
import json
import re

from PIL import Image


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "X"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_numeric_id(filename):
    """Extract the numeric identifier from a filename."""
    match = re.match(r"(\d+)", filename)
    return int(match.group(1)) if match else None


def inspect_dataset():
    images = sorted(IMAGE_DIR.glob("*.png"))
    masks = sorted(MASK_DIR.glob("*.png"))

    image_map = {
        get_numeric_id(path.name): path
        for path in images
        if get_numeric_id(path.name) is not None
    }

    mask_map = {
        get_numeric_id(path.name): path
        for path in masks
        if get_numeric_id(path.name) is not None
    }

    missing_masks = sorted(set(image_map) - set(mask_map))
    missing_images = sorted(set(mask_map) - set(image_map))

    matched_pairs = []
    dimension_mismatches = []
    image_modes = Counter()
    mask_modes = Counter()
    image_dimensions = Counter()
    mask_dimensions = Counter()
    color_counts = Counter()

    for image_id in sorted(set(image_map) & set(mask_map)):
        image_path = image_map[image_id]
        mask_path = mask_map[image_id]

        with Image.open(image_path) as image:
            image_size = image.size
            image_modes[image.mode] += 1
            image_dimensions[str(image_size)] += 1

        with Image.open(mask_path) as mask:
            mask_size = mask.size
            mask_modes[mask.mode] += 1
            mask_dimensions[str(mask_size)] += 1

            # Sample mask pixels at reduced resolution.
            # This makes palette inspection faster.
            sample = mask.convert("RGB")
            sample.thumbnail((500, 500))

            color_counts.update(sample.getdata())

        if image_size != mask_size:
            dimension_mismatches.append({
                "id": image_id,
                "image_size": image_size,
                "mask_size": mask_size
            })

        matched_pairs.append(image_id)

    top_colors = [
        {
            "rgb": list(color),
            "sampled_pixel_count": count
        }
        for color, count in color_counts.most_common(30)
    ]

    report = {
        "dataset_summary": {
            "total_images": len(images),
            "total_masks": len(masks),
            "matched_pairs": len(matched_pairs),
            "missing_masks": missing_masks,
            "missing_images": missing_images,
            "dimension_mismatches": dimension_mismatches
        },
        "image_properties": {
            "modes": dict(image_modes),
            "dimensions": dict(image_dimensions)
        },
        "mask_properties": {
            "modes": dict(mask_modes),
            "dimensions": dict(mask_dimensions)
        },
        "top_mask_colors": top_colors,
        "note": (
            "Mask colors are observed values only. "
            "Semantic class labels have not yet been verified."
        )
    }

    report_path = OUTPUT_DIR / "dataset_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n========== DATASET INSPECTION ==========")
    print(f"Images found:       {len(images)}")
    print(f"Masks found:        {len(masks)}")
    print(f"Matched pairs:      {len(matched_pairs)}")
    print(f"Missing masks:      {len(missing_masks)}")
    print(f"Missing images:     {len(missing_images)}")
    print(f"Dimension issues:   {len(dimension_mismatches)}")

    print("\nImage modes:", dict(image_modes))
    print("Mask modes:", dict(mask_modes))

    print("\nTop 15 mask colors:")
    for color, count in color_counts.most_common(15):
        print(f"RGB {color}: {count} sampled pixels")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    inspect_dataset()
