
from pathlib import Path
from collections import Counter
import json

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "mask_color_inventory.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(
        MASK_DIR.glob("*.png"),
        key=lambda path: int(path.stem.split()[0].split("(")[0])
    )

    report = {
        "total_masks": len(mask_files),
        "masks": []
    }

    for mask_path in mask_files:
        with Image.open(mask_path) as image:
            rgb_image = image.convert("RGB")
            color_counts = Counter(rgb_image.getdata())

        total_pixels = sum(color_counts.values())

        colors = []
        for color, count in color_counts.most_common():
            colors.append({
                "rgb": list(color),
                "pixel_count": count,
                "percentage": round(count / total_pixels * 100, 5)
            })

        report["masks"].append({
            "filename": mask_path.name,
            "dimensions": list(rgb_image.size),
            "unique_colors": len(color_counts),
            "dominant_colors": colors[:20]
        })

        print(
            f"Processed {mask_path.name}: "
            f"{len(color_counts)} unique colors"
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nProcessed {len(mask_files)} masks.")
    print(f"Report saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
