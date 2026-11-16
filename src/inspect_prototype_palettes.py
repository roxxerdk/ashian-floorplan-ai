from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = ROOT / "data" / "raw" / "dataset" / "Y"

PLAN_IDS = [1, 16, 40, 67, 85]

for plan_id in PLAN_IDS:
    mask_path = MASK_DIR / f"{plan_id} (2).png"

    with Image.open(mask_path) as image:
        rgb = image.convert("RGB")
        counts = Counter(rgb.getdata())

    total = sum(counts.values())

    print(f"\n========== PLAN {plan_id} ==========")
    print("Unique RGB colours:", len(counts))

    for color, count in counts.most_common(15):
        percentage = count / total * 100
        print(
            f"RGB {color} | "
            f"Pixels: {count:,} | "
            f"Percentage: {percentage:.3f}%"
        )
