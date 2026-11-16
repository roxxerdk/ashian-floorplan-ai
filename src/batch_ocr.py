from pathlib import Path
import pandas as pd
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

plan_ids = [1, 16, 40, 67, 85]
image_dir = Path("data/raw/dataset/X")
output_dir = Path("outputs/ocr")
output_dir.mkdir(parents=True, exist_ok=True)

all_results = []

for plan_id in plan_ids:
    image_path = image_dir / f"{plan_id}.png"

    if not image_path.exists():
        print(f"Plan {plan_id}: IMAGE NOT FOUND")
        continue

    image = Image.open(image_path).convert("RGB")

    data = pytesseract.image_to_data(
        image,
        lang="eng",
        output_type=pytesseract.Output.DATAFRAME
    )

    data = data.dropna(subset=["text"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    data = data[data["conf"] >= 0]
    data.insert(0, "plan_id", plan_id)

    all_results.append(data)

    print(f"\n{'=' * 45}")
    print(f"PLAN {plan_id} | Dimensions: {image.size}")
    print("=" * 45)
    print(" ".join(data["text"].tolist()))

if all_results:
    combined = pd.concat(all_results, ignore_index=True)
    output_path = output_dir / "five_plan_ocr.csv"
    combined.to_csv(output_path, index=False)

    print(f"\nSaved {len(combined)} detections to {output_path}")
