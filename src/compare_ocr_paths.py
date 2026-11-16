from pathlib import Path
import cv2
import pandas as pd
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

plan_ids = [1, 16, 40, 67, 85]
output_dir = Path("outputs/ocr")
output_dir.mkdir(parents=True, exist_ok=True)

results = []

for plan_id in plan_ids:
    path = Path(f"data/raw/dataset/X/{plan_id}.png")
    image = cv2.imread(str(path))

    if image is None:
        print(f"Could not read Plan {plan_id}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(
        gray, None, fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC
    )
    processed = cv2.adaptiveThreshold(
        enlarged, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )

    for variant, source in [
        ("original", gray),
        ("preprocessed", processed)
    ]:
        data = pytesseract.image_to_data(
            source,
            lang="fra",
            config="--psm 11",
            output_type=pytesseract.Output.DATAFRAME
        )

        data = data.dropna(subset=["text"]).copy()
        data["text"] = data["text"].astype(str).str.strip()
        data = data[
            (data["text"] != "") & (data["conf"] >= 0)
        ]

        for _, item in data.iterrows():
            results.append({
                "plan_id": plan_id,
                "variant": variant,
                "text": item["text"],
                "confidence": item["conf"],
                "left": item["left"],
                "top": item["top"],
                "width": item["width"],
                "height": item["height"]
            })

    print(f"Completed Plan {plan_id}")

result = pd.DataFrame(results)
output_path = output_dir / "controlled_ocr_comparison.csv"
result.to_csv(output_path, index=False)

print(f"\nTotal detections: {len(result)}")
print(f"Saved comparison: {output_path}")
