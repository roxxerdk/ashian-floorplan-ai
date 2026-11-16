import json
import re
import pandas as pd
from pathlib import Path

claude_path = Path("outputs/plan16_claude_vision.json")
ocr_path = Path("outputs/plan16_ocr_detections.csv")

claude = json.loads(claude_path.read_text(encoding="utf-8"))
ocr = pd.read_csv(ocr_path)

ocr_text = " ".join(ocr["text"].dropna().astype(str).tolist())

def normalize(value):
    return str(value).replace(",", ".").strip()

def find_measurement(value):
    target = normalize(value)
    numbers = re.findall(r"\d+[.,]\d+", ocr_text)
    return any(normalize(number) == target for number in numbers)

report = []

for room in claude["rooms_and_spaces"]:
    area = room.get("printed_area_m2")

    if area is None:
        continue

    matched = find_measurement(area)

    report.append({
        "label": room["label_original"],
        "claude_area": area,
        "found_in_tesseract": matched,
        "status": "MATCHED" if matched else "REVIEW_REQUIRED"
    })

print("\n--- CLAUDE VS TESSERACT: PLAN 16 ---\n")

for item in report:
    print(
        f'{item["label"]:12} | '
        f'Claude: {item["claude_area"]:>6} | '
        f'Tesseract: {item["status"]}'
    )

output = Path("outputs/plan16_validation_report.csv")
pd.DataFrame(report).to_csv(output, index=False)

print(f"\nReport saved: {output}")
print("Note: A match means the number was detected, not that its room association is correct.")
