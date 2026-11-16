from pathlib import Path
import pandas as pd
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = Path("data/raw/dataset/X/16.png")
image = Image.open(image_path)

data = pytesseract.image_to_data(
    image,
    lang="eng",
    output_type=pytesseract.Output.DATAFRAME
)

data = data.dropna(subset=["text"])
data["text"] = data["text"].astype(str).str.strip()
data = data[data["text"] != ""]
data = data[data["conf"] >= 0]

output_path = Path("outputs/plan16_ocr_detections.csv")
data.to_csv(output_path, index=False)

print(data[["text", "conf", "left", "top", "width", "height"]].to_string(index=False))
print(f"\nSaved OCR detections to: {output_path}")
