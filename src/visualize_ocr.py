from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

image = Image.open("data/raw/dataset/X/16.png").convert("RGB")
data = pd.read_csv("outputs/plan16_ocr_detections.csv")

draw = ImageDraw.Draw(image)

for _, row in data.iterrows():
    x, y = int(row["left"]), int(row["top"])
    w, h = int(row["width"]), int(row["height"])
    text = str(row["text"])

    draw.rectangle((x, y, x + w, y + h), outline="red", width=3)
    draw.text((x, max(0, y - 18)), text, fill="blue")

output = Path("outputs/plan16_ocr_overlay.png")
image.save(output)

print("Saved:", output)
