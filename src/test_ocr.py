from pathlib import Path
from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

image_path = Path("data/raw/dataset/X/16.png")
image = Image.open(image_path)

print("Image:", image_path)
print("Dimensions:", image.size)
print("\n--- OCR OUTPUT (Plan 16) ---\n")

text = pytesseract.image_to_string(image, lang="eng")
print(text if text.strip() else "[No text detected]")
