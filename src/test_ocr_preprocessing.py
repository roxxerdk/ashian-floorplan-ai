from pathlib import Path
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

source = "data/raw/dataset/X/16.png"
image = cv2.imread(source)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
upscaled = cv2.resize(
    gray, None, fx=2, fy=2,
    interpolation=cv2.INTER_CUBIC
)

processed = cv2.adaptiveThreshold(
    upscaled,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    11
)

output = Path("outputs/plan16_preprocessed.png")
cv2.imwrite(str(output), processed)

config = "--psm 11"

text = pytesseract.image_to_string(
    processed,
    lang="fra",
    config=config
)

print("--- PREPROCESSED FRENCH OCR ---")
print(text)
print(f"\nSaved processed image: {output}")
