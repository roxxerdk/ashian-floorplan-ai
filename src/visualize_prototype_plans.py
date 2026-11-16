from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "dataset"
OUTPUT = ROOT / "outputs" / "overlays"

PLAN_IDS = [1, 16, 40, 67, 85]
PANEL_WIDTH = 420
HEADER_HEIGHT = 30
ROW_HEIGHT = 260

OUTPUT.mkdir(parents=True, exist_ok=True)

sheet = Image.new(
    "RGB",
    (PANEL_WIDTH * 3, (ROW_HEIGHT + HEADER_HEIGHT) * len(PLAN_IDS)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for row, plan_id in enumerate(PLAN_IDS):
    with Image.open(DATA / "X" / f"{plan_id}.png") as img:
        original = img.convert("RGB")

    with Image.open(DATA / "Y" / f"{plan_id} (2).png") as img:
        mask = img.convert("RGB")

    if original.size != mask.size:
        print(f"Skipping plan {plan_id}: dimension mismatch")
        continue

    overlay = Image.blend(original, mask, alpha=0.45)

    ratio = min(
        PANEL_WIDTH / original.width,
        ROW_HEIGHT / original.height
    )
    size = (
        int(original.width * ratio),
        int(original.height * ratio)
    )

    panels = [
        ("Original", original),
        ("Mask", mask),
        ("Overlay", overlay)
    ]

    y = row * (ROW_HEIGHT + HEADER_HEIGHT)

    for col, (label, panel) in enumerate(panels):
        x = col * PANEL_WIDTH
        draw.text((x + 8, y + 8), f"Plan {plan_id} - {label}", fill="black")
        resized = panel.resize(size)
        sheet.paste(resized, (x, y + HEADER_HEIGHT))

output_path = OUTPUT / "prototype_contact_sheet.png"
sheet.save(output_path)

print("Saved:", output_path)
