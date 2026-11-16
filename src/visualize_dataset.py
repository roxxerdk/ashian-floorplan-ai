
from pathlib import Path
from PIL import Image, ImageDraw
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "X"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_id(filename):
    match = re.match(r"(\d+)", filename)
    return int(match.group(1)) if match else None


def visualize_pair(image_id):
    image_path = next(
        (p for p in IMAGE_DIR.glob("*.png") if get_id(p.name) == image_id),
        None
    )
    mask_path = next(
        (p for p in MASK_DIR.glob("*.png") if get_id(p.name) == image_id),
        None
    )

    if image_path is None or mask_path is None:
        print(f"Pair {image_id} not found.")
        return

    with Image.open(image_path) as img:
        original = img.convert("RGB")

    with Image.open(mask_path) as img:
        mask = img.convert("RGB")

    if original.size != mask.size:
        print(f"Dimension mismatch for pair {image_id}")
        return

    overlay = Image.blend(original, mask, alpha=0.45)

    panel_width = 600
    panel_height = int(original.height * panel_width / original.width)

    original = original.resize((panel_width, panel_height))
    mask = mask.resize((panel_width, panel_height))
    overlay = overlay.resize((panel_width, panel_height))

    header_height = 40
    canvas = Image.new(
        "RGB",
        (panel_width * 3, panel_height + header_height),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    panels = [
        ("Original Floor Plan", original),
        ("Segmentation Mask", mask),
        ("Overlay", overlay)
    ]

    for index, (label, panel) in enumerate(panels):
        x = index * panel_width
        draw.text((x + 10, 12), label, fill="black")
        canvas.paste(panel, (x, header_height))

    output_path = OUTPUT_DIR / f"visualization_{image_id}.png"
    canvas.save(output_path)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    for sample_id in [40, 67, 85]:
        visualize_pair(sample_id)