
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "X"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "dataset" / "Y"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Locate floor plan 1 and its mask.
image_path = next(
    p for p in IMAGE_DIR.glob("*.png")
    if p.name.startswith("1.")
)

mask_path = next(
    p for p in MASK_DIR.glob("*.png")
    if p.name.startswith("1 ")
)

original = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path))

mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

target = np.array([0, 142, 142], dtype=np.float32)

distance = np.linalg.norm(
    mask.astype(np.float32) - target,
    axis=2
)

binary = np.uint8(distance <= 35)

number, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )
)

# Create a colour image for all detected components.
visual = np.zeros(
    (mask.shape[0], mask.shape[1], 3),
    dtype=np.uint8
)

rng = np.random.default_rng(42)

for region_id in range(1, number):
    color = rng.integers(50, 256, size=3).tolist()

    visual[labels == region_id] = color

# Draw component numbers at their centroids.
for region_id in range(1, number):
    x, y = centroids[region_id]

    cv2.putText(
        visual,
        str(region_id),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

# Save the component visualization.
output_path = OUTPUT_DIR / "components_plan_1.png"

cv2.imwrite(
    str(output_path),
    cv2.cvtColor(visual, cv2.COLOR_RGB2BGR)
)

print("Components detected:", number - 1)
print("Saved visualization:", output_path)
