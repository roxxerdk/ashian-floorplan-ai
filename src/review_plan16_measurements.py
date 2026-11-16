from PIL import Image

image = Image.open("data/raw/dataset/X/16.png")

crop = image.crop((850, 1250, 1700, 1500))
crop = crop.resize((1700, 500))

crop.save("outputs/plan16_measurement_review.png")

print("Saved: outputs/plan16_measurement_review.png")
