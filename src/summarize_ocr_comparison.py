import pandas as pd

path = "outputs/ocr/controlled_ocr_comparison.csv"
df = pd.read_csv(path)

summary = df.groupby(
    ["plan_id", "variant"]
).agg(
    detections=("text", "count"),
    average_confidence=("confidence", "mean"),
    median_confidence=("confidence", "median")
).round(2)

print("\n--- OCR COMPARISON SUMMARY ---\n")
print(summary.to_string())

print("\n--- DETECTION COUNTS ---")
print(df.groupby("variant")["text"].count())

print("\n--- SAMPLE DETECTIONS ---")
for plan_id in [1, 16, 40, 67, 85]:
    print(f"\nPlan {plan_id}")
    for variant in ["original", "preprocessed"]:
        words = df[
            (df["plan_id"] == plan_id) &
            (df["variant"] == variant)
        ]["text"].tolist()
        print(f"{variant}: {' '.join(words[:50])}")
