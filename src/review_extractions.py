import json
from pathlib import Path

output_dir = Path("outputs/extracted")

for path in sorted(output_dir.glob("plan_*.json")):

    if path.name == "processing_log.json":
        continue

    data = json.loads(path.read_text(encoding="utf-8"))

    print(f"\n{'=' * 45}")
    print(f"PLAN {data['plan_id']}")
    print("=" * 45)

    print("\nROOMS AND SPACES:")

    for room in data.get("rooms_and_spaces", []):
        print(
            f"{room['label_original']}: "
            f"{room.get('printed_area_raw')} | "
            f"{room.get('area_m2')} m2 | "
            f"{room['confidence']}"
        )

    print("\nDOORS:")
    print(json.dumps(data.get("doors"), indent=2, ensure_ascii=False))

    print("\nWINDOWS:")
    print(json.dumps(data.get("windows"), indent=2, ensure_ascii=False))