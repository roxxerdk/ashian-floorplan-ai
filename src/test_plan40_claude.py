import os, json, base64
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def image_block(path):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": data
        }
    }

prompt = """
Analyze this architectural floor plan and its segmentation mask.

The FIRST image is the original architectural drawing.
The SECOND image is its segmentation mask.

Extract:
1. Every readable room and space label.
2. Original French label and English interpretation.
3. Printed area in square metres, preserving the original value.
4. Doors and windows only when visually identifiable.
5. Important spatial observations.
6. Uncertain or unreadable information.

Associate measurements only when supported by the drawing.
Do not infer measurements from pixel areas.
Do not assume mask colours represent particular semantic classes.
Do not invent counts or measurements.

Return valid JSON only:

{
  "plan_id": 40,
  "language": "French",
  "rooms_and_spaces": [
    {
      "label_original": "",
      "label_english": "",
      "printed_area_raw": null,
      "area_m2": null,
      "confidence": "high/medium/low",
      "evidence": ""
    }
  ],
  "doors": {"observations": [], "count": null},
  "windows": {"observations": [], "count": null},
  "uncertain_text": [],
  "notes": []
}
"""

response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=3500,
    messages=[{
        "role": "user",
        "content": [
            image_block("data/raw/dataset/X/40.png"),
            image_block("data/raw/dataset/Y/40 (2).png"),
            {"type": "text", "text": prompt}
        ]
    }]
)

result = response.content[0].text
output = Path("outputs/plan40_claude_vision.json")
output.parent.mkdir(parents=True, exist_ok=True)

try:
    parsed = json.loads(result)
    output.write_text(
        json.dumps(parsed, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )
    print(json.dumps(parsed, indent=4, ensure_ascii=False))
except json.JSONDecodeError:
    output.write_text(result, encoding="utf-8")
    print(result)

print("\nInput tokens:", response.usage.input_tokens)
print("Output tokens:", response.usage.output_tokens)
print("Saved:", output)
