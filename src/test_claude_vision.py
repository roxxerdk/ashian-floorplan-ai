import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

image_path = Path("data/raw/dataset/X/16.png")
image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

prompt = """
Analyze this architectural floor plan.

Extract the information visible in the drawing.

Requirements:
1. Identify every readable room or space label.
2. Preserve the original French text exactly where possible.
3. Provide an English interpretation separately.
4. Extract printed area measurements and units.
5. Associate an area with a room only when the drawing supports
   that association.
6. Do not calculate or invent missing measurements.
7. Do not invent doors, windows, room counts or dimensions.
8. Mark uncertain text and associations as uncertain.
9. Do not assume segmentation-mask colour meanings.

Return valid JSON with this structure:

{
  "plan_id": 16,
  "language": "",
  "rooms_and_spaces": [
    {
      "label_original": "",
      "label_english": "",
      "printed_area_m2": null,
      "evidence": "",
      "confidence": "high/medium/low"
    }
  ],
  "unassigned_measurements": [],
  "uncertain_text": [],
  "notes": []
}

Return JSON only. Do not include Markdown code fences.
"""

response = client.messages.create(
    model="claude-sonnet-5-5",
    max_tokens=3000,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
)

result = response.content[0].text

output_path = Path("outputs/plan16_claude_vision.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

try:
    parsed = json.loads(result)
    output_path.write_text(
        json.dumps(parsed, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )
    print(json.dumps(parsed, indent=4, ensure_ascii=False))
except json.JSONDecodeError:
    output_path.write_text(result, encoding="utf-8")
    print("Claude returned non-JSON output:")
    print(result)

print("\n--- API USAGE ---")
print("Input tokens:", response.usage.input_tokens)
print("Output tokens:", response.usage.output_tokens)
print("Saved result:", output_path)
