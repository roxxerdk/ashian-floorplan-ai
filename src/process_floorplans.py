
import os
import json
import base64
import argparse

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from anthropic import Anthropic
from pydantic import BaseModel, Field
from PIL import Image


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

MODEL = "claude-sonnet-5-5"

DATA_DIR = Path("data/raw/dataset")
OUTPUT_DIR = Path("outputs/extracted")
LOG_PATH = OUTPUT_DIR / "processing_log.json"


# --------------------------------------------------
# PYDANTIC DATA MODELS
# --------------------------------------------------

class Room(BaseModel):
    label_original: Optional[str] = None
    label_english: Optional[str] = None
    printed_area_raw: Optional[str] = None
    area_m2: Optional[float] = None
    confidence: str = "low"
    evidence: str = ""


class FloorPlanResult(BaseModel):
    plan_id: int
    language: str = "unknown"
    rooms_and_spaces: list[Room] = Field(default_factory=list)
    doors: dict = Field(default_factory=dict)
    windows: dict = Field(default_factory=dict)
    uncertain_text: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# --------------------------------------------------
# JSON RESPONSE EXTRACTION
# --------------------------------------------------

def extract_json_response(response):

    text = "\n".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )

    if not text:
        raise ValueError("Claude returned no text response")

    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object found in Claude response")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):

        char = text[index]

        if in_string:

            if escaped:
                escaped = False

            elif char == "\\":
                escaped = True

            elif char == '"':
                in_string = False

        else:

            if char == '"':
                in_string = True

            elif char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    return json.loads(text[start:index + 1])

    raise ValueError("Claude returned incomplete JSON")


# --------------------------------------------------
# IMAGE UTILITIES
# --------------------------------------------------

def encode_image(path):

    with Image.open(path) as img:

        image_format = img.format

    media_types = {
        "PNG": "image/png",
        "JPEG": "image/jpeg"
    }

    if image_format not in media_types:
        raise ValueError(
            f"Unsupported image format: {image_format} ({path})"
        )

    image_data = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    return image_data, media_types[image_format]


# --------------------------------------------------
# MASK MATCHING
# --------------------------------------------------

def find_mask(plan_id):

    matches = list(
        (DATA_DIR / "Y").glob(f"{plan_id} (*).png")
    )

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Plan {plan_id}: expected exactly one mask, "
            f"found {len(matches)}"
        )

    return matches[0]


# --------------------------------------------------
# PROCESS ONE FLOOR PLAN
# --------------------------------------------------

def process_plan(client, plan_id):

    image_path = DATA_DIR / "X" / f"{plan_id}.png"
    mask_path = find_mask(plan_id)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Original image not found: {image_path}"
        )

    # Encode both images using their actual formats.
    original_data, original_media_type = encode_image(image_path)
    mask_data, mask_media_type = encode_image(mask_path)

    prompt = f"""
Analyze this architectural floor plan.

First image: original architectural drawing.
Second image: segmentation mask.

Extract readable room and space labels, original text,
English interpretations, and printed areas in square metres.

Requirements:

1. Preserve original labels exactly where possible.
2. If a label cannot be read, use null. Do not invent a label.
3. Preserve original decimal-comma measurements.
4. Convert readable printed measurements to numeric area_m2.
5. Associate an area only when supported by the drawing.
6. Do not invent missing measurements.
7. Do not invent door or window counts.
8. Do not assume mask colours have semantic meanings.
9. Distinguish rooms from garages, terraces, porches,
   cupboards and other spaces.
10. Preserve combined spaces as combined spaces.
11. Do not create grouped or inferred area measurements.
12. Mark uncertain information explicitly.
13. Return valid JSON only.

The plan ID is {plan_id}.

Required JSON structure:

{{
  "plan_id": {plan_id},
  "language": "French",
  "rooms_and_spaces": [
    {{
      "label_original": null,
      "label_english": null,
      "printed_area_raw": null,
      "area_m2": null,
      "confidence": "low",
      "evidence": ""
    }}
  ],
  "doors": {{
    "observations": [],
    "count": null
  }},
  "windows": {{
    "observations": [],
    "count": null
  }},
  "uncertain_text": [],
  "notes": []
}}
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=12000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": original_media_type,
                            "data": original_data
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mask_media_type,
                            "data": mask_data
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

    print(
        f"Claude stop reason: {response.stop_reason}, "
        f"output tokens: {response.usage.output_tokens}"
    )

    if response.stop_reason == "max_tokens":
        raise ValueError(
            "Claude response reached max_tokens; JSON may be truncated"
        )

    parsed = extract_json_response(response)

    result = FloorPlanResult.model_validate(parsed)

    if result.plan_id != plan_id:
        raise ValueError(
            f"Expected plan {plan_id}, "
            f"received plan {result.plan_id}"
        )

    # Preserve missing labels without inventing names.
    # The JSON keeps null; the database can use a placeholder later.

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"plan_{plan_id}.json"

    output_path.write_text(
        result.model_dump_json(indent=4),
        encoding="utf-8"
    )

    known_areas = [
        room.area_m2
        for room in result.rooms_and_spaces
        if room.area_m2 is not None
    ]

    area_sum = round(sum(known_areas), 2)

    return {
        "plan_id": plan_id,
        "status": "success",
        "rooms_and_spaces": len(result.rooms_and_spaces),
        "unlabelled_spaces": sum(
            room.label_original is None
            for room in result.rooms_and_spaces
        ),
        "known_area_sum_m2": area_sum,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "output_file": str(output_path)
    }


# --------------------------------------------------
# PROCESSING LOG
# --------------------------------------------------

def save_log(log):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOG_PATH.write_text(
        json.dumps(log, indent=4),
        encoding="utf-8"
    )


def load_existing_log():

    if not LOG_PATH.exists():
        return []

    try:
        return json.loads(
            LOG_PATH.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return []


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Claude Vision Floor Plan Processor"
    )

    parser.add_argument(
        "--plans",
        nargs="+",
        type=int,
        default=[1, 16, 40, 67, 85]
    )

    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing from .env"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    client = Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Preserve earlier results and update only requested plan IDs.
    log = load_existing_log()

    log_by_id = {
        item["plan_id"]: item
        for item in log
    }

    for plan_id in args.plans:

        print(f"\nProcessing Plan {plan_id}...")

        try:

            result = process_plan(
                client,
                plan_id
            )

            log_by_id[plan_id] = result

            print(json.dumps(result, indent=4))

        except Exception as error:

            failure = {
                "plan_id": plan_id,
                "status": "failed",
                "error": str(error)
            }

            log_by_id[plan_id] = failure

            print(json.dumps(failure, indent=4))

        # Save after every plan, including failures.
        log = sorted(
            log_by_id.values(),
            key=lambda item: item["plan_id"]
        )

        save_log(log)

    successful = sum(
        item["status"] == "success"
        for item in log
    )

    failed = sum(
        item["status"] == "failed"
        for item in log
    )

    print("\nProcessing completed.")
    print(f"Processing log saved: {LOG_PATH}")
    print(f"Successful plans in log: {successful}")
    print(f"Failed plans in log: {failed}")


if __name__ == "__main__":
    main()
