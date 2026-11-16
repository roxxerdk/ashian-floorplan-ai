
import json
import re
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "floorplans.db"
QUESTION_PATH = ROOT / "tests" / "fixtures" / "evaluation_questions.json"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"


# =========================================================
# DATABASE HELPERS
# =========================================================

def query(connection, sql, parameters=()):
    return [
        dict(row)
        for row in connection.execute(sql, parameters).fetchall()
    ]


def load_questions():
    with QUESTION_PATH.open(encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# DYNAMIC REFERENCE ANSWERS
# =========================================================

def build_reference(connection, item):
    question = item["question"].lower()
    category = item.get("category", "")

    if category == "refusal":
        return None

    # Q1: Total floor plans
    if "how many floor plans" in question:
        return {
            "tool": "get_dataset_statistics",
            "data": query(
                connection,
                "SELECT COUNT(*) AS total_plans FROM floorplans"
            ),
            "fields": ["total_plans"]
        }

    # Q2: Total extracted spaces
    if "how many spaces have been extracted" in question:
        return {
            "tool": "get_dataset_statistics",
            "data": query(
                connection,
                "SELECT COUNT(*) AS total_spaces FROM spaces"
            ),
            "fields": ["total_spaces"]
        }

    # Q3: Largest extracted space
    if "largest extracted space" in question:
        return {
            "tool": "find_largest_space",
            "data": query(connection, """
                SELECT plan_id, label_original, area_m2
                FROM spaces
                WHERE area_m2 IS NOT NULL
                  AND label_original != '[unreadable label]'
                ORDER BY area_m2 DESC
                LIMIT 1
            """),
            "fields": ["plan_id", "label_original", "area_m2"]
        }

    # Q4: Average spaces per plan
    if "average number of extracted spaces" in question:
        return {
            "tool": "average_spaces",
            "data": query(connection, """
                SELECT ROUND(AVG(space_count), 2) AS average_spaces
                FROM (
                    SELECT plan_id, COUNT(*) AS space_count
                    FROM spaces
                    GROUP BY plan_id
                )
            """),
            "fields": ["average_spaces"]
        }

    # Q5: Spaces above an area threshold
    if "area greater than" in question:
        match = re.search(r"(\d+(?:\.\d+)?)", question)

        if match:
            threshold = float(match.group(1))

            return {
                "tool": "spaces_above_area",
                "data": query(connection, """
                    SELECT plan_id, label_original, area_m2
                    FROM spaces
                    WHERE area_m2 IS NOT NULL
                      AND area_m2 > ?
                    ORDER BY area_m2 DESC
                """, (threshold,)),
                "fields": ["plan_id", "label_original", "area_m2"]
            }

    # Q6: Spaces below an area threshold
    if "area smaller than" in question:
        match = re.search(r"(\d+(?:\.\d+)?)", question)

        if match:
            threshold = float(match.group(1))

            return {
                "tool": "spaces_below_area",
                "data": query(connection, """
                    SELECT plan_id, label_original, area_m2
                    FROM spaces
                    WHERE area_m2 IS NOT NULL
                      AND area_m2 < ?
                    ORDER BY area_m2 ASC
                """, (threshold,)),
                "fields": ["plan_id", "label_original", "area_m2"]
            }

    # Q7-Q9: Label searches
    label_match = re.search(r"labelled\s+(.+?)[.?]?$", question)

    if label_match:
        label = label_match.group(1).strip()
        pattern = f"%{label}%"

        return {
            "tool": "search_spaces",
            "data": query(connection, """
                SELECT
                    plan_id,
                    label_original,
                    label_english,
                    area_m2,
                    confidence
                FROM spaces
                WHERE LOWER(label_original) LIKE ?
                   OR LOWER(COALESCE(label_english, '')) LIKE ?
                ORDER BY plan_id
            """, (pattern, pattern)),
            "fields": [
                "plan_id",
                "label_original",
                "label_english",
                "area_m2",
                "confidence"
            ]
        }

    # Q10: Plans containing a BAINS space

    # Floor plans containing a BAINS space.
        # Q10: Plans containing a BAINS space
    if "contain a bains space" in question.lower():
        return {
            "tool": "search_spaces",
            "data": query(connection, """
                SELECT DISTINCT plan_id
                FROM spaces
                WHERE UPPER(label_original) LIKE '%BAINS%'
                ORDER BY plan_id
            """, ()),
            "fields": ["plan_id"]
        }


    # Q11: Spaces in a particular plan
    if "spaces were extracted from floor plan" in question:
        match = re.search(r"floor plan\s+(\d+)", question)

        if match:
            plan_id = int(match.group(1))

            return {
                "tool": "spaces_in_plan",
                "data": query(connection, """
                    SELECT
                        label_original,
                        label_english,
                        printed_area_raw,
                        area_m2,
                        confidence
                    FROM spaces
                    WHERE plan_id = ?
                    ORDER BY label_original
                """, (plan_id,)),
                "fields": [
                    "label_original",
                    "label_english",
                    "printed_area_raw",
                    "area_m2",
                    "confidence"
                ]
            }

    # Q12: Largest measured bedroom
    if "largest measured bedroom" in question:
        return {
            "tool": "find_largest_bedroom",
            "data": query(connection, """
                SELECT
                    plan_id,
                    label_original,
                    area_m2,
                    confidence
                FROM spaces
                WHERE area_m2 IS NOT NULL
                  AND LOWER(TRIM(label_original)) LIKE 'chambre%'
                  AND label_original != '[unreadable label]'
                ORDER BY area_m2 DESC
                LIMIT 1
            """),
            "fields": [
                "plan_id",
                "label_original",
                "area_m2",
                "confidence"
            ]
        }

    # Q13: Plans with more than N spaces
    if "more than" in question and "extracted spaces" in question:
        match = re.search(r"(\d+)", question)

        if match:
            threshold = int(match.group(1))

            return {
                "tool": "plans_above_space_count",
                "data": query(connection, """
                    SELECT
                        plan_id,
                        COUNT(*) AS space_count
                    FROM spaces
                    GROUP BY plan_id
                    HAVING COUNT(*) > ?
                    ORDER BY plan_id
                """, (threshold,)),
                "fields": ["plan_id", "space_count"]
            }

    # Q14: Spaces without recorded areas
    if "no recorded area" in question:
        return {
            "tool": "count_spaces_without_area",
            "data": query(connection, """
                SELECT COUNT(*) AS total
                FROM spaces
                WHERE area_m2 IS NULL
            """),
            "fields": ["total"]
        }

    # Q15: Plan with greatest sum of recorded areas
    if "greatest sum of recorded space areas" in question:
        return {
            "tool": "plan_with_greatest_recorded_area",
            "data": query(connection, """
                SELECT
                    plan_id,
                    ROUND(SUM(area_m2), 2) AS total_area
                FROM spaces
                WHERE area_m2 IS NOT NULL
                GROUP BY plan_id
                ORDER BY total_area DESC
                LIMIT 1
            """),
            "fields": ["plan_id", "total_area"]
        }

    return None


# =========================================================
# RESULT NORMALISATION AND COMPARISON
# =========================================================

def normalise_value(value):
    if isinstance(value, float):
        return round(value, 6)

    return value


def project_rows(rows, fields):
    projected = []

    for row in rows:
        projected.append({
            field: normalise_value(row.get(field))
            for field in fields
        })

    # Sorting makes comparison independent of row ordering.
    return sorted(
        projected,
        key=lambda row: json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False
        )
    )


def compare_evidence(reference, evidence):
    if reference is None:
        return {
            "passed": False,
            "reason": "No reference result available."
        }

    matching_calls = [
        entry for entry in evidence
        if entry.get("tool") == reference["tool"]
    ]

    if not matching_calls:
        return {
            "passed": False,
            "reason": (
                f"Expected tool {reference['tool']} "
                "was not executed."
            )
        }

    # Use the latest successful matching tool execution.
    call = next(
        (
            entry for entry in reversed(matching_calls)
            if not entry.get("error")
        ),
        None
    )

    if call is None:
        return {
            "passed": False,
            "reason": "Matching tool execution returned an error."
        }

    actual_data = call.get("data") or []
    expected_data = reference["data"]
    fields = reference["fields"]

    # Q10 asks for distinct plan IDs, not every matching space.
    if reference["tool"] == "search_spaces" and fields == ["plan_id"]:
        actual_rows = sorted({
            row.get("plan_id")
            for row in actual_data
        })

        expected_rows = sorted({
            row.get("plan_id")
            for row in expected_data
        })

        passed = actual_rows == expected_rows

    else:
        actual_rows = project_rows(actual_data, fields)
        expected_rows = project_rows(expected_data, fields)

        passed = actual_rows == expected_rows

    return {
        "passed": passed,
        "reason": (
            "Database evidence matches the SQLite reference."
            if passed
            else "Database evidence differs from the SQLite reference."
        ),
        "expected_record_count": len(expected_data),
        "actual_record_count": len(actual_data)
    }


# =========================================================
# REFUSAL EVALUATION
# =========================================================

REFUSAL_PATTERNS = [
    r"\bcannot determine\b",
    r"\bcan't determine\b",
    r"\bcannot provide\b",
    r"\bcan't provide\b",
    r"\bcan't answer\b",
    r"\bcannot answer\b",
    r"\bcan't give\b",
    r"\bcannot give\b",
    r"\bunable to determine\b",
    r"\bnot available\b",
    r"\bnot verified\b",
    r"\bunverified\b",
    r"\binsufficient data\b",
    r"\binsufficient information\b",
    r"\bnot captured\b",
    r"\bnot recorded\b",
    r"\bnot supported by the data\b",
    r"\bno data\b",
    r"\bnot possible to determine\b",
    r"\bnot independently verified\b",
    r"\bno verified\b",
    r"\bdo not contain\b",
    r"\bdoes not contain\b",
    r"\bnot enough information\b",
    r"\bwould have to invent\b",
    r"\bno wall data\b",
    r"\bno building or usable-area field\b"
    r"\bI can only answer questions related to\b"
]



def check_refusal(answer):
    text = (answer or "").lower()

    matched = [
        pattern
        for pattern in REFUSAL_PATTERNS
        if re.search(pattern, text)
    ]

    return {
        "passed": bool(matched),
        "matched_refusal_indicators": matched,
        "reason": (
            "A refusal or data-limitation statement was detected."
            if matched
            else "No recognised refusal statement was detected."
        )
    }


# =========================================================
# EVALUATION RUNNER
# =========================================================

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    if not QUESTION_PATH.exists():
        raise FileNotFoundError(
            f"Question bank not found: {QUESTION_PATH}"
        )

    # Import only when the evaluation is run.
    from src.app import ask_claude

    questions = load_questions()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n" + "=" * 65)
    print("ASK YOUR FLOOR PLAN - EVALUATION")
    print("=" * 65)
    print(f"Questions: {len(questions)}")
    print("Running actual assistant queries. API usage may apply.")

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        for item in questions:
            question_id = item.get("id")
            question = item["question"]
            category = item.get("category", "")

            print("\n" + "-" * 65)
            print(f"Q{question_id}: {question}")

            result = {
                "id": question_id,
                "question": question,
                "category": category,
                "answer": None,
                "executed_tools": [],
                "reference": None,
                "evaluation": None
            }

            try:
                reference = build_reference(connection, item)
                result["reference"] = reference

                # Ask the actual application.
                answer, evidence = ask_claude(question)

                result["answer"] = answer
                result["executed_tools"] = evidence

                if category == "refusal":
                    evaluation = check_refusal(answer)
                    evaluation["evaluation_type"] = "refusal"

                elif reference is None:
                    evaluation = {
                        "passed": False,
                        "evaluation_type": "reference",
                        "reason": "Could not construct a SQLite reference."
                    }

                else:
                    evaluation = compare_evidence(
                        reference,
                        evidence
                    )
                    evaluation["evaluation_type"] = "database_evidence"

                result["evaluation"] = evaluation

                status = (
                    "PASS"
                    if evaluation["passed"]
                    else "FAIL"
                )

                print(f"RESULT: {status}")
                print("ANSWER:", answer)

                print(
                    "TOOLS:",
                    [
                        entry.get("tool")
                        for entry in evidence
                    ]
                )

                print("DETAIL:", evaluation["reason"])

            except Exception as error:
                result["evaluation"] = {
                    "passed": False,
                    "reason": str(error),
                    "evaluation_type": "execution_error"
                }

                print("RESULT: ERROR")
                print("ERROR:", str(error))

            results.append(result)

    # =====================================================
    # METRICS
    # =====================================================

    answerable_results = [
        result for result in results
        if result["category"] != "refusal"
    ]

    refusal_results = [
        result for result in results
        if result["category"] == "refusal"
    ]

    passed_answerable = sum(
        1 for result in answerable_results
        if result["evaluation"]
        and result["evaluation"]["passed"]
    )

    passed_refusals = sum(
        1 for result in refusal_results
        if result["evaluation"]
        and result["evaluation"]["passed"]
    )

    passed_total = passed_answerable + passed_refusals

    answerable_accuracy = (
        passed_answerable / len(answerable_results) * 100
        if answerable_results else 0
    )

    refusal_accuracy = (
        passed_refusals / len(refusal_results) * 100
        if refusal_results else 0
    )

    overall_score = (
        passed_total / len(results) * 100
        if results else 0
    )

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "database": str(DB_PATH),
        "question_bank": str(QUESTION_PATH),

        "methodology": {
            "answerable_questions": (
                "Compared executed database tool evidence "
                "against dynamically generated SQLite references."
            ),
            "refusal_questions": (
                "Checked for recognised refusal and data-limitation "
                "language using a phrase-based heuristic."
            ),
            "limitation": (
                "Evidence comparison does not independently verify "
                "every natural-language claim. Refusal classification "
                "requires manual review before reporting final results."
            )
        },

        "summary": {
            "total_questions": len(results),
            "answerable_questions": len(answerable_results),
            "refusal_questions": len(refusal_results),
            "answerable_passed": passed_answerable,
            "refusal_passed": passed_refusals,
            "total_passed": passed_total,
            "answerable_evidence_accuracy_percent": round(
                answerable_accuracy, 2
            ),
            "refusal_detection_percent": round(
                refusal_accuracy, 2
            ),
            "overall_score_percent": round(
                overall_score, 2
            )
        },

        "results": results
    }

    output_file = OUTPUT_DIR / "evaluation_report.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # =====================================================
    # FINAL CONSOLE REPORT
    # =====================================================

    print("\n" + "=" * 65)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 65)

    print(f"Total questions: {len(results)}")
    print(
        f"Answerable questions passed: "
        f"{passed_answerable}/{len(answerable_results)}"
    )
    print(
        f"Refusal questions passed: "
        f"{passed_refusals}/{len(refusal_results)}"
    )

    print(
        f"Answerable evidence accuracy: "
        f"{answerable_accuracy:.2f}%"
    )

    print(
        f"Refusal detection: "
        f"{refusal_accuracy:.2f}%"
    )

    print(f"Overall score: {overall_score:.2f}%")

    print("\nReport saved to:")
    print(output_file)

    print(
        "\nNote: Refusal results use phrase-based detection "
        "and should be manually reviewed."
    )


if __name__ == "__main__":
    main()
