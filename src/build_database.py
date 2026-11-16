
import json
import sqlite3

from pathlib import Path


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

OUTPUT_DIR = Path("outputs/extracted")
DB_PATH = Path("data/processed/floorplans.db")


# --------------------------------------------------
# DATABASE CREATION
# --------------------------------------------------

def create_database():

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(DB_PATH)

    connection.execute("PRAGMA foreign_keys = ON")

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS floorplans (
            plan_id INTEGER PRIMARY KEY,
            language TEXT,
            source_image TEXT,
            source_mask TEXT,
            doors_json TEXT,
            windows_json TEXT,
            notes_json TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spaces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id INTEGER NOT NULL,
            label_original TEXT NOT NULL,
            label_english TEXT,
            printed_area_raw TEXT,
            area_m2 REAL,
            confidence TEXT,
            evidence TEXT,
            FOREIGN KEY (plan_id)
                REFERENCES floorplans(plan_id)
                ON DELETE CASCADE
        )
    """)

    connection.commit()

    return connection


# --------------------------------------------------
# IMPORT EXTRACTIONS
# --------------------------------------------------

def import_extractions(connection):

    cursor = connection.cursor()

    files = sorted(
        OUTPUT_DIR.glob("plan_*.json")
    )

    imported = 0
    total_spaces = 0
    unreadable_labels = 0

    for path in files:

        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        plan_id = data["plan_id"]

        # Replace existing plan records safely.
        # Delete child records before deleting the parent.
        cursor.execute(
            "DELETE FROM spaces WHERE plan_id = ?",
            (plan_id,)
        )

        cursor.execute(
            "DELETE FROM floorplans WHERE plan_id = ?",
            (plan_id,)
        )

        cursor.execute("""
            INSERT INTO floorplans (
                plan_id,
                language,
                source_image,
                source_mask,
                doors_json,
                windows_json,
                notes_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            plan_id,
            data.get("language"),
            f"data/raw/dataset/X/{plan_id}.png",
            f"data/raw/dataset/Y/{plan_id} (*.png)",
            json.dumps(
                data.get("doors", {}),
                ensure_ascii=False
            ),
            json.dumps(
                data.get("windows", {}),
                ensure_ascii=False
            ),
            json.dumps(
                data.get("notes", []),
                ensure_ascii=False
            )
        ))

        for space in data.get("rooms_and_spaces", []):

            original_label = space.get("label_original")

            if not original_label:
                original_label = "[unreadable label]"
                unreadable_labels += 1

            cursor.execute("""
                INSERT INTO spaces (
                    plan_id,
                    label_original,
                    label_english,
                    printed_area_raw,
                    area_m2,
                    confidence,
                    evidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                plan_id,
                original_label,
                space.get("label_english"),
                space.get("printed_area_raw"),
                space.get("area_m2"),
                space.get("confidence"),
                space.get("evidence")
            ))

            total_spaces += 1

        imported += 1

    connection.commit()

    print("\nDATABASE IMPORT SUMMARY")
    print("-----------------------")
    print(f"JSON files imported: {imported}")
    print(f"Total extracted spaces: {total_spaces}")
    print(f"Unreadable labels preserved: {unreadable_labels}")

    cursor.execute(
        "SELECT COUNT(*) FROM floorplans"
    )
    print(
        f"Plans in database: {cursor.fetchone()[0]}"
    )

    cursor.execute(
        "SELECT COUNT(*) FROM spaces"
    )
    print(
        f"Spaces in database: {cursor.fetchone()[0]}"
    )

    connection.close()


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    db = create_database()

    import_extractions(db)

    print(f"\nDatabase saved: {DB_PATH}")
