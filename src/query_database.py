
import sqlite3
from pathlib import Path

DB_PATH = Path("data/processed/floorplans.db")


def connect():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run build_database.py first."
        )

    return sqlite3.connect(DB_PATH)


def run_query(sql, params=()):
    connection = connect()
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def total_plans():
    return run_query(
        "SELECT COUNT(*) AS total FROM floorplans"
    )


def total_spaces():
    return run_query(
        "SELECT COUNT(*) AS total FROM spaces"
    )


def largest_space():
    return run_query("""
        SELECT plan_id, label_original, area_m2
        FROM spaces
        WHERE area_m2 IS NOT NULL
        ORDER BY area_m2 DESC
        LIMIT 1
    """)


def average_spaces_per_plan():
    return run_query("""
        SELECT ROUND(AVG(space_count), 2) AS average_spaces
        FROM (
            SELECT plan_id, COUNT(*) AS space_count
            FROM spaces
            GROUP BY plan_id
        )
    """)


def plans_with_space_above(area):
    return run_query("""
        SELECT DISTINCT plan_id, label_original, area_m2
        FROM spaces
        WHERE area_m2 > ?
        ORDER BY area_m2 DESC
    """, (area,))


def search_space(label):
    return run_query("""
        SELECT plan_id, label_original, label_english, area_m2
        FROM spaces
        WHERE LOWER(label_original) LIKE ?
           OR LOWER(COALESCE(label_english, '')) LIKE ?
        ORDER BY plan_id
    """, (f"%{label.lower()}%", f"%{label.lower()}%"))


def main():

    print("\nFLOOR PLAN DATABASE QUERY INTERFACE")
    print("-----------------------------------")

    questions = {
        "1": ("Total floor plans", total_plans),
        "2": ("Total extracted spaces", total_spaces),
        "3": ("Largest extracted space", largest_space),
        "4": ("Average extracted spaces per plan", average_spaces_per_plan),
    }

    while True:

        print("\n1. Total floor plans")
        print("2. Total extracted spaces")
        print("3. Largest extracted space")
        print("4. Average extracted spaces per plan")
        print("5. Find spaces above a given area")
        print("6. Search for a room label")
        print("0. Exit")

        choice = input("\nSelect an option: ").strip()

        if choice == "0":
            break

        elif choice in questions:
            title, function = questions[choice]
            print(f"\n{title}")
            print(function())

        elif choice == "5":
            try:
                area = float(input("Enter area in m2: "))
                print(plans_with_space_above(area))
            except ValueError:
                print("Enter a valid number.")

        elif choice == "6":
            label = input("Enter room label: ").strip()

            if label:
                print(search_space(label))
            else:
                print("Enter a room label.")

        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
