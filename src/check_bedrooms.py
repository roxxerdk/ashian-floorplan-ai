import sqlite3

connection = sqlite3.connect("data/processed/floorplans.db")

query = """
SELECT
    plan_id,
    label_original,
    area_m2,
    confidence
FROM spaces
WHERE area_m2 IS NOT NULL
AND LOWER(label_original) LIKE '%chambre%'
ORDER BY area_m2 DESC
LIMIT 10
"""

results = connection.execute(query).fetchall()

print("\nTOP 10 LARGEST MEASURED BEDROOM-LABEL RECORDS")
print("-" * 55)

for row in results:
    print(row)

connection.close()