import re
import os
import json
import sqlite3
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

DB_PATH = Path("data/processed/floorplans.db")
MODEL = "claude-sonnet-5-5"

st.set_page_config(
    page_title="Ask Your Floor Plan",
    page_icon="🏠",
    layout="wide"
)


# ==================================================
# DATABASE CONNECTION
# ==================================================

def get_connection():

    if not DB_PATH.exists():
        raise FileNotFoundError("SQLite database not found.")

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def execute_query(sql, params=()):

    connection = get_connection()

    try:
        cursor = connection.execute(sql, params)

        return [
            dict(row)
            for row in cursor.fetchall()
        ]

    finally:
        connection.close()


# ==================================================
# DATABASE QUERY FUNCTIONS
# ==================================================

def get_dataset_statistics():

    sql = """
        SELECT
            (SELECT COUNT(*) FROM floorplans) AS total_plans,
            (SELECT COUNT(*) FROM spaces) AS total_spaces,
            (SELECT COUNT(*)
             FROM spaces
             WHERE area_m2 IS NOT NULL) AS measured_spaces
    """

    return {
        "sql": sql,
        "parameters": [],
        "data": execute_query(sql)
    }


def find_largest_space():

    sql = """
        SELECT
            plan_id,
            label_original,
            label_english,
            area_m2,
            confidence
        FROM spaces
        WHERE area_m2 IS NOT NULL
          AND label_original != '[unreadable label]'
        ORDER BY area_m2 DESC
        LIMIT 1
    """

    return {
        "sql": sql,
        "parameters": [],
        "data": execute_query(sql)
    }


def find_largest_bedroom():

    # Use original French labels only.
    # Do not use English descriptions because they can
    # contain misleading phrases such as "children's bedrooms".

    sql = """
        SELECT
            plan_id,
            label_original,
            area_m2,
            confidence
        FROM spaces
        WHERE area_m2 IS NOT NULL
          AND label_original != '[unreadable label]'
          AND LOWER(TRIM(label_original)) LIKE 'chambre%'
        ORDER BY area_m2 DESC
        LIMIT 1
    """

    return {
        "sql": sql,
        "parameters": [],
        "data": execute_query(sql)
    }


def average_spaces():

    sql = """
        SELECT
            ROUND(AVG(space_count), 2) AS average_spaces
        FROM (
            SELECT
                plan_id,
                COUNT(*) AS space_count
            FROM spaces
            GROUP BY plan_id
        )
    """

    return {
        "sql": sql,
        "parameters": [],
        "data": execute_query(sql)
    }


def search_spaces(label):

    sql = """
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
        LIMIT 200
    """

    pattern = f"%{label.lower()}%"

    return {
        "sql": sql,
        "parameters": [pattern, pattern],
        "data": execute_query(
            sql,
            (pattern, pattern)
        )
    }


def spaces_above_area(area):

    sql = """
        SELECT
            plan_id,
            label_original,
            label_english,
            area_m2,
            confidence
        FROM spaces
        WHERE area_m2 > ?
        ORDER BY area_m2 DESC
        LIMIT 200
    """

    return {
        "sql": sql,
        "parameters": [area],
        "data": execute_query(sql, (area,))
    }


def spaces_in_plan(plan_id):

    sql = """
        SELECT
            label_original,
            label_english,
            printed_area_raw,
            area_m2,
            confidence
        FROM spaces
        WHERE plan_id = ?
        ORDER BY label_original
    """

    return {
        "sql": sql,
        "parameters": [plan_id],
        "data": execute_query(sql, (plan_id,))
    }


# ==================================================
# CLAUDE TOOL DEFINITIONS
# ==================================================

TOOLS = [

    {
        "name": "get_dataset_statistics",
        "description": (
            "Retrieve the total floor plans, "
            "extracted spaces and measured spaces."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },

    {
        "name": "find_largest_space",
        "description": (
            "Find the largest extracted space with "
            "a recorded area, regardless of space type."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },

    {
        "name": "find_largest_bedroom",
        "description": (
            "Find the largest measured bedroom. "
            "Uses original French labels beginning "
            "with CHAMBRE. Does not use English descriptions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },

    {
        "name": "average_spaces",
        "description": (
            "Calculate the average number of extracted "
            "space records per floor plan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },

    {
        "name": "search_spaces",
        "description": (
            "Search extracted spaces using their "
            "original or English labels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string"
                }
            },
            "required": ["label"]
        }
    },

    {
        "name": "spaces_above_area",
        "description": (
            "Find extracted spaces larger than "
            "a specified area in square metres."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "area": {
                    "type": "number"
                }
            },
            "required": ["area"]
        }
    },

    {
        "name": "spaces_in_plan",
        "description": (
            "Retrieve all extracted spaces belonging "
            "to a specified floor plan ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "plan_id": {
                    "type": "integer"
                }
            },
            "required": ["plan_id"]
        }
    }
]


# ==================================================
# TOOL EXECUTION
# ==================================================

TOOL_FUNCTIONS = {

    "get_dataset_statistics":
        lambda args: get_dataset_statistics(),

    "find_largest_space":
        lambda args: find_largest_space(),

    "find_largest_bedroom":
        lambda args: find_largest_bedroom(),

    "average_spaces":
        lambda args: average_spaces(),

    "search_spaces":
        lambda args: search_spaces(args["label"]),

    "spaces_above_area":
        lambda args: spaces_above_area(args["area"]),

    "spaces_in_plan":
        lambda args: spaces_in_plan(args["plan_id"])
}


# ==================================================
# DETERMINISTIC QUESTION HANDLING
# ==================================================


def handle_deterministic_questions(question):

    normalized = question.lower()

    # ----------------------------------------------
    # Bedrooms above a specified area
    # ----------------------------------------------

    threshold_match = re.search(
        r"(?:above|over|greater than)\s*(\d+(?:\.\d+)?)",
        normalized
    )

    bedroom_question = (
        "bedroom" in normalized
        or "bedrooms" in normalized
    )

    if threshold_match and bedroom_question:

        threshold = float(threshold_match.group(1))

        sql = """
            SELECT
                plan_id,
                label_original,
                area_m2,
                confidence
            FROM spaces
            WHERE area_m2 IS NOT NULL
              AND LOWER(TRIM(label_original)) LIKE 'chambre%'
              AND area_m2 > ?
            ORDER BY area_m2 DESC
            LIMIT 200
        """

        data = execute_query(sql, (threshold,))

        result = {
            "sql": sql,
            "parameters": [threshold],
            "data": data
        }

        if not data:
            answer = (
                f"No measured bedroom records were found "
                f"above {threshold} m²."
            )

        else:
            answer = (
                f"Found {len(data)} measured bedroom records "
                f"above {threshold} m².\n\n"
            )

            for bedroom in data:
                answer += (
                    f"- Plan {bedroom['plan_id']}: "
                    f"{bedroom['label_original']} — "
                    f"{bedroom['area_m2']:.2f} m² "
                    f"(confidence: {bedroom['confidence']})\n"
                )

        return answer, [{
            "tool": "spaces_above_area",
            **result
        }]

    # ----------------------------------------------
    # Largest bedroom
    # ----------------------------------------------

    bedroom_phrases = [
        "largest bedroom",
        "biggest bedroom",
        "largest bed room",
        "biggest bed room"
    ]

    if any(phrase in normalized for phrase in bedroom_phrases):

        result = find_largest_bedroom()

        if not result["data"]:
            return (
                "No measured bedroom records were found.",
                [{
                    "tool": "find_largest_bedroom",
                    **result
                }]
            )

        bedroom = result["data"][0]

        answer = (
            f"Plan {bedroom['plan_id']} has the largest "
            f"measured bedroom. "
            f"{bedroom['label_original']} measures "
            f"{bedroom['area_m2']:.2f} m². "
            f"Recorded extraction confidence: "
            f"{bedroom['confidence']}."
        )

        return answer, [{
            "tool": "find_largest_bedroom",
            **result
        }]

    return None


# ==================================================
# NATURAL LANGUAGE QUESTION ANSWERING
# ==================================================

def ask_claude(question):

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY missing from .env"
        )

    # Handle the largest-bedroom query deterministically.
    deterministic_result = handle_deterministic_questions(
        question
    )

    if deterministic_result is not None:
        return deterministic_result

    client = Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    executed_queries = []

    system_prompt = """
You are an assistant for architectural floor-plan data.

Your job is to answer questions using the supplied
SQLite database tools.

STRICT RULES:

1. Use database tools for factual questions.
2. Use only actual returned database records.
3. Never invent room labels, areas or plan IDs.
4. Never generate or execute SQL yourself.
5. Never manually reinterpret or override a tool result.
6. Never claim to have performed an additional search
   unless another tool was actually called.
7. If the tools cannot answer a question, explain why.
8. Door and window counts are currently unverified.
9. Missing measurements must not be treated as zero.
10. Extracted spaces are not necessarily verified rooms.
11. Mention uncertainty when confidence is low.
12. Do not use your general knowledge to fill database gaps.

Provide concise answers.
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=system_prompt,
        tools=TOOLS,
        messages=messages
    )

    for attempt in range(12):

        tool_blocks = [
            block
            for block in response.content
            if block.type == "tool_use"
        ]

        if not tool_blocks:

            answer = "\n".join(
                block.text
                for block in response.content
                if block.type == "text"
            )

            if not answer:
                answer = (
                    "Claude returned no final text answer."
                )

            return answer, executed_queries

        messages.append({
            "role": "assistant",
            "content": response.content
        })

        tool_results = []

        for tool_call in tool_blocks:

            function = TOOL_FUNCTIONS.get(
                tool_call.name
            )

            if function is None:

                result = {
                    "error": "Unsupported database tool"
                }

            else:

                try:
                    result = function(tool_call.input)

                except Exception as error:
                    result = {
                        "error": str(error)
                    }

            executed_queries.append({
                "tool": tool_call.name,
                "sql": result.get("sql"),
                "parameters": result.get("parameters", []),
                "data": result.get("data", []),
                "error": result.get("error")
            })

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": json.dumps(
                    result,
                    ensure_ascii=False
                )
            })

        messages.append({
            "role": "user",
            "content": tool_results
        })

        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )

    return (
        "The database lookup exceeded the allowed tool calls.",
        executed_queries
    )


# ==================================================
# STREAMLIT USER INTERFACE
# ==================================================

st.title("🏠 Ask Your Floor Plan")

st.caption(
    "An AI assistant for exploring architectural floor plans "
    "using vision extraction, SQLite and Claude."
)


if not DB_PATH.exists():

    st.error(
        "Database not found. Run src/build_database.py first."
    )

    st.stop()


# ==================================================
# DASHBOARD METRICS
# ==================================================

statistics = get_dataset_statistics()["data"][0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Floor plans",
    statistics["total_plans"]
)

col2.metric(
    "Extracted spaces",
    statistics["total_spaces"]
)

col3.metric(
    "Spaces with area",
    statistics["measured_spaces"]
)


st.divider()


# ==================================================
# QUESTION INTERFACE
# ==================================================

st.subheader("Ask your floor plan")

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: Which floor plan has the largest bedroom?"
    ),
    height=110
)


if st.button("Go", type="primary"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Analysing your question and querying the database..."
        ):

            try:

                answer, queries = ask_claude(
                    question.strip()
                )

                st.session_state["last_question"] = question
                st.session_state["last_answer"] = answer
                st.session_state["last_queries"] = queries

            except Exception as error:

                st.error(
                    f"Question processing failed: {error}"
                )


# ==================================================
# ANSWER DISPLAY
# ==================================================

if "last_answer" in st.session_state:

    st.subheader("Answer")

    st.write(
        st.session_state["last_answer"]
    )

    st.subheader("Database evidence")

    queries = st.session_state["last_queries"]

    if queries:

        for index, query in enumerate(queries, 1):

            with st.expander(
                f"Query {index}: {query['tool']}"
            ):

                if query["sql"]:

                    st.markdown("**SQL executed**")

                    st.code(
                        query["sql"],
                        language="sql"
                    )

                if query["parameters"]:

                    st.markdown("**Query parameters**")

                    st.json(
                        query["parameters"]
                    )

                if query.get("error"):

                    st.error(
                        query["error"]
                    )

                st.markdown("**Returned database records**")

                st.json(
                    query["data"]
                )

    else:

        st.info(
            "This answer did not require a database lookup."
        )


# ==================================================
# DATA LIMITATIONS
# ==================================================

st.divider()

with st.expander("Data limitations and reliability"):

    st.write("""
    1. Extracted spaces are not necessarily independently
       verified rooms.

    2. Some original labels could not be read.

    3. Missing measurements are not treated as zero.

    4. Door and window counts are not validated.

    5. Physical wall dimensions and room geometry are
       not yet fully extracted.

    6. Answers are based on records stored in SQLite.

    7. Confidence values describe extraction confidence,
       not independent ground-truth verification.
    """)
