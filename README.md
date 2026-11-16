
# Ashian – Ask Your Floor Plan

An AI-powered assistant that enables users to ask natural-language questions about architectural floor plans using visually extracted information, structured data, and database-backed queries.

## 1. Project Overview

Ask Your Floor Plan is an AI-powered floor-plan exploration system developed to make architectural information accessible through natural-language interaction.

The system processes architectural floor-plan images and segmentation masks, extracts labelled spaces and available measurements, stores structured information in an SQLite database, and enables users to query the extracted information through an AI-powered interface.

The application emphasizes evidence-grounded responses, structured database retrieval, and responsible handling of questions when the required information is unavailable.

## 2. Project Objectives

- Process architectural floor-plan images and segmentation masks.
- Inspect and analyse image components.
- Extract labelled spaces and available measurements.
- Store extracted information in structured JSON files.
- Build a centralized SQLite database.
- Answer natural-language questions using database queries.
- Provide answers supported by retrieved database records.
- Prevent unsupported or unverified information from being presented as fact.
- Evaluate answer accuracy and refusal behaviour through a predefined evaluation framework.

## 3. Key Features

### 3.1 Floor Plan Processing

- Architectural floor-plan image inspection.
- Segmentation-mask analysis.
- Image processing using OpenCV.
- Visual analysis and extraction of labelled spaces.
- Structured JSON generation.
- Extraction confidence information.

### 3.2 Structured Database

- SQLite-based storage.
- Floor-plan and extracted-space records.
- Recorded space measurements.
- Extraction confidence information.
- Database-backed search and filtering.

### 3.3 AI-Powered Question Answering

- Natural-language question interface built with Streamlit.
- Database tool calling for information retrieval.
- Deterministic database-backed responses for supported queries.
- Evidence-grounded answers.
- Domain restrictions for unrelated questions.
- Appropriate refusals when information is unavailable or unverified.

### 3.4 Evaluation Framework

- 20 predefined evaluation questions.
- 15 answerable questions.
- 5 unsupported questions designed to test refusal behaviour.
- SQLite-based reference comparisons.
- Evaluation reporting.
- Separate measurement of answerable-question accuracy and refusal detection.

## 4. Dataset and Processing Results

The current processed dataset contains:

| Metric | Result |
|---|---:|
| Floor plans | 121 |
| Extracted spaces | 1,511 |
| Spaces with recorded areas | 1,101 |
| Spaces without recorded areas | 410 |

**Important:** Extracted spaces should not automatically be considered independently verified rooms.

Segmentation-mask colour families were inspected, but their semantic meaning was not assumed without a verified class legend.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Core programming language |
| OpenCV | Image processing and analysis |
| NumPy | Numerical operations |
| SQLite | Structured database storage |
| Streamlit | Interactive application interface |
| Vision-language model API | Visual analysis and natural-language processing |
| JSON | Structured extraction storage |
| Database tool calling | Evidence-based information retrieval |

## 6. System Architecture

The application follows a structured processing and retrieval workflow.

1. **Input:** Architectural floor-plan images and segmentation masks.
2. **Inspection:** Analyse image characteristics and segmentation-mask information.
3. **Extraction:** Identify labelled spaces and available measurements.
4. **Structured Storage:** Save extraction results in JSON format.
5. **Database Construction:** Import structured records into SQLite.
6. **Question Processing:** Interpret natural-language questions.
7. **Information Retrieval:** Execute relevant database tools and retrieve supporting records.
8. **Response Generation:** Generate answers grounded in retrieved information.
9. **Evaluation:** Compare supported answers against database references and test appropriate refusals.

### Architecture Flow

```text
Architectural Floor Plans
          |
          v
 Image and Mask Inspection
          |
          v
 Visual Analysis and Extraction
          |
          v
 Structured JSON Records
          |
          v
    SQLite Database
          |
          v
 Natural Language Question
          |
          v
  AI Question Processing
          |
          v
 Database Tool Execution
          |
          v
 Evidence-Based Response
          |
          v
    Streamlit Interface
```

## 7. Repository Structure

```text
ASHIAN-FLOORPLAN-AI/
│
├── data/
│   ├── raw/
│   │   ├── X/
│   │   └── Y/
│   │
│   └── processed/
│
├── outputs/
│   ├── extracted/
│   ├── overlays/
│   ├── ocr/
│   └── evaluation/
│
├── src/
│   ├── app.py
│   ├── evaluation.py
│   ├── process_floorplans.py
│   ├── build_database.py
│   ├── query_database.py
│   ├── inspect_dataset.py
│   ├── overlay_components.py
│   └── ...
│
├── tests/
│   └── fixtures/
│       └── evaluation_questions.json
│
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## 8. Installation and Setup

### Prerequisites

- Python 3.11
- Git
- pip
- A valid API key for the configured vision-language service

### Step 1: Clone the Repository

```bash
git clone https://github.com/roxxerdk/ashian-floorplan-ai.git
cd ashian-floorplan-ai
```

### Step 2: Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root using `.env.example` as a template.

Add your API key to the appropriate environment variable.

Never commit your actual `.env` file or API credentials.

### Step 5: Prepare the Dataset and Database

The application requires the processed floor-plan extraction data and SQLite database.

The raw dataset and generated database may need to be obtained or generated separately. Follow the processing scripts and dataset instructions before running the application on a fresh installation.

## 9. Running the Application

Start the Streamlit application:

```bash
streamlit run src/app.py
```

Open the local URL displayed in the terminal.

Users can enter natural-language questions about the available floor plans and extracted spaces.

## 10. Running the Evaluation Framework

Run the evaluation from the project root:

```bash
python -m src.evaluation
```

The evaluation framework uses the predefined question bank:

```text
tests/fixtures/evaluation_questions.json
```

The generated evaluation report is saved under:

```text
outputs/evaluation/evaluation_report.json
```

### Current Evaluation Results

| Evaluation Category | Result |
|---|---:|
| Answerable questions passed | 15/15 |
| Refusal questions passed | 5/5 |
| Answerable evidence accuracy | 100% |
| Refusal detection | 100% |
| Overall evaluation | 20/20 |

These results apply to the predefined evaluation suite. They do not establish complete extraction accuracy across the entire dataset.

Refusal detection uses phrase-based checks and should also be manually reviewed.

## 11. Example Questions

### Supported Questions

The following examples demonstrate queries supported by the current database tools:

- How many floor plans are available?
- How many spaces have been extracted?
- Which floor plan contains the largest extracted space?
- What is the average number of extracted spaces per plan?
- Which spaces have an area greater than 20 square metres?
- Which spaces have an area smaller than 5 square metres?
- What spaces were extracted from floor plan 16?
- What is the largest measured bedroom?
- Which floor plans contain a BAINS space?
- How many extracted spaces have no recorded area?

### Unsupported Questions

The following examples demonstrate questions where the system should avoid inventing information:

- How many doors are present across all floor plans?
- Which floor plan has the most windows?
- What is the exact physical length of the largest wall?
- Which room has no windows?
- What is the exact total usable area of every building?

The system should communicate its data limitations rather than generate unsupported answers.

## 12. Limitations

- Door and window counts are not currently verified.
- Wall geometry and physical wall measurements are not available as validated database records.
- Segmentation-mask colour classes do not have a fully verified semantic mapping.
- Some extracted spaces do not have recorded areas.
- Extracted labels, spaces, and measurements may contain recognition errors.
- Extracted spaces are not necessarily independently verified rooms.
- Physical scale cannot be assumed for every floor plan.
- The current evaluation framework covers 20 predefined questions.
- Refusal detection is phrase-based and requires additional validation.
- The application depends on the configured external API for AI-powered functionality.

## 13. Future Improvements

- Improve extraction accuracy through additional validation and review.
- Establish a verified semantic legend for segmentation-mask classes.
- Implement validated door and window detection.
- Develop wall geometry extraction and measurement.
- Improve OCR and multilingual architectural-label handling.
- Expand the evaluation dataset with more diverse questions.
- Introduce automated regression testing.
- Improve confidence calibration and evidence presentation.
- Support processing of newly uploaded floor plans.
- Improve evaluation coverage and automated quality checks.

## 14. Project Status

The current implementation includes:

- Floor-plan image and segmentation-mask inspection.
- Structured extraction of labelled spaces and available measurements.
- JSON-based extraction storage.
- SQLite database construction.
- AI-powered natural-language question answering.
- Database-backed information retrieval.
- Domain restrictions and unsupported-question handling.
- A predefined 20-question evaluation framework.

Further validation is required before treating extracted architectural information as authoritative building measurements.

## 15. Author

Developed as part of the Ashian Tech Internship project.

**Repository:** https://github.com/roxxerdk/ashian-floorplan-ai
