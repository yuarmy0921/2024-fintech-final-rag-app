# 2024-fintech-final-rag-app
## 1. Installation

Install Poetry if you haven't already:
```
pip install poetry
```

Then, install the project dependencies:
```
poetry install
```

Install Tesseract OCR 5 on user system. I expect you to use an Ubuntu system:
- [Install Tesseract OCR 5 on Ubuntu: A Complete Guide](https://www.wwwinsights.com/tesseract-ocr-5-ubuntu/)

Running the virtual environment:
```
poetry shell
```

## 2. Database Preparation
You can skip [data preprocessing](#data-preprocessing) by just importing the database snapshot (`Preprocess/neo4j.backup`). Follow the steps provided in the docs: [Importing an existing database](https://neo4j.com/docs/aura/auradb/importing/import-database/)

### Access to Neo4j AuraDB and GCP
1. Register an account for [Neo4j AuraDB](https://neo4j.com/product/auradb/). You'll have one free instance to use.
2. Create a new project in Google Cloud Platform (GCP) and enable the following APIs:
    - Vertex AI API
    - API Keys API
    - Discovery Engine API

It may incur some charge if your free credits are exhausted.

3. Prepare a `.env` file under `module/`, where should contain the following information:
```
NEO4J_URI=
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=
NEO4J_DATABASE=neo4j
PROJECT_ID=
REGION=asia-east1-a
```
Fill in your credentials accordingly.

4. Generate an API key for accessing GCP services.
```
python3 gen_gcp_api_key.py --project_id {YOUR_PROJECT_ID}
```

### Data preprocessing
1. Prepare the following directory structure at the project root.
```
data
├── cleaned
│   ├── faq
│   ├── finance
│   └── insurance
├── raw
│   ├── faq
│   ├── finance
│   └── insurance
└── tmp
    ├── finance
    └── insurance
```
2. Under `cleaned` directory, put your source documents according to the category.
3. Preprocessing data
```
cd Preprocess && python3 db.py
```
Make sure to use an Ubuntu system, or some errors wiil occur during the execution. 

## 3. Answer Retrieval
### Vector cosine similarity approach
```
cd Retrieval && python3 qa.py --question {YOUR_QUESTION_JSON_FILE} --pred {YOUR_OUTPUT_JSON_FILE}
```

## 4. Get Final Prediction
