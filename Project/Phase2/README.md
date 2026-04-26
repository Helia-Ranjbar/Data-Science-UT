# 🎬 Movie Dataset Processing & Feature Engineering Pipeline

This project provides a full pipeline for preprocessing a movie dataset, importing it into a structured SQLite database, extracting and transforming key metadata, and engineering advanced features suitable for machine learning models or analysis.

---

## 📁 Directory Structure

```
.
├── data/
│   ├── movies_data.csv              # Raw input data
│   └── final_df.csv                 # Final output after feature engineering
├── database/
│   └── dataset.db                   # SQLite database (auto-generated)
├── scripts/
│   ├── preprocess_and_import.py     # Step 1 - Preprocessing & DB insertion
│   ├── database_connection.py       # Step 2 - SQLite connection function
│   ├── load_data.py                 # Step 3 - Load all tables from DB
│   ├── preprocess.py                # Step 4 - Normalize rating & certification
│   ├── feature_engineering.py       # Step 5 - Feature engineering on all data
├── pipeline.py                      # 🔁 Master script that runs all steps
└── README.md                        # 📖 This file
```

---

## ✅ Key Features

* Cleans and normalizes raw movie metadata
* Imports into a relational SQLite database
* Maps international certifications to ordinal scale
* Embeds overviews and reviews using `SentenceTransformer`
* One-hot encodes top actors and directors
* Applies TF-IDF to keywords
* Binarizes genres using `MultiLabelBinarizer`
* Produces a model-ready `final_df.csv` for ML pipelines

---

## 🔁 Step-by-Step Workflow

### **Step 1: Preprocess and Import**

```bash
python scripts/preprocess_and_import.py
```

* Parses and cleans raw CSV (`movies_data.csv`)
* Normalizes nested fields like genres, cast, and reviews
* Populates six normalized tables in `database/dataset.db`:

  * `movies`
  * `movie_genres`
  * `movie_casts`
  * `movie_directors`
  * `movie_keywords`
  * `movie_reviews`

---

### **Step 2: Connect to Database**

```python
from database_connection import create_connection
conn = create_connection()
```

* Uses relative paths to connect to SQLite database.
* Reusable across scripts.

---

### **Step 3: Load Tables from Database**

```bash
python scripts/load_data.py
```

* Loads all 6 normalized tables into pandas DataFrames.
* Validates table presence and prints row counts.

---

### **Step 4: Basic Preprocessing**

```bash
python scripts/preprocess.py
```

* Normalizes movie rating with `StandardScaler`
* Maps global `certification` labels into consistent ordinal scale (`G`, `PG`, `12`, `15`, `18`, `unknown`)
* Returns the `final_df` foundation for further feature engineering

---

### **Step 5: Feature Engineering**

```bash
python scripts/feature_engineering.py
```

* Computes normalized `budget_revenue_ratio`
* Embeds movie `overview` and grouped `reviews` using Sentence-BERT (`all-MiniLM-L6-v2`)
* One-hot encodes top 100 actors and top 100 directors
* Applies TF-IDF on:

  * `normal_keyword_(rounded)`
  * `tone_keyword_(bold)`
* Binarizes genre lists per movie
* Outputs a clean, ML-ready `data/final_df.csv`

---

## ⚙️ Run the Full Pipeline

You can run all five steps automatically using:

```bash
python pipeline.py
```

---

## 📦 Requirements

Install all dependencies via pip:

```bash
pip install pandas numpy scikit-learn sentence-transformers
```

---

## 🗃️ Database Schema Overview

| Table             | Description                               |
| ----------------- | ----------------------------------------- |
| `movies`          | Core metadata: name, rating, budget, etc. |
| `movie_genres`    | One row per genre per movie               |
| `movie_casts`     | One row per actor per movie               |
| `movie_directors` | One row per director per movie            |
| `movie_keywords`  | Cleaned and categorized keywords          |
| `movie_reviews`   | User reviews with rating and preferences  |

---

## 📌 Output File

**`data/final_df.csv`** contains all structured and engineered features.
