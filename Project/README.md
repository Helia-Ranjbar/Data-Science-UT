# Data Science Project - TMDb Movie Genre Classification

## 📌 Project Overview

This project is a comprehensive end-to-end data science pipeline for classifying movie genres based on multiple features including textual descriptions, cast and director information, keywords, and financial metadata. The project spans three phases:

1. **Data Scraping & Database Implementation** – Collecting movie data from TMDb and storing in a normalized SQLite database
2. **Feature Engineering & Preprocessing** – Transforming raw data into ML-ready features
3. **Model Development & MLOps** – Building classification models with CI/CD, Docker, and MLflow tracking

---

## 👥 Team Information

- **Students**: Helia Ranjbar, Mehrnaz Hosseini
- **University**: University of Tehran
- **Course**: Data Science
- **Semester**: Spring 2025

---

## Phase 1: Data Scraping & Database Implementation

### Data Source
- **Platform**: The Movie Database (TMDB)
- **Data collected**: Movies, genres, casts, directors, keywords, reviews, financial data

### Data Scraping Process

| Step | Description |
|------|-------------|
| 1 | API connection to TMDb |
| 2 | Fetch movie metadata (title, release date, rating, runtime, certification, overview, tagline, language, budget, revenue) |
| 3 | Extract related entities (genres, cast, directors, keywords) |
| 4 | Collect user reviews |

### Data Preparation Before Database Insertion

| Operation | Description |
|-----------|-------------|
| **Runtime parsing** | Convert '2h 30m' → total minutes |
| **Budget & Revenue cleaning** | Remove $, commas, convert to float |
| **Missing value normalization** | Replace empty lists and dashes with NaN |
| **Release date correction** | Convert '17-Feb-95' → '1995-02-17' |
| **Column explosion** | Extract multi-value columns (cast, genre, keywords) |

### Database Schema (SQLite)

#### Table 1: movies
| Column | Type | Description |
|--------|------|-------------|
| movie_id (PK) | INT | Unique identifier |
| movie_name | VARCHAR(255) | Movie title |
| release_date | DATE | Official release date |
| rating | FLOAT | Viewer rating |
| run_time_minutes | INT | Runtime in minutes |
| certification | VARCHAR(20) | Age-based content rating |
| overview, tagline | TEXT | Description text |
| language | VARCHAR(50) | Original language |
| budget, revenue | BIGINT | Financial data in USD |
| content_score | FLOAT | Computed content relevance score |
| content_score_description | TEXT | Qualitative description |

#### Table 2: movie_genres
| Column | Type |
|--------|------|
| movie_id (FK) | INT |
| genre | VARCHAR(100) |

#### Table 3: movie_casts
| Column | Type |
|--------|------|
| movie_id (FK) | INT |
| actor | VARCHAR(100) |

#### Table 4: movie_directors
| Column | Type |
|--------|------|
| movie_id (FK) | INT |
| director | VARCHAR(100) |

#### Table 5: movie_keywords
| Column | Type |
|--------|------|
| movie_id (FK) | INT |
| normal_keyword_rounded | VARCHAR(100) |
| tone_keyword_bold | VARCHAR(100) |

#### Table 6: movie_reviews
| Column | Type |
|--------|------|
| movie_id (FK) | INT |
| writer | VARCHAR(255) |
| score | FLOAT |
| review | TEXT |
| most_watched_genres | TEXT (JSON) |

---

## Phase 2: EDA, Feature Engineering & Preprocessing


### Key EDA Insights

1. **Budget vs Revenue by Genre** – Action and adventure genres show highest revenue potential
2. **Rating vs Revenue** – Moderate correlation between viewer ratings and box office success
3. **Language Distribution** – English dominates, followed by Spanish, French, and Hindi
4. **Popularity vs Critical Acclaim** – Some genres show disconnect between audience and critic scores

### Feature Selection

| Feature | Processing Method |
|---------|-------------------|
| movie_id | Identifier (not used as feature) |
| movie_name | Dropped (identifier) |
| rating | Standardization (normal distribution) |
| budget_revenue_ratio | New feature, standardized |
| certification | Ordinal encoding (7 categories) |
| overview | Sentence-BERT embeddings |
| review | Sentence-BERT embeddings (aggregated) |
| casts | One-hot encoding (top 100 actors) |
| directors | One-hot encoding (top 100 directors) |
| normal_keyword_rounded | TF-IDF vectorization |
| tone_keyword_bold | TF-IDF vectorization |
| genre (target) | Multi-label encoding (17 genres) |

### Preprocessing Details

#### Rating Column
- Distribution: Normal
- Method: Standardization (not normalization)
- Null values (8): Filled with column mean

#### Certification Column Encoding

| Category | Label |
|----------|-------|
| General Audience (All ages) | 1 |
| Parental Guidance (~7+) | 2 |
| Teen Audiences (~12-16) | 3 |
| Older Teens (~14-16) | 4 |
| Mature (~16-18) | 5 |
| Adults Only (18+) | 6 |
| Special/Regional/Null | 7 |

#### Text Vectorization
- **Overview & Reviews**: Sentence-BERT (semantic embeddings)
- **Keywords**: TF-IDF (term frequency-inverse document frequency)

#### Actor/Director Encoding
- Selection: Top 100 most frequent actors and directors
- Method: One-hot encoding per movie

#### Target Variable (Genre)
- Type: Multi-label classification
- Total genres: 17
- Encoding: Binary vector per movie (1 if genre present, 0 otherwise)
- Unknown genre labeled as "unknown"

---

## Phase 3: Model Development & MLOps

### Pipeline Architecture

```
Database → Data Loading → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction
```

### CI/CD Implementation (GitHub Actions)

**Workflow stages:**
1. Code checkout
2. Python environment setup
3. Dependency installation
4. Linting and formatting checks
5. Unit tests
6. Model training (triggered on push to main)
7. MLflow experiment logging
8. Docker image build and push

### Docker Containerization

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
CMD ["python", "src/06_prediction.py"]
```

### MLflow Integration

**Tracked metrics:**
- Loss (train/validation)
- Accuracy
- Precision
- Recall
- F1-score (macro, micro, weighted)
- Training duration

**Logged artifacts:**
- Model parameters
- Evaluation metrics
- Confusion matrices
- ROC curves

---

## Model Performance Comparison

| Model | Duration (s) | F1-Macro | F1-Micro | Precision | Recall | Final Val Loss |
|-------|--------------|----------|----------|-----------|--------|----------------|
| Neural Network (DNN) | 33.74 | 0.484 | 0.691 | 0.782 | 0.618 | 0.260 |
| XGBoost | 33.09 | 0.478 | 0.718 | 0.809 | 0.645 | 0.282 |
| LightGBM | 168.50 | 0.596 | 0.677 | 0.689 | 0.666 | 0.673 |

### Key Observations
- **LightGBM** achieved highest F1-Macro (0.596) but took longest to train
- **XGBoost** had best F1-Micro (0.718) and precision (0.809)
- **Neural Network** balanced performance with moderate training time

### Final Model Selection
**Ensemble: DNN + XGBoost**
- Combines semantic understanding (DNN) with gradient-boosted feature importance (XGBoost)
- Achieves best overall performance across all metrics