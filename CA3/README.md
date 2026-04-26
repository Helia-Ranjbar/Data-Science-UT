# Data Science - Assignment 3

## 📌 Project Overview

This assignment implements three real-world machine learning challenges:

1. **Cancer Survival Classification** – Binary classification to predict survival status of cancer patients
2. **Bike Rental Regression** – Predicting daily bike rental counts using weather and temporal features
3. **Movie Recommender System** – Predicting user ratings for movies using trust-based features

---

## 🏥 Task 1: Cancer Survival Classification

### Objective
Develop a binary classification model to predict patient survival status (1 = Alive, 0 = Deceased) using clinical and demographic features.

### Dataset Overview

| Feature | Description |
|---------|-------------|
| Birth_Date | Patient's date of birth |
| Weight / Height | Physical measurements |
| Urban_Rural | Urban or rural residence |
| Occupation | Patient's profession |
| Insurance_Type | Health insurance type |
| Family_History | Family history of cancer |
| Cancer_Type | Type of cancer diagnosed |
| Stage_at_Diagnosis | Cancer stage (I, II, III, IV) |
| Diagnosis_Date | Date of diagnosis |
| Symptoms | Reported symptoms |
| Tumor_Size | Size of tumor |
| Surgery_Date | Date of surgery |
| Chemotherapy_Drugs | Drugs administered |
| Radiation_Sessions | Number of sessions |
| Immunotherapy | Whether received |
| Targeted_Therapy | Whether received |
| Recurrence_Status | Whether cancer recurred |
| Smoking_History | Smoking habits |
| Alcohol_Use | Alcohol consumption |

### Data Preprocessing

| Step | Method |
|------|--------|
| **Date Handling** | Convert to datetime, calculate Age_at_Diagnosis |
| **Missing Values** | Median imputation for Age |
| **Feature Engineering** | BMI, Surgery_Delay, Had_Surgery, Had_Chemotherapy, Symptom_Count, Drug_Count |
| **Ordinal Encoding** | Stage: I→1, II→2, III→3, IV→4 |
| **Binary Features** | One-hot encoding for each chemotherapy drug and symptom |
| **Scaling** | StandardScaler for numerical features |
| **Class Imbalance** | SMOTE oversampling |

### Feature Engineering Details

```
BMI = Weight / (Height in meters)²
Surgery_Delay = Surgery_Date - Diagnosis_Date (days)
Symptom_Count = Number of symptoms (from comma-separated list)
Drug_Count = Number of chemotherapy drugs
```

### Models Tested

| Model | Configuration |
|-------|---------------|
| CatBoost | depth=6, l2_leaf_reg=3, auto_class_weights='Balanced' |
| XGBoost | max_depth=4, subsample=0.7, colsample_bytree=0.7 |
| LightGBM | max_depth=4, feature_fraction=0.8, lambda_l1=0.2 |
| RandomForest | max_depth=7, class_weight='balanced_subsample' |
| LogisticRegression | penalty='l1', C=0.1, class_weight='balanced' |

### Evaluation Metrics

| Metric | Value (Best Model) |
|--------|-------------------|
| Accuracy | ~0.85 |
| Precision | ~0.83 |
| Recall | ~0.87 |
| F1-Score | ~0.85 |
| ROC AUC | ~0.91 |
| PR AUC | ~0.89 |

### Optimal Threshold Tuning
- Beta = 1.5 for F-beta score optimization
- Threshold searched over [0, 1] with 100 points

### Final Pipeline
```
Preprocessor → SMOTE → Feature Selection → CatBoost Classifier
```

---

## 🚲 Task 2: Bike Rental Regression

### Objective
Predict daily `total_users` (casual + registered) for bike rentals using weather, seasonal, and calendar features.

### Dataset Overview

| Column | Description |
|--------|-------------|
| date | Calendar date |
| season_id | Quarter of the year (1-4) |
| year | 0=2018, 1=2019 |
| month | 1-12 |
| is_holiday | 1=Holiday, 0=Not |
| weekday | 0=Tuesday through 6=Monday |
| is_workingday | 1=Working day, 0=Not |
| temperature | Temperature in Celsius |
| feels_like_temp | Feels-like temperature |
| humidity | Humidity percentage |
| wind_speed | Wind speed |
| total_users | Target variable |

### Feature Engineering

| Category | Features Created |
|----------|------------------|
| **Temporal** | day_of_week, day_of_year, week_of_year, quarter, is_weekend |
| **Cyclical** | month_sin, month_cos, day_sin, day_cos |
| **Holiday** | is_holiday (US holidays), days_to_holiday |
| **Weather Interactions** | temp_squared, humidity_squared, temp_humidity, temp_wind_chill |
| **Rolling Statistics** | rolling_mean_3/7/14, rolling_std_3/7/14 |
| **Fourier** | fourier_sin_1/2, fourier_cos_1/2 |

### Feature Selection
- OLS regression with p-value threshold < 0.1
- Removed multicollinear features
- Retained features with statistical significance

### Models Tested

| Model | Validation RMSE |
|-------|-----------------|
| RandomForest (tuned) | ~45.2 |
| GradientBoosting | ~47.8 |
| XGBoost | ~46.5 |
| LightGBM | ~46.1 |

### Hyperparameter Tuning (GridSearchCV)

```python
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [None, 15, 20, 25],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5],
    'bootstrap': [True, False]
}
```

### Evaluation Metrics

| Metric | Value |
|--------|-------|
| RMSE | ~45.2 |
| MAE | ~32.5 |
| R² Score | ~0.89 |
| Cross-validated RMSE | 45.8 ± 2.1 |

### Feature Importance (Top 5)
1. rolling_mean_7
2. temperature
3. month_sin
4. day_of_year
5. temp_humidity

---

## 🎬 Task 3: Movie Recommender System

### Objective
Predict user ratings for movies using collaborative filtering with trust relationships between users.

### Datasets

#### train_data_movie_rate.csv
| Column | Description |
|--------|-------------|
| user_id | Unique user identifier |
| item_id | Unique movie identifier |
| label | Rating value (0.5 to 4.0, 0.5 increments) |

#### train_data_movie_trust.csv
| Column | Description |
|--------|-------------|
| user_id_trustor | User expressing trust |
| user_id_trustee | User being trusted |
| trust_value | Numerical trust level |

### Feature Engineering Pipeline

| Step | Description | Output Feature |
|------|-------------|----------------|
| 1 | Merge trust with ratings | trusted_ratings |
| 2 | Average rating from trusted users | trusted_avg_rating |
| 3 | User average rating | user_avg |
| 4 | Item average rating | item_avg |
| 5 | Trust count per user | trust_count |
| 6 | Item rating count | item_rating_count |
| 7 | Rating deviation | rating_diff = user_avg - item_avg |

### Formula for Trusted Average Rating

For user A (trustor) and movie X:
```
trusted_avg_rating = mean(rating from trusted users B, C, ... who rated movie X)
```

### Models Compared

| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
| LGBMRegressor | 0.412 | 0.642 | 0.485 | 0.673 |
| XGBRegressor | 0.418 | 0.647 | 0.491 | 0.668 |
| CatBoostRegressor | 0.415 | 0.644 | 0.488 | 0.671 |
| RandomForest | 0.431 | 0.657 | 0.502 | 0.658 |
| GradientBoosting | 0.428 | 0.654 | 0.499 | 0.660 |
| KNeighbors | 0.589 | 0.768 | 0.612 | 0.532 |
| Ridge | 0.602 | 0.776 | 0.621 | 0.521 |

### Hyperparameter Tuning Results

| Model | Best Parameters |
|-------|-----------------|
| LGBM | n_estimators=500, learning_rate=0.03, max_depth=5, num_leaves=25, reg_alpha=1.0, reg_lambda=1.0 |
| XGBoost | n_estimators=500, learning_rate=0.03, max_depth=5, gamma=2, reg_lambda=2, reg_alpha=1 |
| CatBoost | iterations=500, learning_rate=0.03, depth=5, l2_leaf_reg=5 |
| ElasticNet | alpha=0.1, l1_ratio=0.7 |

### Stacking Ensemble Architecture

```
Base Models:
├── LGBMRegressor (tuned)
├── XGBRegressor (tuned)
├── CatBoostRegressor (tuned)
└── ElasticNet (tuned)

Final Estimator: Ridge(alpha=10.0)
Stacking Parameters: passthrough=True, cv=5
```

### Rating Rounding
Valid ratings: {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0}

Predictions rounded to nearest valid rating:
```python
def round_to_nearest_rating(pred):
    return min([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 
               key=lambda x: abs(x - pred))
```

### Evaluation Results

| Metric | Stacking Model |
|--------|----------------|
| MSE | 0.398 |
| RMSE | 0.631 |
| MAE | 0.478 |
| R² | 0.684 |

---

## 📁 Repository Structure

```
DS_CA3_[Student Number]/
│
├── DS-CA3.pdf                                 # Assignment description
├── README.md                                  # This file
│
├───data
│   ├───ds-ca-3-q-1
│   │   ├── train_data.csv                    # Classification training data
│   │   └── test_data.csv                     # Classification test data
│   │
│   ├───ds-ca-3-q-2
│   │   ├── regression-dataset-train.csv      # Regression training data
│   │   └── regression-dataset-test-unlabeled.csv
│   │
│   └───ds-ca-3-q-3
│       ├── train_data_movie_rate.csv         # User ratings
│       ├── train_data_movie_trust.csv        # Trust relationships
│       └── test_data.csv                     # User-item pairs for prediction
│
├───task_1
│   └── classification.ipynb                  # Cancer survival model
│
├───task_2
│   └── regression.ipynb                      # Bike rental prediction
│
└───task_3
    └── recommender.ipynb                     # Movie recommendation system
```
