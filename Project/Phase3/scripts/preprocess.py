from sklearn.preprocessing import StandardScaler
import pandas as pd
import mlflow
import mlflow.sklearn
from load_data import load_all_tables
import os

TABLES = ["movies", "movie_genres", "movie_casts", "movie_directors", "movie_keywords", "movie_reviews"]

def preprocess_data():
    df_dict = load_all_tables()

    if df_dict:
        movies = df_dict.get("movies")

    final_df = pd.DataFrame()
    final_df['movie_id'] = movies['movie_id']
    final_df['movie_name'] = movies['movie_name']
    final_df['rating'] = movies['rating']
    final_df['certification'] = movies['certification']

    scaler = StandardScaler()
    final_df[['rating']] = scaler.fit_transform(final_df[['rating']])
    final_df['rating'] = final_df['rating'].fillna(final_df['rating'].mean())

    cert_map = {
        'G': 'G', 'U': 'G', 'ALL': 'G', 'AL': 'G', 'Genel İzleyici Kitlesi': 'G',
        'PG': 'PG', 'PG-12': 'PG', 'PG-13': 'PG', 'PG13': 'PG', 'PG12': 'PG', 
        '6': 'PG', '6+': 'PG', '7': 'PG', '8': 'PG', 'T': 'PG', 'UA': 'PG',
        '12': '12', '12A': '12', '12+': '12', '12PG': '12', 'K12': '12', 
        'M/12': '12', 'NC16': '12', 'PG-12': '12',
        '14': '15', '14+': '15', '14A': '15', '15': '15', '15A': '15', 
        '15+': '15', 'M/14': '15', 'R': '15', 'R15+': '15', 'R-15': '15', 
        'B-15': '15', 'MA15+': '15', 'MA 15+': '15', 'K-16': '15', 
        'KT': '15', 'K-15': '15', '15세 이상 관람가': '15', 'NC16': '15',
        '16': '18', '16+': '18', 'M/16': '18', 'M18': '18', 'VM18': '18', 
        'R18': '18', 'R 18+': '18', 'R-18': '18', 'NC-17': '18', 'III': '18', 
        'N-18': '18', '청소년 관람불가': '18', '19': '18', '21+': '18',
        '18': '18', '18+': '18', '18A': '18', 'IIB': '18', 'C': '18', 'ITA': '18',
        'NR': 'unknown', 'TP': 'unknown', 'B': 'unknown', 'B15': 'unknown', 
        'KN': 'unknown', 'S': 'unknown', 'Κ': 'unknown', 'II': 'unknown', 
        'IIA': 'unknown', 'AA': 'unknown', 'Btl': 'unknown', 'K-18': 'unknown',
        'K-7': 'unknown', 'Κ-15': 'unknown', 'APTA': 'unknown', None: 'unknown'
    }

    final_df['certification'] = movies['certification'].map(cert_map).fillna('unknown')
    ordinal_map = {'unknown': -1, 'G': 0, 'PG': 1, '12': 2, '15': 3, '18': 4}
    final_df['certification'] = final_df['certification'].map(ordinal_map)

    return final_df, df_dict


if __name__ == "__main__":
    with mlflow.start_run(run_name="Preprocessing"):
        final_df, df_dict = preprocess_data()

        mlflow.log_param("scaling", "StandardScaler")
        mlflow.log_param("certification_encoding", "ordinal")

        mlflow.log_metric("num_movies", len(final_df))
        mlflow.log_metric("num_missing_ratings", final_df['rating'].isnull().sum())
        mlflow.log_metric("num_unknown_certifications", (final_df['certification'] == -1).sum())

        schema_info = {
            "columns": final_df.columns.tolist(),
            "dtypes": final_df.dtypes.astype(str).to_dict()
        }
        mlflow.log_dict(schema_info, "preprocessing_schema.json")

        preview_path = "preprocessing_outputs/sample_preprocessed.csv"
        os.makedirs("preprocessing_outputs", exist_ok=True)
        final_df.head(100).to_csv(preview_path, index=False)
        mlflow.log_artifact(preview_path)

        print("Preprocessing step completed and logged to MLflow.")
