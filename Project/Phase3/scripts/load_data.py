import pandas as pd
import mlflow
from database_connection import create_connection

TABLES = ["movies", "movie_genres", "movie_casts", "movie_directors", "movie_keywords", "movie_reviews"]

def load_all_tables():
    """Load all tables into separate Pandas DataFrames and log with MLflow."""
    conn = create_connection()

    with mlflow.start_run(run_name="Load All Tables"):
        if conn:
            dataframes = {}

            for table in TABLES:
                query = f"SELECT * FROM {table}"
                df = pd.read_sql_query(query, conn)
                dataframes[table] = df

                mlflow.log_metric(f"{table}_rows", df.shape[0])
                mlflow.set_tag(f"{table}_columns", ','.join(df.columns))

            conn.close()

            mlflow.log_param("tables_loaded", len(dataframes))
            mlflow.set_tag("status", "success")
            return dataframes
        else:
            mlflow.set_tag("status", "db_connection_failed")
            return None

if __name__ == "__main__":
    df_dict = load_all_tables()

    if df_dict:
        for table_name, df in df_dict.items():
            print(f"Loaded {table_name}: {df.shape[0]} rows")
