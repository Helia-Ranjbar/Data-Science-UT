import numpy as np
import pandas as pd
import sqlite3
from tensorflow.keras.models import load_model
from joblib import load
from sklearn.metrics import f1_score, precision_score, recall_score
import os

def save_predictions_to_db(test_df, y_pred, db_path, table_name="predictions"):
    conn = sqlite3.connect(db_path)
    pred_df = pd.DataFrame(y_pred, columns=[f"pred_genre_{i}" for i in range(y_pred.shape[1])])

    result_df = pd.concat([test_df.reset_index(drop=True), pred_df], axis=1)
    result_df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"Predictions saved to table '{table_name}' in database '{db_path}'.")

def make_predictions(model_dir="models", test_data_path="data/test_data.npz", db_path="database/final_db.db"):
    test_data = np.load(test_data_path)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]


    feature_extractor = load_model(f"{model_dir}/feature_extractor.keras")
    multi_xgb = load(f"{model_dir}/multi_xgb.joblib")

    X_test_feat = feature_extractor.predict(X_test)
    y_pred = multi_xgb.predict(X_test_feat)

    f1_micro = f1_score(y_test, y_pred, average="micro")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")

    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    save_predictions_to_db(X_test, y_pred, db_path)

if __name__ == "__main__":
    make_predictions()
