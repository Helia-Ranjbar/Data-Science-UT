import pandas as pd
import numpy as np
import sqlite3
import mlflow
import mlflow.keras
import mlflow.sklearn
import time
import os
from joblib import dump
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import xgboost as xgb


def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def load_and_prepare_data(df):
    genre_columns = ['genre_Comedy', 'genre_Crime', 'genre_Documentary',
                     'genre_Drama', 'genre_Family', 'genre_Fantasy', 'genre_History',
                     'genre_Horror', 'genre_Music', 'genre_Mystery', 'genre_Romance',
                     'genre_Science Fiction', 'genre_TV Movie', 'genre_Thriller',
                     'genre_War', 'genre_Western', 'genre_unknown']

    y = df[genre_columns]
    X = df.drop(columns=["movie_id", "movie_name"] + genre_columns)

    X['overview_parsed'] = X['overview'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
    overview_array = np.vstack(X['overview_parsed'].values)
    overview_df = pd.DataFrame(overview_array, columns=[f"overview_emb_{i}" for i in range(overview_array.shape[1])])
    X = pd.concat([X.drop(columns=['overview', 'overview_parsed']), overview_df], axis=1)

    X['review_parsed'] = X['review'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
    review_array = np.vstack(X['review_parsed'].values)
    review_df = pd.DataFrame(review_array, columns=[f"review_emb_{i}" for i in range(review_array.shape[1])])
    X = pd.concat([X.drop(columns=['review', 'review_parsed']), review_df], axis=1)

    return X, y

def train_model(db_path, table_name, model_dir="models", test_data_path="data/test_data.npz"):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

    df = load_data_from_sqlite(db_path, table_name)
    X, y = load_and_prepare_data(df)

    # 80% train, 10% val, 10% test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42, shuffle=True)

    np.savez(test_data_path, X_test=X_test.to_numpy(), y_test=y_test.to_numpy())

    start_time = time.time()

    dnn_params = {
        "layers": "512-512-512-256",
        "dropout_rate": 0.3,
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "epochs": 60,
        "batch_size": 64
    }

    xgb_params = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "verbosity": 0
    }

    with mlflow.start_run() as run:
        mlflow.log_params({**dnn_params, **xgb_params})

        input_layer = Input(shape=(X_train.shape[1],))
        x = Dense(512, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)

        feature_output = x
        output_layer = Dense(17, activation='sigmoid')(feature_output)

        dnn = Model(inputs=input_layer, outputs=output_layer)
        dnn.compile(loss='binary_crossentropy', optimizer=Adam(dnn_params['learning_rate']), metrics=['accuracy'])

        history = dnn.fit(X_train, y_train, epochs=dnn_params['epochs'], batch_size=dnn_params['batch_size'],
                          validation_data=(X_val, y_val), verbose=1)

        feature_extractor = Model(inputs=dnn.input, outputs=feature_output)
        X_train_dnn_feat = feature_extractor.predict(X_train)

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0,
            **xgb_params
        )
        multi_xgb = MultiOutputClassifier(xgb_model)
        multi_xgb.fit(X_train_dnn_feat, y_train)

        metrics = {
            "training_time": time.time() - start_time,
            "final_dnn_loss": history.history['loss'][-1],
            "final_dnn_val_loss": history.history['val_loss'][-1]
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        dnn_input_schema = Schema([TensorSpec(np.float32, shape=(-1, X_train.shape[1]))])
        dnn_output_schema = Schema([TensorSpec(np.float32, shape=(-1, 256))])
        xgb_input_schema = Schema([TensorSpec(np.float32, shape=(-1, 256))])
        xgb_output_schema = Schema([TensorSpec(np.float32, shape=(-1, 17))])

        mlflow.keras.log_model(
            feature_extractor,
            "dnn_feature_extractor",
            signature=ModelSignature(dnn_input_schema, dnn_output_schema),
            registered_model_name="Hybrid_DNN_Feature_Extractor",
            metadata={"params": dnn_params, "metrics": metrics}
        )

        mlflow.sklearn.log_model(
            multi_xgb,
            "xgb_model",
            signature=ModelSignature(xgb_input_schema, xgb_output_schema),
            registered_model_name="Hybrid_XGBoost",
            metadata={"dnn_params": dnn_params, "xgb_params": xgb_params, "metrics": metrics}
        )

        feature_extractor.save(f"{model_dir}/feature_extractor.keras")
        dump(multi_xgb, f"{model_dir}/multi_xgb.joblib")

        print(f"Run completed: {run.info.run_id}")
        print(f"Training time: {metrics['training_time']:.2f}s\n")

if __name__ == "__main__":
    db_path = "database/final_db.db"
    table_name = "final_movie_db"
    train_model(db_path, table_name)
