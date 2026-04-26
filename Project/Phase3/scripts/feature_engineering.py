import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
from preprocess import preprocess_data

def feature_engineering():
    with mlflow.start_run(run_name="Feature Engineering"):
        final_df, df_dict = preprocess_data()

        movies = df_dict["movies"]
        movie_genres = df_dict["movie_genres"]
        movie_casts = df_dict["movie_casts"]
        movie_directors = df_dict["movie_directors"]
        movie_keywords = df_dict["movie_keywords"]
        movie_reviews = df_dict["movie_reviews"]

        final_df['budget_revenue_ratio'] = movies.apply(
            lambda row: row['budget'] / row['revenue'] if row['revenue'] != 0 else 0, axis=1
        )
        Q1 = final_df['budget_revenue_ratio'].quantile(0.25)
        Q3 = final_df['budget_revenue_ratio'].quantile(0.75)
        IQR = Q3 - Q1
        final_df['budget_revenue_ratio'] = np.clip(final_df['budget_revenue_ratio'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        final_df['budget_revenue_ratio'] = StandardScaler().fit_transform(final_df[['budget_revenue_ratio']])
        final_df['budget_revenue_ratio'] = final_df['budget_revenue_ratio'].fillna(final_df['budget_revenue_ratio'].mean())

        final_df['overview'] = movies['overview'].fillna('').str.strip().str.replace('\n', ' ')
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        overview_embeddings = embed_model.encode(final_df['overview'].tolist(), show_progress_bar=True)
        final_df['overview'] = list(overview_embeddings)
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")

        movie_reviews_clean = movie_reviews[['movie_id', 'review']].copy()
        movie_reviews_clean['review'] = movie_reviews_clean['review'].fillna('').str.strip().str.replace('\n', ' ')
        reviews_grouped = movie_reviews_clean.groupby('movie_id')['review'].apply(lambda x: ' '.join(x)).reset_index()
        review_embeddings = embed_model.encode(reviews_grouped['review'].tolist(), show_progress_bar=True)
        final_df['review'] = list(review_embeddings)

        top_100_actors = movie_casts['cast'].value_counts().head(100).index.tolist()
        filtered_cast = movie_casts[movie_casts['cast'].isin(top_100_actors)]
        actor_ohe = pd.crosstab(filtered_cast['movie_id'], filtered_cast['cast'])
        actor_ohe.columns = ['actor_' + c.replace(' ', '_') for c in actor_ohe.columns]
        final_df = final_df.merge(actor_ohe, how='left', left_on='movie_id', right_index=True).fillna(0)
        mlflow.log_param("num_top_actors", len(top_100_actors))

        top_100_directors = movie_directors['director'].value_counts().head(100).index.tolist()
        filtered_directors = movie_directors[movie_directors['director'].isin(top_100_directors)]
        director_ohe = pd.crosstab(filtered_directors['movie_id'], filtered_directors['director'])
        director_ohe.columns = ['director_' + c.replace(' ', '_') for c in director_ohe.columns]
        final_df = final_df.merge(director_ohe, how='left', left_on='movie_id', right_index=True).fillna(0)
        mlflow.log_param("num_top_directors", len(top_100_directors))

        keyword_grouped = movie_keywords.groupby('movie_id').agg({
            'normal_keyword_(rounded)': lambda x: ' '.join(sorted(set(x.dropna().astype(str)))),
            'tone_keyword_(bold)': lambda x: ' '.join(sorted(set(x.dropna().astype(str))))
        }).reset_index()

        tfidf = TfidfVectorizer(max_features=300)
        tfidf_normal = tfidf.fit_transform(keyword_grouped['normal_keyword_(rounded)'])
        normal_df = pd.DataFrame(tfidf_normal.toarray(), columns=tfidf.get_feature_names_out())
        normal_df = normal_df.add_suffix("_normal_kw")
        final_df = pd.concat([final_df, normal_df], axis=1)

        tfidf_tone = tfidf.fit_transform(keyword_grouped['tone_keyword_(bold)'])
        tone_df = pd.DataFrame(tfidf_tone.toarray(), columns=tfidf.get_feature_names_out())
        tone_df = tone_df.add_suffix("_tone_kw")
        final_df = pd.concat([final_df, tone_df], axis=1)

        mlflow.log_param("tfidf_max_features", 300)

        genres = movie_genres.groupby('movie_id')['genre'].apply(list).reset_index()
        genres['genre'] = genres['genre'].apply(lambda g: ['unknown' if pd.isna(x) else x for x in g])
        mlb = MultiLabelBinarizer()
        genre_encoded = pd.DataFrame(mlb.fit_transform(genres['genre']), columns=["genre_" + g for g in mlb.classes_])
        genre_encoded['movie_id'] = genres['movie_id']
        final_df = final_df.merge(genre_encoded, on='movie_id', how='left')
        mlflow.log_metric("num_genres", len(mlb.classes_))

        os.makedirs("data", exist_ok=True)
        final_path = "data/final_df.csv"
        final_df.to_csv(final_path, index=False)
        mlflow.log_artifact(final_path)

        sample_path = "data/sample_features.csv"
        final_df.sample(5).to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path)

        mlflow.log_metric("final_num_features", final_df.shape[1])
        mlflow.log_metric("final_num_samples", final_df.shape[0])

        mlflow.set_tag("step", "feature_engineering")
        mlflow.set_tag("script", "feature_engineering.py")

        print("Feature engineering logged to MLflow")

if __name__ == "__main__":
    feature_engineering()
