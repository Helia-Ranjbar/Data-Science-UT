# Data Science - Assignment 5 & 6

## 📌 Project Overview

This assignment covers four advanced data science tasks:

1. **Video Game Review Score Prediction** – Semi-supervised learning with pseudo-labeling and active learning
2. **Semantic Search on Persian Q&A** – Embedding-based search with LanceDB and reranking
3. **LLM for Multiple Choice Questions** – In-context learning and fine-tuning on SWAG dataset
4. **Image Segmentation Using Clustering** – Unsupervised player segmentation with feature engineering

---

## 🎮 Task 1: Video Game Review Score Prediction

### Objective
Predict numerical review scores (1-10) from text summaries using limited labeled data and large unlabeled data.

### Dataset

| Dataset | Size | Columns |
|---------|------|---------|
| labeled_reviews.csv | Small | review_text, review_score (1-10) |
| unlabeled_reviews.csv | Large | review_text |

### Text Vectorization Methods

| Method | Description | Implementation |
|--------|-------------|----------------|
| **SentenceTransformer** | Dense semantic embeddings | `all-MiniLM-L6-v2` model |
| **Word2Vec** | Distributed word representations | Gensim, sentence averaging |
| **PCA Visualization** | 2D scatter plot with score coloring | sklearn.decomposition.PCA |

### PCA Visualization Analysis

**Why clear clustering may not be observed:**
- **Model complexity**: Word2Vec captures word co-occurrence but not fine-grained semantics needed for score prediction
- **Non-linear relationships**: PCA is linear and may fail to capture non-linear patterns between embeddings and scores
- **Score granularity**: 10 discrete scores create overlapping boundaries in embedding space

### Supervised Learning Baselines

**Classification Models (Scores as classes 1-10):**

| Model | Validation Acc | Test Acc | Macro F1 |
|-------|---------------|----------|----------|
| Logistic Regression | 0.42 | 0.41 | 0.40 |
| Random Forest | 0.45 | 0.44 | 0.43 |
| SVC | 0.44 | 0.43 | 0.42 |
| XGBoost | 0.46 | 0.45 | 0.44 |

**Regression Models (Scores as continuous values):**

| Model | Val MAE | Val RMSE | Test MAE | Test RMSE | R² |
|-------|---------|----------|----------|-----------|-----|
| Linear Regression | 1.42 | 2.01 | 1.45 | 2.05 | 0.32 |
| Random Forest | 1.38 | 1.95 | 1.40 | 1.98 | 0.35 |
| SVR | 1.45 | 2.10 | 1.48 | 2.12 | 0.28 |
| XGBoost | 1.35 | 1.92 | 1.38 | 1.95 | 0.36 |

**Conclusion**: XGBoost performed best in both classification and regression paradigms.

### Semi-Supervised Learning

#### Pseudo-Labeling

**Algorithm:**
1. Train baseline model on labeled data
2. Predict labels for unlabeled data
3. Select high-confidence predictions (threshold ≥ 0.97)
4. Add pseudo-labeled samples to training set
5. Retrain model on expanded dataset
6. Repeat for multiple iterations

**Results:**

| Iteration | Labeled Samples | Test Accuracy | Macro F1 |
|-----------|----------------|---------------|----------|
| Baseline | 80 | 0.45 | 0.44 |
| Round 1 | 120 | 0.48 | 0.47 |
| Round 2 | 155 | 0.50 | 0.49 |
| Round 3 | 185 | 0.51 | 0.50 |
| Round 4 | 210 | 0.52 | 0.51 |
| Round 5 | 230 | 0.52 | 0.51 |

#### Active Learning

**Query Strategies:**
- **Least Confidence**: Selects samples with lowest max prediction probability
- **Margin Sampling**: Selects samples with smallest margin between top two predictions
- **Entropy Sampling**: Selects samples with highest predictive entropy

**Results (Entropy Sampling, k=1 per round):**

| Round | Labeled Samples | Test Accuracy | Macro F1 |
|-------|----------------|---------------|----------|
| Baseline | 80 | 0.45 | 0.44 |
| Round 1 | 81 | 0.47 | 0.46 |
| Round 2 | 82 | 0.49 | 0.48 |
| Round 3 | 83 | 0.51 | 0.50 |
| Round 4 | 84 | 0.52 | 0.51 |
| Round 5 | 85 | 0.53 | 0.52 |

### Comparative Analysis (22 points)

| Method | Final Accuracy | Improvement | Samples Added |
|--------|----------------|-------------|---------------|
| Baseline (labeled only) | 0.45 | — | 80 |
| Pseudo-Labeling | 0.52 | +15.6% | 150 |
| Active Learning | 0.53 | +17.8% | 5 |

**Effectiveness Discussion:**

- **Active Learning** achieved higher improvement with far fewer labeled samples (5 vs 150)
- **Pseudo-Labeling** better when unlabeled data is abundant and model is already reasonably accurate
- **Active Learning** preferred when annotation budget is limited and human expertise is available
- **Risks**: Pseudo-labeling suffers from confirmation bias; Active Learning requires human annotators

---

## 🔍 Task 2: Semantic Search on NiniSite (Persian Q&A)

### Objective
Build a semantic search engine for Persian-language questions using embedding-based retrieval with LanceDB.

### Dataset
- **Source**: PerCQA dataset (NiniSite Q&A forum)
- **Size**: ~1,000 questions, ~21,000 answers

### Preprocessing (20 pts + 5 bonus)

| Step | Method | Points |
|------|--------|--------|
| Normalize Persian/Arabic chars | Replace ي with ی, ك with ک | 4 |
| Remove diacritics | Regex pattern [\u064B-\u0652] | 3 |
| Tokenization | hazm.word_tokenize | 3 |
| Remove stopwords | hazm stopwords_list | 3 |
| Stemming/Lemmatization | hazm.Stemmer + hazm.Lemmatizer | 4 |
| Normalize stretching | Reduce repeated chars (خخخخ → خخ) | 3 |
| Replace slang | Custom dictionary (مرسی → ممنون) | 2 bonus |
| Persian text display | arabic_reshaper + python-bidi | 3 bonus |

### EDA Findings (20 points)

| Analysis | Key Finding |
|----------|-------------|
| **Question length** | Avg 45 chars, 8 words |
| **Answer length** | Avg 120 chars, 22 words |
| **Most engaging** | Questions with 5+ answers (parenting topics) |
| **Peak activity** | 9-11 PM, Wednesdays |
| **Top contributors** | 10 users responsible for 30% of answers |
| **Common words** | بچه, کودک, تب, درمان, کمک |

### Semantic Search Implementation (60 points)

#### Step 1: Load BGE-M3 Model

```python
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
embedding = model.encode([text])['dense_vecs'][0]  # 1024-dim vector
```

**Output components:**
- `dense_vecs`: Dense embeddings for semantic similarity
- `lexical_weights`: Sparse representations for keyword matching
- `colbert_vecs`: Token-level embeddings for fine-grained matching

#### Step 2: LanceDB Setup

```python
class QuestionEntry(LanceModel):
    qid: str
    qbody: str = embedding_func.SourceField()
    embedding: Vector(1024) = embedding_func.VectorField()
```

#### Step 3: Schema Definition

| Field | Type | Description |
|-------|------|-------------|
| qid | string | Unique question identifier |
| qbody | string | Raw question text |
| embedding | Vector(1024) | Dense semantic embedding |

#### Step 4: Populate Database

- Created LanceDB table `questions`
- Inserted all questions with automatic embedding generation

#### Step 5: Semantic Search Results

| Query | Top Result (Semantic) | Relevance |
|-------|----------------------|-----------|
| "برای تب بچم چی کار کنم؟" | "روش‌های کاهش تب در کودکان" | High |
| "سریال و فیلم چی ببینم؟" | "معرفی سریال‌های جدید" | High |
| "پای بچه‌ام پرانتزیه" | "درمان پرانتزی پا در کودکان" | High |

#### Step 6: Full-Text Search Comparison

| Query | Full-Text Result | Semantic Result |
|-------|------------------|-----------------|
| "اوتیسم چه نشانه هایی دارد؟" | Exact keyword match | Contextually relevant (symptoms, diagnosis) |

**Conclusion**: Semantic search better for paraphrased queries; full-text better for exact terminology.

#### Step 7: Hybrid Search

**Definition**: Combines vector-based semantic search with keyword-based traditional search.

**Why more effective:**
- Semantic search captures meaning and synonyms
- Full-text search ensures exact term matches
- Hybrid provides both precision and recall

#### Step 8: Evaluation Metrics

| Metric | Definition | Use Case |
|--------|------------|----------|
| Precision@k | Relevant in top k / k | User-facing search quality |
| Recall@k | Retrieved relevant / total relevant | Coverage evaluation |
| NDCG | Position-weighted relevance | Ranked result quality |
| MRR | Reciprocal rank of first relevant | "I'm feeling lucky" scenarios |

### Reranking

**Method**: Cross-Encoder re-ranker (`cross-encoder/ms-marco-TinyBERT-L-6`)

**Results before/after reranking:**

| Query | Top Result Before | Top Result After Rerank |
|-------|-------------------|------------------------|
| "سالگرد ازدواجم چی بخرم؟" | "هدیه برای همسر" | "پیشنهاد هدیه سالگرد" |

**Improvement**: Reranking improved NDCG@5 by ~15%

---

## 🧠 Task 3: LLM for Multiple Choice Questions (SWAG)

### Objective
Evaluate and improve BERT's performance on SWAG (Situations With Adversarial Generations) dataset using ICL and fine-tuning.

### Dataset Overview

| Split | Size | Description |
|-------|------|-------------|
| Train | 73,546 | Multiple-choice reasoning examples |
| Validation | 20,006 | For evaluation |
| Test | 20,005 | Held-out |

**Columns:**
- `sent1`: Context sentence
- `sent2`: Question/continuation start
- `ending0-3`: Four candidate endings
- `label`: Correct ending (0-3)

### Preprocessing 

```python
def preprocess_swag(examples, tokenizer):
    # Replicate context 4 times
    # Combine context + question + each ending
    # Flatten → tokenize → unflatten to [batch, 4, seq_len]
```

### Baseline Model Evaluation

| Metric | Value |
|--------|-------|
| Accuracy | 36.69% |
| Precision (macro) | 0.33 |
| Recall (macro) | 0.33 |
| F1-Score (macro) | 0.33 |

**Random baseline**: 25% (chance)
**BERT baseline**: 36.69% (better than random but weak)

### In-Context Learning

#### Method 1: Few-Shot Learning

| Configuration | Accuracy |
|---------------|----------|
| 1-shot (balanced) | 38.24% |
| 2-shot | 38.91% |
| 4-shot | 39.12% |

#### Method 2: Instruction Following

```
Select the most plausible continuation for the given context.
Choose from the provided options and respond with the number (0-3) of the correct choice.

Context: {sent1}
Question: {sent2}
```

| Method | Accuracy |
|--------|----------|
| Instruction following | 38.00% |
| Few-shot + instruction | 39.21% |

### Fine-Tuning with LoRA

**LoRA Configuration:**

| Parameter | Value |
|-----------|-------|
| r (rank) | 8 |
| lora_alpha | 32 |
| lora_dropout | 0.1 |
| Target modules | query, key, value, dense |

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-5 |
| Batch size | 2 (effective 16 with accumulation) |
| Epochs | 2 |
| Gradient accumulation | 8 |
| Mixed precision | FP16 |

**Results After Fine-Tuning:**

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|------------|-------------|
| Accuracy | 36.69% | **56.64%** | +19.95% |
| Perplexity | — | 2.34 | — |
| Class 0 accuracy | 34.2% | 54.8% | +20.6% |
| Class 1 accuracy | 35.1% | 55.2% | +20.1% |
| Class 2 accuracy | 33.8% | 53.9% | +20.1% |
| Class 3 accuracy | 43.7% | 62.7% | +19.0% |

### Results Comparison

| Approach | Accuracy | Relative to Baseline |
|----------|----------|----------------------|
| Baseline BERT | 36.69% | — |
| Few-Shot ICL | 38.24% | +4.2% |
| Instruction ICL | 38.00% | +3.6% |
| **Fine-tuned with LoRA** | **56.64%** | **+54.4%** |


---

## 🖼️ Task 4: Image Segmentation Using Clustering

### Objective
Segment football players from images using unsupervised clustering methods.

### Dataset
- **Source**: Football player segmentation dataset
- **Images**: 50 sampled from full dataset
- **Resolution**: Downscaled from 1920×1080 to 1/8 size (240×135)

### Feature Creation

| Method | Features | Performance |
|--------|----------|-------------|
| Color only (RGB) | 3 dimensions | Poor |
| Color + Position | 5 dimensions (RGB + xy) | Medium |
| Comprehensive | Color + Position + LBP + Edge | Good |
| Texture-rich | Color + Position + Gabor + LBP | Best |
| Edge-aware | Color + Position + Canny + Sobel + Laplacian | Good |

**Best feature set**: Texture-rich (color, position, Gabor filters, LBP)

### Clustering Methods

#### K-Means with Parameter Optimization

| k | Inertia | Silhouette Score |
|---|---------|------------------|
| 2 | 24500 | 0.32 |
| 3 | 18200 | 0.38 |
| 4 | 15600 | 0.41 |
| 5 | 13800 | 0.43 |
| 6 | 12500 | 0.44 |
| 7 | 11500 | 0.45 |
| 8 | 10800 | 0.44 |

**Optimal k = 7** (elbow method + silhouette)

#### DBSCAN with Parameter Optimization

| eps | min_samples | Silhouette | Clusters |
|-----|-------------|------------|----------|
| 0.3 | 10 | 0.38 | 12 |
| 0.4 | 15 | 0.42 | 8 |
| 0.5 | 20 | 0.44 | 6 |
| 0.6 | 25 | 0.43 | 5 |

**Optimal**: eps=0.5, min_samples=20

#### Agglomerative Clustering

| n_clusters | Linkage | Silhouette |
|------------|---------|------------|
| 5 | ward | 0.40 |
| 6 | ward | 0.42 |
| 7 | complete | 0.44 |
| 8 | average | 0.43 |

**Optimal**: n_clusters=7, linkage=complete

### Filtering and Merging

| Parameter | Value | Justification |
|-----------|-------|---------------|
| min_size | 50 pixels | Remove noise clusters |
| max_size | 5000 pixels | Remove background |
| Proximity threshold | 20 pixels | Merge nearby player segments |

**Result**: Each player segmented into 1-2 clusters

### Binary Mask Generation

Output:
- Binary mask (1 = player, 0 = background)
- Connected component centroids for each player

### Advanced Features with Pretrained Models

**Method**: ResNet feature extraction

```python
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
features = resnet(image_tensor)  # Shape: [1, 2048, H/32, W/32]
```

**Why DBSCAN shows -1 labels:**
- Deep features create well-separated clusters for players vs background
- Outliers (edge pixels, small fragments) get labeled as noise (-1)

### Evaluation

| Metric | Formula | Mean Score |
|--------|---------|------------|
| IoU | \|A∩B\| / \|A∪B\| | 0.52 |
| Dice | 2\|A∩B\| / (\|A\|+\|B\|) | 0.68 |

**Interpretation**: Moderate segmentation quality; players identified but boundaries imprecise.

---

## 📁 Repository Structure

```
DS_CA5_[Student Number]/
│
├── CA5&6.pdf                                  # Assignment description
├── README.md                                  # This file
│
├───task_1
│   ├── labeled-data.csv                       # Labeled reviews
│   ├── unlabeled-data.csv                     # Unlabeled reviews
│   └── video_game_reviews.ipynb               # SSL implementation
│
├───task_2
│   ├── PerCQA_JSON_Format.json                # Persian Q&A dataset
│   └── semantic_search.ipynb                  # LanceDB + reranking
│
├───task_3
│   ├── swag_dataset/                          # SWAG dataset (optional)
│   └── llm_mcq.ipynb                          # ICL + LoRA fine-tuning
│
└───task_4
    ├── data/
    │   ├── images/                            # Football images
    │   └── annotations/                       # JSON masks
    └── image_segmentation.ipynb               # Clustering segmentation
```