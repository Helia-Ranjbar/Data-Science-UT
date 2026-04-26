# Data Science - Assignment 4

## 📌 Project Overview

This assignment implements three deep learning projects:

1. **Football Match Prediction (MLP)** – Predicting World Cup match outcomes using a Multi-Layer Perceptron
2. **Flower Classification (CNN)** – Classifying flower species using VGG-style CNN and fine-tuned ResNet
3. **Bitcoin Price Prediction (RNN)** – Forecasting Bitcoin price movements using RNN and LSTM architectures

---

## ⚽ Task 1: Football Match Prediction (MLP)

### Objective
Predict football match outcomes (home win, tie, away win) using team performance statistics from World Cup qualifiers.

### Data Preparation Steps

| Step | Method |
|------|--------|
| **Filtering** | Train on qualifier matches (wcm=0), test on World Cup matches |
| **Feature Exclusion** | Remove home_team, away_team, home_goals, away_goals, status |
| **Label Encoding** | sklearn LabelEncoder for status |
| **Train-Test Split** | 70% train, 30% test, stratified |
| **Standardization** | StandardScaler (fit only on training data) |
| **Tensor Conversion** | PyTorch FloatTensor for features, LongTensor for labels |

### Model Architecture (MLP)

```python
FootballPredictor(
    input_dim = 12 features
    hidden_dims = [64, 32]
    output_dim = 3 classes
    
    Layers:
    - Linear(12 → 64) + ReLU + Dropout(0.3)
    - Linear(64 → 32) + ReLU + Dropout(0.3)
    - Linear(32 → 3)  # No softmax (CrossEntropyLoss handles it)
)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam (lr=0.001) |
| Batch Size | 32 |
| Epochs | 100 |
| Dropout | 0.3 |

### EDA Visualizations Generated
1. Feature distributions with skewness
2. Match outcome distribution (Home Win, Tie, Away Win)
3. Correlation matrix of features
4. World Cup vs Qualifier comparison
5. Q-Q plots for normality assessment

### Results

| Metric | Value |
|--------|-------|
| Test Accuracy | >50% (achieved full marks) |
| Final Test Accuracy | ~55-60% |

*Note: Football is unpredictable; accuracies above 90% are not expected*

### World Cup 2022 Prediction Output

The model simulated:
- **Group Stage**: All 48 group matches
- **Round of 16**: 8 matches
- **Quarter-finals**: 4 matches
- **Semi-finals**: 2 matches
- **Final**: 1 match

---

## 🌸 Task 2: Flower Classification (CNN)

### Objective
Classify flower species using VGG-style CNN from scratch and fine-tuned ResNet50.

### Dataset
- **Source**: Kaggle Flowers Multiclass Dataset
- **Classes**: Daisy, Dandelion, Roses, Sunflowers, Tulips (5 classes)
- **Split**: 70% train, 10% validation, 20% test

### Image Preprocessing

| Step | Specification |
|------|---------------|
| Resize | 224×224 pixels |
| Normalization | Pixel values scaled to [0,1] |
| Label Format | Categorical (one-hot encoding) |

### Data Augmentation (Training Only)

```python
data_augmentation = Sequential([
    RandomRotation(0.1),      # ±10% rotation
    RandomFlip("horizontal"), # 50% chance horizontal flip
    RandomZoom(0.1),          # ±10% zoom
    RandomContrast(0.2)       # ±20% contrast change
])
```

### Model 1: VGG-Style CNN (From Scratch)

**Architecture:**

| Layer | Configuration |
|-------|---------------|
| Conv2D | 16 filters, 3×3, ReLU |
| Conv2D | 16 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Conv2D | 32 filters, 3×3, ReLU |
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| Conv2D | 128 filters, 3×3, ReLU |
| Conv2D | 128 filters, 3×3, ReLU |
| MaxPooling | 2×2 |
| GlobalAveragePooling2D | |
| Dense | 64, ReLU, Dropout(0.5) |
| Dense | 5, Softmax |

### Model 2: Fine-Tuned ResNet50

**Stage 1: Frozen Base**
- Freeze all convolutional layers
- Train new classifier head only
- Learning rate: 1e-4
- Epochs: 10

**Stage 2: Unfreeze Last Conv Block**
- Unfreeze conv5_block layers
- Train with learning rate: 1e-5
- Epochs: 10

**Stage 3: Unfreeze All Layers**
- All layers trainable
- Learning rate: 1e-5
- Epochs: 10

### Evaluation Metrics

| Metric | Method |
|--------|--------|
| Accuracy | Classification accuracy |
| Precision | Macro average |
| Recall | Macro average |
| F1-Score | Macro average |
| Confusion Matrix | Seaborn heatmap |
| ROC Curves | One-vs-rest for each class |
| AUC | Area Under Curve |

### Results Comparison

| Model | Validation Accuracy |
|-------|---------------------|
| VGG from Scratch | ~75-80% |
| ResNet Stage 1 (Frozen) | ~80-85% |
| ResNet Stage 2 (Last block) | ~85-88% |
| ResNet Stage 3 (Full fine-tune) | ~88-92% |

---

## 💰 Task 3: Bitcoin Price Prediction (RNN)

### Objective
Forecast Bitcoin price movements using historical OHLCV data with RNN and LSTM architectures.

### Dataset
- **File**: BTC-USD.csv
- **Features**: Open, High, Low, Close, Volume
- **Time period**: Daily Bitcoin prices

### EDA Visualizations
1. Open price over time
2. High price over time
3. Low price over time
4. Close price over time
5. Volume over time
6. Correlation heatmap (OHLCV features)
7. Box plots for outlier detection

### Key EDA Insights
- Bitcoin experienced stability (2015-2017), bull run to $20K (late 2017)
- Explosive bull market to $70K (2021-2022)
- Volume spikes during major price movements
- High correlation between Open, High, Low, Close prices

### Target Definition
```python
# Target: 1 if price increases after 3 days, 0 otherwise
df['Target'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
df['Target'] = (df['Target'] > 0).astype(int)
```

### Data Processing

| Step | Method |
|------|--------|
| **Missing Values** | None found |
| **Outlier Handling** | Remove Volume > 99th percentile |
| **Normalization** | MinMaxScaler [0,1] |
| **Sequence Creation** | Lookback window of past days |
| **Lookback Tuning** | Tested 30, 60, 90 days |

### Sequence Creation
```python
def create_sequences(data, target, lookback):
    # Each sample: lookback days of features
    # Each label: target for the next day
```

### Train-Validation-Test Split
```
Original Data → 70% Train → 15% Validation → 15% Test
(Split chronologically with shuffle=False)
```

### Model 1: Simple RNN Architecture

```python
Sequential([
    SimpleRNN(512, return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),
    
    SimpleRNN(128, return_sequences=True),
    
    SimpleRNN(64, return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),
    
    SimpleRNN(32),
    Dropout(0.2),
    BatchNormalization(),
    
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid')  # Binary classification
])
```

### Model 2: LSTM Architecture

```python
Sequential([
    LSTM(512, return_sequences=True),
    LSTM(128, return_sequences=True),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    BatchNormalization(),
    LSTM(32),
    Dropout(0.2),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Regression output
])
```

### Training Configuration

| Parameter | RNN | LSTM |
|-----------|-----|------|
| Loss | Binary Crossentropy | MSE / Binary Crossentropy |
| Optimizer | Adam (lr=0.0001) | Adam (lr=0.001) |
| Batch Size | 16 | 16 |
| Early Stopping | patience=3 | patience=3 |
| Regularization | L2 (0.01) | Dropout + BatchNorm |

### Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| MSE | Mean( (y_true - y_pred)² ) | Penalizes large errors |
| RMSE | √MSE | Error in same units as target |
| MAE | Mean( |y_true - y_pred| ) | Average absolute error |
| MAPE | Mean( |error/y_true| ) × 100% | Percentage error |
| CE | Σ(y_true - y_pred) | Total prediction bias |

### Results Comparison

| Metric | Simple RNN | LSTM |
|--------|------------|------|
| Test Accuracy | ~0.55-0.60 | ~0.60-0.65 |
| MSE | ~0.25 | ~0.22 |
| RMSE | ~0.50 | ~0.47 |
| MAE | ~0.48 | ~0.45 |
| MAPE | ~95% | ~90% |

---

## 📁 Repository Structure

```
DS_CA4_[Student Number]/
│
├── DS-CA4.pdf                                 # Assignment description
├── README.md                                  # This file
│
├───task_1
│   ├── matches.csv                            # Football match data
│   └── MLP.ipynb                              # MLP implementation
│
├───task_2
│   ├── CNN.ipynb                              # CNN implementation
│   └── cnn_images/                            # Visualization outputs
│       ├── download.png
│       ├── download (1).png
│       ├── download (2).png
│       ├── download (3).png
│       └── download (4).png
│
└───task_3
    ├── BTC-USD.csv                            # Bitcoin price data
    ├── RNN.ipynb                              # RNN/LSTM implementation
    └── rnn_images/                            # Visualization outputs
        ├── newplot.png
        ├── newplot (1).png
        ├── newplot (2).png
        ├── newplot (3).png
        ├── newplot (4).png
        ├── newplot (5).png
        ├── newplot (6).png
        ├── newplot (7).png
        └── newplot (8).png
```