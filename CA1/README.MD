# Data Science - Assignment 1

## 📌 Project Overview

This assignment implements two main tasks:

1. **Langevin Dynamics Sampling** – Implementing a stochastic sampling algorithm for 2D Gaussian distributions, analyzing score functions, and comparing sampling methods
2. **Tableau Dashboard Analysis** – Creating interactive dashboards for Airbnb data analysis with multiple visualizations and storytelling

---

## 🧠 Task 1: Sampling using Langevin Dynamics

### Objective
Implement Langevin Dynamics, a sampling algorithm that generates samples from complex distributions even when the normalization constant is unknown.

### Key Concepts

#### Score Function
For a probability distribution with PDF \( p(x) \), the score function is:
\[
S(x) = \nabla_x \log p(x)
\]

For a multivariate Gaussian distribution:
\[
S(x) = -\Sigma^{-1}(x - \mu)
\]

#### Langevin Dynamics Update Rule
\[
X_{t+1} \leftarrow X_t + \epsilon S(X_t) + \sqrt{2\epsilon} Z_t
\]

Where:
- \( \epsilon \): Step size hyperparameter
- \( S(X_t) \): Score function
- \( Z_t \sim \mathcal{N}(0, I) \): Gaussian noise

### Implementation Steps

| Step | Description | Method |
|------|-------------|--------|
| **1** | Create 2D Gaussian distribution | Mean = [-5, 5], Cov = 5I |
| **2** | Plot Gaussian heatmap | 500×500 grid visualization |
| **3** | Derive and visualize score function | Quiver plot showing gradient vectors |
| **4** | Implement Langevin Dynamics | 1000 steps, epsilon = 0.05 |
| **5** | Plot sampling trajectories | Single sample path visualization |
| **6** | Generate 1000 samples | Compare with direct sampling |
| **7** | Compare sampling methods | Scatter, density, marginal, joint, timing |

### Gaussian Distribution Parameters
```
Mean (μ) = [-5, 5]
Covariance (Σ) = 5I = [[5, 0], [0, 5]]
```

### Score Function Derivation

For multivariate Gaussian:
\[
\nabla_x \log p(x) = -\Sigma^{-1}(x - \mu)
\]

With Σ = 5I, Σ⁻¹ = (1/5)I:
\[
S(x) = -\frac{1}{5}(x - [-5, 5]) = -\frac{1}{5}(x_1 + 5, x_2 - 5)
\]

### Visualizations Produced

1. **Gaussian Heatmap** – 2D probability density with mean marker
2. **Score Field Quiver Plot** – Gradient vectors pointing toward high-density regions
3. **Langevin Trajectories** – Single sample path over 1000 iterations
4. **Sample Comparison** – Direct vs Langevin (scatter, density, marginal, joint)

### Sampling Comparison Results

| Evaluation Method | Finding |
|-------------------|---------|
| **Scatter Plot** | Both methods show dense clustering near mean (-5,5) |
| **Density Heatmap** | Langevin maintains smooth density even in low-probability regions |
| **Marginal Distributions** | Strong overlap in histograms and KDE lines |
| **Joint Distribution** | Both display elliptical clustering consistent with Gaussian |
| **Timing** | Direct sampling faster; Langevin ~X slower |

### Computational Performance
```
Direct Sampling Time: 0.000X seconds
Langevin Dynamics Time: 0.00X seconds
```

---

## 📊 Task 2: Tableau Dashboard - Airbnb Analysis

### Objective
Create interactive Tableau dashboards to analyze Airbnb data, building a structured data story through multiple visualizations.

### Datasets

#### 1. Airbnb_Listings.xlsx
| Column | Description |
|--------|-------------|
| Neighbourhood | Specific area of listing |
| Price | Cost per night |
| Availability 365 | Days available per year |
| Calculated Host Listings Count | Total listings per host |
| Host Id / Host Name | Host identifiers |
| Last Review | Date of most recent review |
| Minimum Nights | Minimum booking nights |
| Number Of Reviews | Total reviews received |
| Reviews Per Month | Average monthly reviews |
| Room Type | Entire home/apt, Private room, Shared room |

#### 2. Neighborhood_Locations.xlsx
| Column | Description |
|--------|-------------|
| Neighbourhood | Specific area |
| Neighbourhood Group | Broader classification |
| Latitude / Longitude | Geographic coordinates |

### Data Preparation Steps
1. Import both datasets into Tableau
2. Join datasets using neighbourhood information
3. Ensure each location has longitude and latitude
4. Handle null values as needed

### Dashboard Structure

#### Dashboard 1: Introduction & Market Overview
- **Focus**: Define key analysis objectives
- **KPIs**: Total listings, average price, total hosts
- **Visualizations**: Geographic price distribution map

#### Dashboard 2: Pricing Trends
- **Focus**: Price patterns across neighbourhoods
- **Filters**: Room type, neighbourhood group
- **Visualizations**: Price heatmaps, bar charts

#### Dashboard 3: Host Activities
- **Focus**: Host behavior and listing patterns
- **KPIs**: Average listings per host, top hosts
- **Visualizations**: Host distribution charts

#### Dashboard 4: Review Insights
- **Focus**: Customer engagement metrics
- **Filters**: Time range, neighbourhood
- **Visualizations**: Review trends over time

### Interactive Elements
- **Filters**: Room type, neighbourhood, price range
- **Parameters**: Customizable thresholds
- **KPIs**: Key performance metrics highlighted

### Design Principles Applied
- **Gestalt Principles**: Structured, readable layouts
- **Preattentive Attributes**: Color, size, position for key insights
- **Consistent Formatting**: Unified color schemes and fonts

### Storytelling Structure
1. **Introduction**: Define key focus of analysis
2. **Findings**: Insights across multiple dashboards
   - Seasonal trends
   - Geographic price distribution
   - Host behavior patterns
   - Review engagement
3. **Conclusion**: Summarize main findings and patterns

---

## 📁 Repository Structure

```
DS_CA1_[Student Number]/
│
├── CA1.pdf                                    # Assignment description
├── README.md                                  # This file
│
├───task_1
│   ├── task1_sampling.ipynb                   # Langevin Dynamics implementation
│   └── task1_sampling.html                    # Exported HTML version
│
└───task_2
    ├── task2-Airbnb-Tableau.pdf               # Tableau dashboard documentation
    ├── Airbnb_Listings.xlsx                   # Main listings dataset
    └── Neighborhood_Locations.xlsx            # Geographic coordinates data
```

---

## 🛠️ Requirements & Setup

### Python Libraries for Task 1
```python
numpy          # Numerical computations
matplotlib     # Plotting and visualization
seaborn        # Statistical visualizations
scipy          # Multivariate normal distribution
pandas         # Data manipulation
time           # Performance timing
```

### Installation
```bash
pip install numpy matplotlib seaborn scipy pandas
```

### Tableau for Task 2
- Tableau Public (free) or Tableau Desktop
- Download from: https://public.tableau.com/

### Running Task 1 Code
```bash
jupyter notebook task_1/task1_sampling.ipynb
```

---

## 📊 Key Results Summary

| Task | Key Finding |
|------|-------------|
| **Score Function** | Points toward mean (-5,5) with magnitude proportional to distance |
| **Langevin Dynamics** | Successfully generates Gaussian samples without normalization constant |
| **Method Comparison** | Langevin matches direct sampling but slower |
| **GMM Sampling** | Possible but requires annealing or multiple chains |
| **Airbnb Analysis** | Interactive dashboards with geographic and pricing insights |

---

## 🔍 Theoretical Questions (Bonus)

### Q: Can Langevin Dynamics sample from a Gaussian Mixture Model?

**Answer**: Yes, with conditions.

#### Challenges:
1. **Mode Collapse**: Sampler may get stuck in one mode
2. **High-Curvature Regions**: Score becomes large between modes

#### Solutions:
1. **Small step size (ε)**: Prevents overshooting
2. **Annealed Langevin**: Gradually reduce ε
3. **Parallel chains**: Multiple independent sampling processes

#### Formula for GMM Score:
\[
p(x) = \alpha \mathcal{N}(x;\mu_1,\Sigma_1) + (1-\alpha)\mathcal{N}(x;\mu_2,\Sigma_2)
\]

\[
\nabla_x \log p(x) = \frac{\alpha \nabla_x \mathcal{N}(x;\mu_1,\Sigma_1) + (1-\alpha) \nabla_x \mathcal{N}(x;\mu_2,\Sigma_2)}{\alpha \mathcal{N}(x;\mu_1,\Sigma_1) + (1-\alpha) \mathcal{N}(x;\mu_2,\Sigma_2)}
\]

#### Conditions for Success:
- Modes not too far apart
- Carefully chosen step size
- Balanced mixture weights (α not near 0 or 1)

