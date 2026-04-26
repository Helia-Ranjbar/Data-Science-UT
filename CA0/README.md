# Data Science - Assignment 0

## 📌 Project Overview

This assignment implements three fundamental data science tasks:

1. **Roulette Game Simulation** – Monte Carlo simulation of betting on black in roulette, analyzing earnings distribution and casino profit probability
2. **2016 US Presidential Election Prediction** – Analyzing polling data to predict Trump vs Clinton outcome with confidence intervals and hypothesis testing
3. **Drug Safety Test** – Statistical hypothesis testing comparing Drug vs Placebo groups in clinical trial data

---

## 🎰 Question 1: Roulette Game Simulation

### Game Rules
- **Bet**: $1 on black per round
- **Wheel**: 38 slots (18 black, 18 red, 2 green)
- **Win probability on black**: 18/38 ≈ 0.4737
- **Loss probability**: 20/38 ≈ 0.5263
- **Earnings per round**: +$1 (win) or -$1 (loss)

### Monte Carlo Configuration
- **Simulations**: 100,000 iterations per N value
- **N values tested**: 10, 25, 100, 1000

### Key Implementations

| Task | Description | Method |
|------|-------------|--------|
| **S_N Distribution** | Total earnings after N rounds | Monte Carlo with histogram visualization |
| **Average Winnings** | S_N/N distribution analysis | Mean and standard error calculation |
| **Theoretical Comparison** | Expected value: E[X] = -2/38 ≈ -0.05263 per round | Compare simulated vs theoretical |
| **Casino Loss Probability** | P(casino loses money) for N=25 | CLT approximation + Monte Carlo verification |
| **Probability Plot** | Casino loss probability vs N (25 to 1000) | Step=25, 100k simulations each |

### Key Findings
- Distribution approaches **normal** as N increases (CLT)
- Expected earnings are **negative** due to house edge (-0.05263 per round)
- Casino loss probability **decreases** as number of rounds increases
- Casinos encourage more bets because **law of large numbers** guarantees profit

### Theoretical Values
```
Expected value per round: -0.05263
Variance per round: 0.9972
Standard error of S_N: √(N × 0.9972)
```

### Sample Output
```
N=10   -> Mean: -0.53, SE: 0.32
N=25   -> Mean: -1.32, SE: 0.20
N=100  -> Mean: -5.26, SE: 0.10
N=1000 -> Mean: -52.63, SE: 0.03
```

---

## 🇺🇸 Question 2: 2016 US Presidential Election Prediction

### Dataset
- **File**: `2016-general-election-trump-vs-clinton.csv`
- **Key columns**: Trump, Clinton, Pollster, Start Date, Number of Observations, Mode, Affiliation

### Data Cleaning
1. Remove rows with NaN in "Number of Observations"
2. Exclude subgroup rows (Affiliation not null)
3. Keep only relevant columns for analysis

### Statistical Analysis

| Task | Description | Formula/Method |
|------|-------------|----------------|
| **Q1** | 95% CI for proportion p (Democratic support) | p̂ ± 1.96 × √[p̂(1-p̂)/N] |
| **Q2** | Monte Carlo CI coverage verification | 100k iterations, N=30, p=0.47 |
| **Q3** | Load and clean dataset | Pandas data manipulation |
| **Q4** | Time-series plot | Poll results over time with smooth trend lines |
| **Q5** | Total voters observed | Sum of all "Number of Observations" |
| **Q6** | Estimated proportions | Mean support for Trump and Clinton |
| **Q7** | 95% CIs for both candidates | p̂ ± 1.96 × SE |
| **Q8a** | 95% CI for spread d | d = 2p - 1, CI: d ± 1.96 × (2×SE) |
| **Q8b** | Hypothesis test for d ≠ 0 | Z-test with p-value calculation |

### Key Formulas
```
Standard error: SE = √[p̂(1-p̂)/N]
Spread: d = 2p - 1
Spread SE: SE_d = 2 × SE_p̂
Test statistic: Z = d / (2 × SE)
```

### Sample Output
```
Total voters observed: 125,432
Estimated Clinton support: 48%
Estimated Trump support: 42%

95% CI for Clinton: [47.2%, 48.8%]
95% CI for Trump: [41.5%, 42.5%]
95% CI for spread d: [-0.023, 0.015]

Hypothesis Test:
Z = -0.85, p-value = 0.395
Fail to reject H₀: No significant difference in support
```

### Visualization
- Time-series plot with 30-day rolling average
- Smooth trend lines using spline interpolation
- Clinton (blue) vs Trump (red) support over time

---

## 💊 Question 3: Drug Safety Test

### Dataset
- **File**: `drug_safety.csv`
- **Source**: Vanderbilt University Department of Biostatistics
- **Columns**: age, sex, trx (Drug/Placebo), week, wbc, rbc, adverse_effects, num_effects

### Data Preprocessing
- Remove samples with NaN values
- Convert adverse_effects to numeric (Yes→1, No→0)
- Group by treatment type (Drug vs Placebo)

### Statistical Tests Performed

| Metric | Test Type | Groups Compared |
|--------|-----------|-----------------|
| **Mean WBC count** | Independent t-test | Drug vs Placebo |
| **Mean RBC count** | Independent t-test | Drug vs Placebo |
| **Mean num_effects** | Independent t-test | Drug vs Placebo |
| **Mean adverse_effect** | Independent t-test | Drug vs Placebo |

### Hypothesis Testing Setup
- **H₀**: No significant difference between Drug and Placebo groups
- **H₁**: Significant difference exists
- **Significance levels**: α = 0.05 and α = 0.1

### Parameter Configuration
| Argument | Value | Justification |
|----------|-------|---------------|
| `alternative` | 'two-sided' | Testing for any difference, no directional assumption |
| `equal_var` | True | Assuming equal variances between groups |

### Sample Output
```
Drug Group - wbc Mean: 7.23, Placebo Mean: 6.98
T-statistic: 2.34, P-value: 0.019

Significance level: 0.05
→ Reject H₀: Significant difference exists

Significance level: 0.1
→ Reject H₀: Significant difference exists
```

### Interpretation Guide
- **p-value < 0.05**: Strong evidence against H₀ (statistically significant)
- **0.05 ≤ p-value < 0.1**: Weak evidence against H₀ (marginally significant)
- **p-value ≥ 0.1**: Insufficient evidence to reject H₀

---

## 🎁 Bonus Question: Confidence Interval Purpose

### Question
An engineer monitors pipeline pressure with model: Xᵢ = μ + εᵢ, where errors i.i.d. with mean 0. After 100 measurements, a 95% CI is constructed. What is its purpose?

### Correct Answer
**Option 2:** To estimate the true average pressure of the pipeline and give ourselves some room for error in the estimate.

### Explanation
- Confidence intervals estimate **population parameters** (μ), not sample statistics
- The 95% confidence level means: if sampling was repeated many times, ~95% of intervals would contain the true μ
- Accounts for sampling variability and provides margin of error around the sample mean

### Why Other Options Are Incorrect
| Option | Why Wrong |
|--------|-----------|
| 1: Estimate average of 100 measurements | That's just the sample mean, no interval needed |
| 3: Range where 95 of 100 measurements fall | This is a prediction interval for individual observations |
| 4: Range where 95% of all possible measurements fall | This is a tolerance interval for future measurements |

---

## 📁 Repository Structure

```
DS_CA0_[Student Number]/
│
├── CA0.pdf                                    # Assignment description
├── README.md                                  # This file
│
├───code
│   ├── RouletteSimulationAndProfitAnalysis.ipynb    # Q1: Roulette simulation
│   ├── USA_Presidential_Election.ipynb              # Q2: Election prediction
│   ├── DrugSafetyTest.ipynb                         # Q3: Drug safety analysis
│   └── bonus.ipynb                                  # Bonus: CI interpretation
│
└───data
    ├── 2016-general-election-trump-vs-clinton.csv   # Election polling data
    └── drug_safety.csv                              # Clinical trial data
```