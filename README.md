> Predicting stunting, wasting, and underweight prevalence across Indian districts using NFHS-5 survey data and ensemble machine learning models.

#  Prediction of Malnutrition in Children Under Five Years of Age Using Machine Learning

##  Overview

Child malnutrition remains one of India's most critical public health challenges. This project applies supervised machine learning to **predict the percentage of children under 5 years** who are stunted, wasted, or underweight at the district level — using socioeconomic, maternal health, and household environment indicators from India's **National Family Health Survey (NFHS-5)**.

The goal is to identify which districts are at highest risk and which upstream factors (literacy, sanitation, maternal BMI, etc.) are the strongest predictors of malnutrition — insights that can inform policy and resource allocation.

---

## 📊 Dataset

| Property             | Detail                                   |
| --------------------- | ----------------------------------------- |
| **Source**            | NFHS-5 (2019–21), Government of India     |
| **Unit of analysis**  | District-level                            |
| **Total records**     | 705 districts (after dropping duplicates/NaNs) |
| **Raw feature columns** | 93 (≥50% non-null, cleaned from the original survey export) |
| **Final feature matrix** | 108 columns (raw features + state target-encoding + composite scores + interaction terms + log transforms) |
| **Target variables**  | 3 (stunting %, wasting %, underweight %)  |

### Target Variables

- `Children under 5 years who are stunted (height-for-age)` — chronic malnutrition
- `Children under 5 years who are wasted (weight-for-height)` — acute malnutrition
- `Children under 5 years who are underweight (weight-for-age)` — composite indicator

### Feature Categories

- **Child health & diet** — breastfeeding rates, vaccination coverage, anaemia prevalence, diarrhoea
- **Maternal health** — literacy, antenatal/postnatal care, BMI, anaemia
- **Household environment** — electricity, sanitation, clean fuel, drinking water, health insurance
- **Engineered features** — out-of-fold target-encoded state, composite domain scores (maternal health, household infra, child diet, maternal metabolic), interaction terms, and log transforms

---

## 🔬 Methodology

### 1. Data Cleaning & Preprocessing

- Replaced dirty values (non-numeric characters, stray symbols) and coerced columns to numeric
- Kept feature columns with ≥50% non-null values, dropped the rest
- Median imputation for remaining missing values; rows with residual NaNs after imputation dropped
- Removed duplicate rows

### 2. Feature Engineering

- Excluded identifiers, survey metadata, and sibling nutrition outcomes to avoid data leakage (10 columns explicitly excluded)
- Out-of-fold (5-fold) target encoding for `State/UT` to avoid leakage
- Built composite domain scores (e.g. maternal health score, household infrastructure score, child diet score, maternal metabolic score) from weighted combinations of related indicators
- Added interaction terms and log transforms
- Final feature matrix: 108 columns
- Mutual-information feature selection (fit on train split only) used to rank top predictors

### 3. Models Trained

#### Regression (predicting raw % values)

| Model                     | Notes |
| -------------------------- | ----- |
| Elastic Net (ElasticNetCV) | Regularised linear baseline |
| Random Forest              | 600 trees, `max_features='sqrt'` |
| XGBoost                    | RandomizedSearchCV hyperparameter tuning |
| LightGBM                   | RandomizedSearchCV hyperparameter tuning |
| Extra Trees                | 600 trees |
| **Weighted Blend**         | LightGBM × 0.40 + Extra Trees × 0.35 + XGBoost × 0.25 (weights set by relative CV R²) |

#### Classification (WHO threshold-based severity categories)

Class imbalance handled per-target with SMOTE variants (SMOTEENN, SMOTETomek), Balanced Random Forest, LightGBM, and XGBoost, evaluated with 5-fold stratified CV; the best-performing model per target was selected on macro F1.

### 4. Regression Results (held-out test set, 20% split)

| Target             | Best model      | R²    | RMSE (pct pts) | MAE  | MAPE   |
| ------------------- | ---------------- | ----- | ---------------- | ---- | ------ |
| Stunting (HAZ)      | Weighted Blend    | 0.584 | 5.56             | 4.32 | 14.2%  |
| Wasting (WHZ)       | XGBoost           | 0.408 | 4.49             | 3.64 | 22.4%  |
| Underweight (WAZ)   | LightGBM          | 0.725 | 4.76             | 3.80 | 14.8%  |

Full per-model comparison (R², RMSE, MAE, MAPE for all 6 regressors × 3 targets) is in the notebook.

### 5. Classification Results (WHO severity categories, held-out test set)

| Target             | Best model         | Accuracy | Macro F1 | ROC-AUC (OVR) |
| ------------------- | -------------------- | -------- | -------- | --------------- |
| Stunting (HAZ)      | SMOTETomek + RF       | 58.2%    | 0.501    | 0.808            |
| Wasting (WHZ)       | Balanced RF           | 63.8%    | 0.533    | 0.777            |
| Underweight (WAZ)   | XGBoost               | 71.6%    | 0.518    | 0.903            |

Classification accuracy looks high in places largely because the classes are imbalanced (e.g. "Severe Wasting" and "Very High Underweight" dominate); macro F1 — which weights all classes equally — is the more honest performance signal, and it's modest (0.50–0.53) across all three targets, reflecting the difficulty of separating adjacent severity tiers on this sample size (705 districts, ~141 in the test set).

---

##  Top Predictive Features

Based on mutual-information scores computed on the training split (summed across all three targets), the strongest predictors were:

1. **Women's BMI below normal** — strongest single predictor by MI score
2. **Maternal metabolic score** (engineered composite)
3. **Metabolic × infrastructure interaction** (engineered feature)
4. **Women's literacy rate**
5. **Women who are overweight or obese (BMI ≥ 25)**

Maternal nutritional status (both under- and over-nutrition) and literacy consistently rank as the dominant upstream signals across all three malnutrition indicators.

---

##  Project Structure

```
📦 malnutrition-prediction
 ┣ 📓 Malnutrition_Analysis_NFHS5.ipynb   # Main notebook (data cleaning → modelling → SHAP)
 ┣ 📄 datafile.csv                        # NFHS-5 district-level dataset
 ┣ 📄 requirements.txt
 ┗ 📄 README.md
```

---

## ⚙️ Requirements

```
pip install -r requirements.txt
```

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
```

---

##  How to Run

1. Clone the repository

```
git clone https://github.com/M1004G/Prediction-of-Malnutrition-in-Children-Under-Five-Years-of-Age-using-Machine-Learning.git
cd Prediction-of-Malnutrition-in-Children-Under-Five-Years-of-Age-using-Machine-Learning
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Open the notebook

```
jupyter notebook Malnutrition_Analysis_NFHS5.ipynb
```

4. Upload `datafile.csv` to `/content/` if running on Google Colab, or update `DATA_PATH` in the config cell for local runs.

---

##  Key Findings

- **Underweight** was the most predictable target (R² = 0.725), likely because it integrates both chronic and acute malnutrition signals
- **Wasting** was the hardest target to model (R² ≈ 0.41), consistent with its episodic, illness-driven nature that's less captured by slow-moving socioeconomic indicators
- The **weighted blend ensemble** outperformed any single regressor on stunting, but individual tuned models (XGBoost/LightGBM) edged it out on wasting and underweight — ensembling didn't help uniformly across targets
- WHO-severity **classification** is workable for coarse risk stratification (macro F1 ~0.50–0.53, ROC-AUC 0.78–0.90) but not reliable enough yet for fine-grained tier assignment, especially for minority classes with few districts
- **Maternal BMI and literacy** are consistently the strongest upstream predictors across all three targets

---

##  Limitations

- Dataset is cross-sectional (NFHS-5, 2019–21) — causal inference is not possible
- District-level aggregates mask intra-district variation
- Regression R² (0.41–0.73) and classification macro F1 (0.50–0.53) indicate meaningful but incomplete predictive power — useful for directional risk flagging, not precise estimation
- Small test set (141 districts) and class imbalance in the classification task make minority-class metrics noisy
- Model performance is bounded by the quality and granularity of NFHS survey data

---

##  Data Source

> **National Family Health Survey-5 (NFHS-5), 2019–21**
> Ministry of Health and Family Welfare, Government of India
> <http://rchiips.org/nfhs/nfhs5.shtml>

---

##  Acknowledgements

This project was developed as part of an academic machine learning course. The NFHS-5 data is publicly available and collected by the International Institute for Population Sciences (IIPS), Mumbai.
