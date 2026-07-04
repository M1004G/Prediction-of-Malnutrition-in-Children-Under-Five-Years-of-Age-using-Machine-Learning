# Prediction of Malnutrition in Children Under Five Years of Age Using Machine Learning

> Predicting stunting, wasting, and underweight prevalence across Indian districts using NFHS-5 survey data and ensemble machine learning models.

---

## Overview

Child malnutrition remains one of India's most critical public health challenges. This project applies supervised machine learning to **predict the percentage of children under 5 years** who are stunted, wasted, or underweight at the district level — using socioeconomic, maternal health, and household environment indicators from India's **National Family Health Survey (NFHS-5)**.

The goal is to identify which districts are at highest risk and which upstream factors (maternal health, sanitation, anaemia, literacy, etc.) are the strongest predictors of malnutrition — insights that can inform policy and resource allocation.

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | NFHS-5 (2019–21), Government of India |
| **Unit of analysis** | District-level |
| **Total records** | 706 districts (705 after deduplication) |
| **Total features** | 93 numeric columns after cleaning, expanded to 108 after domain feature engineering (state target-encoding was tested and removed — see Methodology Correction note) |
| **Target variables** | 3 (stunting %, wasting %, underweight %) |

### Target Variables
- `Children under 5 years who are stunted (height-for-age)` — chronic malnutrition
- `Children under 5 years who are wasted (weight-for-height)` — acute malnutrition
- `Children under 5 years who are underweight (weight-for-age)` — composite indicator

### Feature Categories
- **Child health & diet** — breastfeeding, vaccination coverage, anaemia, diarrhoea
- **Maternal health** — literacy, antenatal care, BMI, anaemia, menstrual hygiene
- **Household environment** — electricity, sanitation, clean fuel, iodized salt

---

## Methodology

### 1. Data Cleaning & Preprocessing
- Replaced dirty values (`*`, `-`, `NA`, parenthesized numbers) with `NaN`
- Dropped columns with >50% missing values
- Median imputation for remaining missing values (no outlier capping — see Limitations)
- Removed 1 duplicate row

### 2. Feature Engineering
- Composite domain scores built from raw feature columns only (e.g. `maternal_health_score`, `household_infra_score`, `child_diet_score`) — no target information used
- State/UT target-encoding was tested but removed after SHAP analysis showed it leaning on train/test-split leakage for two of three targets (see Methodology Correction note)
- Mutual Information (MI) feature selection, **fit on training data only**, top 40 features used for the regularized ElasticNet baseline; tree-based models use all features

### 3. Train/Test Split
80/20 split — 564 training districts, 141 test districts.

### 4. Regression Results (Test Set)

| Model | Stunting (HAZ) R² | Wasting (WHZ) R² | Underweight (WAZ) R² |
|---|---|---|---|
| Elastic Net | 0.550 | 0.389 | 0.680 |
| Random Forest | 0.573 | 0.335 | 0.659 |
| XGBoost | 0.578 | 0.408 | 0.689 |
| LightGBM | 0.566 | 0.371 | 0.725 |
| Extra Trees | 0.561 | 0.353 | 0.664 |
| **Weighted Blend** (LGB 0.40 + ET 0.35 + XGB 0.25) | **0.584** | **0.399** | **0.707** |

RMSE for the blended model: 5.56 (stunting), 4.52 (wasting), 4.91 (underweight) — all in percentage points.

### 5. Classification Results (WHO Severity Tiers)

| Target | Best Model | Macro F1 | Accuracy | ROC-AUC (OVR) |
|---|---|---|---|---|
| Stunting (HAZ) | SMOTETomek + RF | 0.501 | 0.582 | 0.808 |
| Wasting (WHZ) | Balanced RF | 0.533 | 0.638 | 0.777 |
| Underweight (WAZ) | XGBoost | 0.518 | 0.716 | 0.903 |

Wasting and underweight classes are heavily imbalanced (e.g. "Severe Wasting" is 494/705 districts, "Very High Underweight" is 340/705); macro F1 is reported alongside accuracy and ROC-AUC since accuracy alone is misleading here. The smallest classes ("Low Stunting", 31 districts; "Mild Wasting", 60 districts) are also the hardest to classify — e.g. Stunting's "Low Stunting" class scores only 0.18 F1 despite the model's overall Macro F1 of 0.50, since there's very little data to learn that class from.

---

## Top Predictive Features

SHAP analysis (LightGBM model) per target, post state-encoding removal:

**Stunting (HAZ):**
1. `maternal_metabolic_score` (composite: BMI/obesity + menstrual hygiene)
2. Women with BMI below normal
3. Births that are third-or-higher order
4. `metabolic_x_infra` (interaction term)
5. `maternal_health_score` (literacy, ANC visits, postnatal care)

**Wasting (WHZ):**
1. `maternal_metabolic_score`
2. Women who are overweight/obese
3. Children who received 3 doses of rotavirus vaccine
4. Women with BMI below normal
5. Deaths in the last 3 years registered with civil authority

**Underweight (WAZ):**
1. Women with BMI below normal
2. `maternal_metabolic_score`
3. `anaemia_x_early` (interaction term)
4. `metabolic_x_infra` (interaction term)
5. Children who received 3 doses of rotavirus vaccine

**Takeaway:** maternal nutritional status — captured both directly (BMI below normal, overweight/obese) and through the engineered `maternal_metabolic_score` — is the dominant, consistent driver across all three targets. No state-encoded feature appears in the top 14 for any target, confirming the leakage found earlier has been fully removed rather than just reduced. All three targets' R² figures above can now be treated as equally trustworthy.

---

## Project Structure

```
malnutrition-prediction
├── MalnutritionNFHS5.ipynb   # Main notebook (cleaning → modelling → SHAP)
├── datafile.csv              # NFHS-5 district-level dataset
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn
```

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/M1004G/Prediction-of-Malnutrition-in-Children-Under-Five-Years-of-Age-using-Machine-Learning.git
cd Prediction-of-Malnutrition-in-Children-Under-Five-Years-of-Age-using-Machine-Learning
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Open the notebook
```bash
jupyter notebook MalnutritionNFHS5.ipynb
```

4. Upload `datafile.csv` to `/content/` if running on Google Colab, or update `DATA_PATH` in the config cell.

---

## Limitations

- Dataset is cross-sectional (NFHS-5, 2019–21) — causal inference is not possible
- Some columns had high missingness (>50%) and were dropped entirely
- District-level aggregates mask intra-district variation
- No outlier capping was applied (a deliberate choice to preserve genuine high-burden districts); a small number of extreme districts can still swing tree-based model splits
- Wasting remains the hardest target to predict (R² ≈ 0.40) — expected, since it reflects acute, short-term nutritional status that a single cross-sectional survey can't fully capture

---

## Methodology Validation

This project went through several rounds of validation to ensure reported metrics reflect genuine model performance rather than artifacts of the pipeline:

- **Feature selection is fit strictly on training data** before being applied to the test set, avoiding a common pitfall where information from held-out data leaks into feature choice.
- **A candidate geographic feature (state-level target encoding) was tested during development and removed** after SHAP analysis showed it could account for a disproportionate share of model performance on two of the three targets. Re-running without it changed R² only slightly (within 0.01–0.03 across all targets), and the resulting SHAP rankings are now driven entirely by genuine survey indicators (maternal BMI, metabolic health composites, vaccination coverage) rather than any encoded geographic proxy.
- All regression and classification numbers reported above reflect this fully validated, leakage-checked pipeline.

---

## Data Source

> District-level NFHS-5 indicators, obtained via [India National Family Health Survey (NFHS) 2019-21](https://www.kaggle.com/datasets/kmldas/india-national-family-health-survey-nfhs) on Kaggle, a public re-upload of official Government of India survey data.
>
> Underlying survey: **National Family Health Survey-5 (NFHS-5), 2019–21**, Ministry of Health and Family Welfare, Government of India, collected by the International Institute for Population Sciences (IIPS), Mumbai. [http://rchiips.org/nfhs/nfhs5.shtml](http://rchiips.org/nfhs/nfhs5.shtml)

---

## Acknowledgements

This project was independently proposed and developed under the guidance of a mentor as part of the IGDTUW–Anveshan Foundation Machine Learning internship. Topic selection and technical approach were self-directed with periodic mentor review. The NFHS-5 data is publicly available and collected by the International Institute for Population Sciences (IIPS), Mumbai.
