# 🍽️ Prediction of Malnutrition in Children Under Five Years of Age Using Machine Learning

> Predicting stunting, wasting, and underweight prevalence across Indian districts using NFHS-5 survey data and ensemble machine learning models.

---

## 📌 Overview

Child malnutrition remains one of India's most critical public health challenges. This project applies supervised machine learning to **predict the percentage of children under 5 years** who are stunted, wasted, or underweight at the district level — using socioeconomic, maternal health, and household environment indicators from India's **National Family Health Survey (NFHS-5)**.

The goal is to identify which districts are at highest risk and which upstream factors (maternal health, sanitation, anaemia, literacy, etc.) are the strongest predictors of malnutrition — insights that can inform policy and resource allocation.

---

## ⚠️ Note on Methodology Correction

An earlier version of this project reported inflated performance due to two sources of data leakage: (1) target-derived severity flags and cross-target ratios computed from the label itself and fed back in as input features, and (2) feature selection performed on the full dataset before the train/test split. This version removes both — feature selection is now fit strictly on training data, and reported metrics reflect genuine train/test separation. As a result, headline R² dropped from the low 0.8s to a more realistic 0.4–0.7 range, which is the honest number for this kind of noisy, cross-sectional survey data.

One smaller-scale leakage source remains under review: state-level target encoding is currently computed with an internal KFold safeguard (a district can't see its own value), but is not yet strictly re-fit within the train/test boundary. SHAP analysis shows this feature ranks outside the top 10 for the targets checked so far, so its effect on reported metrics is likely minor — but it hasn't been fully eliminated yet.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | NFHS-5 (2019–21), Government of India |
| **Unit of analysis** | District-level |
| **Total records** | 706 districts (705 after deduplication) |
| **Total features** | 93 numeric columns after cleaning, expanded to 111 after state encoding and domain feature engineering |
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

## 🔬 Methodology

### 1. Data Cleaning & Preprocessing
- Replaced dirty values (`*`, `-`, `NA`, parenthesized numbers) with `NaN`
- Dropped columns with >50% missing values
- Median imputation for remaining missing values (no outlier capping — see Limitations)
- Removed 1 duplicate row

### 2. Feature Engineering
- Composite domain scores built from raw feature columns only (e.g. `maternal_health_score`, `household_infra_score`, `child_diet_score`) — no target information used
- State/UT target-encoded via KFold to capture geographic clustering, without a district seeing its own value
- Mutual Information (MI) feature selection, **fit on training data only**, top 40 features used for the regularized ElasticNet baseline; tree-based models use all features

### 3. Train/Test Split
80/20 split — 564 training districts, 141 test districts.

### 4. Regression Results (Test Set)

| Model | Stunting (HAZ) R² | Wasting (WHZ) R² | Underweight (WAZ) R² |
|---|---|---|---|
| Elastic Net | 0.558 | 0.417 | 0.698 |
| Random Forest | 0.578 | 0.394 | 0.685 |
| XGBoost | 0.569 | 0.434 | 0.703 |
| LightGBM | 0.575 | 0.392 | 0.721 |
| Extra Trees | 0.576 | 0.407 | 0.690 |
| **Weighted Blend** (LGB 0.40 + ET 0.35 + XGB 0.25) | **0.586** | **0.428** | **0.720** |

RMSE for the blended model: 5.55 (stunting), 4.41 (wasting), 4.81 (underweight) — all in percentage points.

### 5. Classification Results (WHO Severity Tiers)

| Target | Best Model | Macro F1 | Accuracy |
|---|---|---|---|
| Stunting (HAZ) | SMOTETomek + RF | 0.506 | 0.589 |
| Wasting (WHZ) | Balanced RF | 0.561 | 0.660 |
| Underweight (WAZ) | LightGBM | 0.521 | 0.716 |

Wasting and underweight classes are heavily imbalanced (e.g. "Severe Wasting" is 494/705 districts); macro F1 is reported alongside accuracy since accuracy alone is misleading here.

---

## 🔑 Top Predictive Features

SHAP analysis on the stunting model surfaces:

1. **Maternal metabolic score** (composite: BMI/obesity + menstrual hygiene) — top driver
2. **Women with BMI below normal** — second strongest individual predictor
3. **Births that are third-or-higher order** — proxy for family planning access
4. **Maternal health score** (literacy, ANC visits, postnatal care)
5. **Anaemia in pregnant women**

Maternal nutritional status is the dominant upstream driver for stunting specifically; SHAP was not separately reviewed for wasting and underweight in this write-up, so the same ranking shouldn't be assumed to hold for those targets without checking.

---

## 🗂️ Project Structure

```
📦 malnutrition-prediction
 ┣ 📓 Malnutrition_NFHS5.ipynb   # Main notebook (cleaning → modelling → SHAP)
 ┣ 📄 datafile.csv                  # NFHS-5 district-level dataset
 ┗ 📄 README.md
```

---

## ⚙️ Requirements

```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn
```

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/M1004G/Prediction-of-Malnutrition-in-Children-Under-Five-Years-of-Age-using-Machine-Learning.git
cd Prediction-of-Malnutrition-in-Children-Under-Five-Years-of-Age-using-Machine-Learning
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn
```

3. Open the notebook
```bash
jupyter notebook Malnutrition_NFHS5.ipynb
```

4. Upload `datafile.csv` to `/content/` if running on Google Colab, or update `DATA_PATH` in the config cell.

---

## ⚠️ Limitations

- Dataset is cross-sectional (NFHS-5, 2019–21) — causal inference is not possible
- Some columns had high missingness (>50%) and were dropped entirely
- District-level aggregates mask intra-district variation
- No outlier capping was applied (a deliberate choice to preserve genuine high-burden districts); a small number of extreme districts can still swing tree-based model splits
- State-level target encoding has an internal leakage safeguard but is not yet strictly re-fit within the train/test boundary; SHAP indicates its impact on reported metrics is likely minor but not yet fully verified

---

## 📚 Data Source

> District-level NFHS-5 indicators, obtained via [India National Family Health Survey (NFHS) 2019-21](https://www.kaggle.com/datasets/kmldas/india-national-family-health-survey-nfhs) on Kaggle, a public re-upload of official Government of India survey data.
>
> Underlying survey: **National Family Health Survey-5 (NFHS-5), 2019–21**, Ministry of Health and Family Welfare, Government of India, collected by the International Institute for Population Sciences (IIPS), Mumbai. [http://rchiips.org/nfhs/nfhs5.shtml](http://rchiips.org/nfhs/nfhs5.shtml)

---

## 🙏 Acknowledgements

This project was independently proposed and developed under the guidance of a mentor as part of the IGDTUW–Anveshan Foundation Machine Learning internship. Topic selection and technical approach were self-directed with periodic mentor review. The NFHS-5 data is publicly available and collected by the International Institute for Population Sciences (IIPS), Mumbai.
