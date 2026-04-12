# 🍽️ Prediction of Malnutrition in Children Under Five Years of Age Using Machine Learning

> Predicting stunting, wasting, and underweight prevalence across Indian districts using NFHS-5 survey data and ensemble machine learning models.

---

## 📌 Overview

Child malnutrition remains one of India's most critical public health challenges. This project applies supervised machine learning to **predict the percentage of children under 5 years** who are stunted, wasted, or underweight at the district level — using socioeconomic, maternal health, and household environment indicators from India's **National Family Health Survey (NFHS-5)**.

The goal is to identify which districts are at highest risk and which upstream factors (literacy, sanitation, anaemia, breastfeeding, etc.) are the strongest predictors of malnutrition — insights that can inform policy and resource allocation.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | NFHS-5 (2019–21), Government of India |
| **Unit of analysis** | District-level |
| **Total records** | 706 districts (705 after deduplication) |
| **Total features** | 109 columns → 32 used after cleaning |
| **Target variables** | 3 (stunting %, wasting %, underweight %) |

### Target Variables
- `Children under 5 years who are stunted (height-for-age)` — chronic malnutrition
- `Children under 5 years who are wasted (weight-for-height)` — acute malnutrition
- `Children under 5 years who are underweight (weight-for-age)` — composite indicator

### Feature Categories
- **Child health & diet** — breastfeeding rates, vaccination coverage, anaemia prevalence, diarrhoea
- **Maternal health** — literacy, antenatal care, iron supplementation, anaemia
- **Household environment** — electricity, sanitation, clean fuel, iodized salt, health insurance

---

## 🔬 Methodology

### 1. Data Cleaning & Preprocessing
- Replaced dirty values (`*`, `-`, `NA`, parenthesized numbers) with `NaN`
- Dropped columns with >50% missing values
- Imputed remaining missing values using **mean** (symmetric) or **median** (skewed), determined per-column by skewness test
- Capped outliers using IQR method (1.5× rule), replacing with column median
- Removed 1 duplicate row

### 2. Feature Engineering
- Separated target columns from feature matrix prior to any modelling (no leakage)
- Applied `StandardScaler` for models requiring normalized input
- Computed feature-target correlations; selected features with |r| > 0.4 for baseline

### 3. Models Trained

#### Regression (predicting raw % values)
| Model | Targets | Best R² |
|---|---|---|
| XGBoost Regressor (enhanced, per-target) | All 3 | **0.845** (underweight) |
| Multi-Output Random Forest + CV | All 3 | 0.783 (wasting) |
| Random Forest per Target | All 3 | 0.775 (wasting) |
| Linear Regression (baseline) | All 3 | 0.763 (underweight) |
| XGBoost + RandomizedSearchCV | All 3 | 0.673 (underweight) |

#### Classification (WHO threshold-based categories)
| Model | Target | Macro F1 |
|---|---|---|
| XGBoost Classifier | Stunting (HAZ) | **0.909** |
| Balanced Random Forest | Stunting (HAZ) | 0.906 |
| XGBoost Classifier | Wasting (WHZ) | 0.867 |
| RF + SMOTEENN | Wasting (WHZ) | **0.887** |
| XGBoost Classifier | Underweight (WAZ) | 0.743 |

### 4. Best Regression Results (Enhanced XGBoost, per-target)

| Target | CV R² | Test R² | Test RMSE |
|---|---|---|---|
| Stunting | 0.729 | **0.797** | 3.89 |
| Underweight | 0.816 | **0.845** | 3.58 |
| Wasting | 0.831 | **0.837** | 2.36 |

### 5. Best Classification Results (WHO % Categories)

| Target | Accuracy | Macro F1 |
|---|---|---|
| Stunting (HAZ) | 82% | 0.819–0.909 |
| Wasting (WHZ) | 96% | 0.867–0.887 |
| Underweight (WAZ) | 92% | 0.672–0.758 |

---

## 🔑 Top Predictive Features

From XGBoost feature importance and Pearson correlation analysis, the strongest predictors of child malnutrition are:

1. **Women's BMI below normal** (r = 0.74 with underweight) — strongest single predictor
2. **Population below age 15 years** — demographic pressure indicator
3. **Women's literacy rate** — maternal education strongly linked to all 3 targets
4. **Women with 10+ years of schooling**
5. **Improved sanitation access**
6. **Households using clean fuel**
7. **Female school attendance**
8. **Child marriage rate** (women married before 18)
9. **Children aged 6–59 months who are anaemic**
10. **All women who are anaemic**

SHAP analysis confirmed that **maternal nutritional status and education are the dominant upstream drivers** of child malnutrition across Indian districts.

---

## 🗂️ Project Structure

```
📦 malnutrition-prediction
 ┣ 📓 ML_project.ipynb        # Main notebook (data cleaning → modelling → SHAP)
 ┣ 📄 datafile.csv            # NFHS-5 district-level dataset
 ┗ 📄 README.md
```

---

## ⚙️ Requirements

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn catboost
```

| Library | Version |
|---|---|
| Python | 3.10+ |
| pandas | 2.x |
| scikit-learn | 1.x |
| xgboost | 1.7+ |
| imbalanced-learn | 0.11+ |
| shap | 0.44+ |

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/malnutrition-prediction.git
cd malnutrition-prediction
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Open the notebook
```bash
jupyter notebook ML_project.ipynb
```

4. Upload `datafile.csv` to `/content/` if running on Google Colab, or update the file path in Cell 1.

---

## 📈 Key Findings

- **Wasting** (acute malnutrition) is the hardest to predict with regression (R² ~0.78–0.84), likely due to its episodic nature linked to immediate illness
- **Underweight** is the most predictable target (R² up to 0.845), as it integrates both chronic and acute malnutrition signals
- **Classification into WHO severity categories** works very well (F1 up to 0.91), making the models practical for district-level risk stratification
- Districts with **low female literacy, high child marriage rates, and poor sanitation** consistently cluster in the high-malnutrition categories
- **Maternal BMI** alone explains ~27% of feature importance for stunting prediction

---

## ⚠️ Limitations

- Dataset is cross-sectional (NFHS-5, 2019–21) — causal inference is not possible
- Some columns had high missingness (>90% for diarrhoea-related treatment columns) and were dropped
- District-level aggregates mask intra-district variation
- Model performance is bounded by the quality and granularity of NFHS survey data

---

## 📚 Data Source

> **National Family Health Survey-5 (NFHS-5), 2019–21**
> Ministry of Health and Family Welfare, Government of India
> [http://rchiips.org/nfhs/nfhs5.shtml](http://rchiips.org/nfhs/nfhs5.shtml)

---

## 🙏 Acknowledgements

This project was developed as part of an academic machine learning course. The NFHS-5 data is publicly available and collected by the International Institute for Population Sciences (IIPS), Mumbai.

