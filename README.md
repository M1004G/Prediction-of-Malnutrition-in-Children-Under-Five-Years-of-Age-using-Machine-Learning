# Malnutrition Prediction & Classification using Machine Learning

## 1. Project Overview
This project analyzes and predicts child malnutrition indicators using WHO categories.  
It combines **classification models** (predicting categorical malnutrition status) and **regression models** (predicting continuous malnutrition rates) to offer both qualitative and quantitative insights.

**Targets:**
- **HAZ**: Height-for-Age Z-score → *Stunting*
- **WAZ**: Weight-for-Age Z-score → *Underweight*
- **WHZ**: Weight-for-Height Z-score → *Wasting*


## 2. Data Cleaning & Preprocessing
1. **Initial Inspection**
   - Checked for missing values, duplicates, and incorrect data types.
2. **Missing Value Handling**
   - Numerical: Median imputation.
   - Categorical: Mode or 'Unknown'.
3. **Feature Engineering**
   - Generated WHO malnutrition category columns.
   - One-hot encoded categorical variables.
   - Normalized numerical features (for models like KNN).
4. **Balancing**
   - Applied **SMOTEENN** to handle class imbalance.
5. **Splitting**
   - Used **Stratified Train-Test Split** for classification to preserve class distribution.


## 3. Models Used

### Classification Models
- **Random Forest with SMOTEENN (RF_SMOTEENN)**
- **Balanced Random Forest (BalancedRF)**
- **XGBoost Classifier (XGB)**

### Regression Models
- **Linear Regression (Baseline)**
- **Random Forest Regressor**
- **Enhanced Random Forest (Cross-validated)**
- **Enhanced XGBoost Multi-Output Regressor**

## 4. Classification Results

### HAZ_Category (Stunting)
| Model        | Macro F1 | Accuracy |  
|--------------|----------|----------|
| RF_SMOTEENN  | 0.844    | -        |      
| BalancedRF   | 0.911    | -        |       
| XGB          | **0.929**| 87%      | 

**XGB Class Report:**
- High: Precision=0.98, Recall=0.82, F1=0.86
- Low: Precision=1.00, Recall=1.00, F1=1.00
- Moderate: Precision=0.83, Recall=0.83, F1=0.83
- Very High: Precision=0.87, Recall=1.00, F1=0.93


### WAZ_Category (Underweight)
| Model        | Macro F1 | Accuracy | 
|--------------|----------|----------|
| RF_SMOTEENN  | 0.699    | -        |       
| BalancedRF   | **0.798**| 92%      | 
| XGB          | 0.755    | -        |       

**BalancedRF Class Report:**
- High: Precision=0.95, Recall=0.90, F1=0.92
- Moderate: Precision=0.70, Recall=0.79, F1=0.84
- Other: Precision=1.00, Recall=0.67, F1=0.81
- Very High: Precision=1.00, Recall=0.99, F1=0.92


### WHZ_Category (Wasting)
| Model        | Macro F1 | Accuracy | 
|--------------|----------|----------|
| RF_SMOTEENN  | 0.914    | -        |      
| BalancedRF   | 0.916    | -        |       
| XGB          | **0.936**| 94%      | 

**XGB Class Report:**
- Mild: Precision=0.77, Recall=0.77, F1=0.77
- Moderate: Precision=0.89, Recall=0.83, F1=0.86
- Severe: Precision=0.98, Recall=1.00, F1=0.99


## 5. Regression Results

### Baseline Linear Regression
| Target       | MSE    | R²    |
|--------------|--------|-------|
| Stunting     | 18.41  | 0.739 |
| Wasting      | 8.78   | 0.734 |
| Underweight  | 18.72  | 0.780 |

---

### Random Forest (Per Target)
| Target       | MSE    | R²    |
|--------------|--------|-------|
| Stunting     | 22.67  | 0.679 |
| Wasting      | 6.72   | 0.796 |
| Underweight  | 24.43  | 0.713 |

---

### Enhanced Random Forest (Cross-Validated)
- **Mean CV R²** = 0.758

| Target       | MSE    | R²    |
|--------------|--------|-------|
| Stunting     | 21.26  | 0.699 |
| Wasting      | 6.43   | 0.805 |
| Underweight  | 21.03  | 0.753 |

---

### Enhanced XGBoost Multi-Output Regressor
| Target       | Mean CV R² | Test R² | Test RMSE |
|--------------|------------|---------|-----------|
| Stunting     | 0.734      | 0.812   | 3.65      |
| Underweight  | 0.821      | 0.841   | 3.68      |
| Wasting      | 0.839      | 0.835   | 2.33      |

---

## 6. Key Takeaways
- **XGBoost Classifier** gave best results for classification tasks.
- **Enhanced XGBoost Regressor** achieved highest R² scores for regression.
- **SMOTEENN balancing** significantly improved performance for minority classes.
- Combining categorical WHO-based classification with continuous regression yields better insights.

---

## 7. Authors
- **Muskan Goel**  
- **Kriti Jangra**
