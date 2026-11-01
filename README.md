<div align="center">

# CardioGuard: Machine Learning Heart Disease Detection & Risk Stratification

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-88%25--91%25-brightgreen?style=for-the-badge)](https://github.com/pkpranav-projects/Heart_Disease_Detection/releases)
[![Release](https://img.shields.io/badge/Release-v1.0.0-blue?style=for-the-badge)](https://github.com/pkpranav-projects/Heart_Disease_Detection/releases/tag/v1.0.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An end-to-end clinical machine learning framework that evaluates multi-parametric patient vitals to predict coronary artery disease with high diagnostic fidelity using both Linear Discriminant and Ensemble Decision Forest architectures.

[Key Features](#-key-features) • [Clinical Dataset](#-clinical-dataset--feature-dictionary) • [Model Architectures](#-model-architectures--mathematical-formulation) • [Quickstart](#-quickstart--installation) • [Patient Prediction](#-live-patient-inference) • [Author](#-author)

</div>

---

## Overview

Cardiovascular diseases (CVDs) remain the leading global cause of mortality. Early predictive diagnosis enables preventative interventions before irreversible cardiac remodeling occurs. 

**CardioGuard** implements an automated clinical pipeline that:
1. Ingests raw biometric patient profiles (resting blood pressure, cholesterol, resting electrocardiographic results, maximum heart rate).
2. Performs automated feature scaling and variance normalization via `StandardScaler`.
3. Trains and benchmarks dual classification models (**Logistic Regression** with L2 regularization and a **Random Forest Ensemble** with 200 estimators).
4. Provides real-time inference vectors for single-patient clinical screening.

---

## Key Features

- **Dual-Model Comparative Benchmark**: Includes both a calibrated linear classifier (`LogisticRegression(max_iter=1000)`) and a non-linear ensemble (`RandomForestClassifier(n_estimators=200, max_depth=10)`).
- **Z-Score Feature Normalization**: Mitigates feature-magnitude bias across continuous physiological indicators (e.g., serum cholesterol vs. fluoroscopy vessels).
- **Comprehensive Diagnostic Reporting**: Computes Precision, Recall, F1-Score, Support, and Confusion Matrices across healthy and at-risk cohorts.
- **Interactive Patient Inference**: Ships with out-of-the-box test vectors allowing clinicians to feed arbitrary 13-point physiological profiles for instantaneous risk verdicts.

---

## Clinical Dataset & Feature Dictionary

The model is trained on the canonical Cleveland/UCI Heart Disease cohort comprising 13 key physiological features:

| Parameter | Identifier | Clinical Description | Typical Range / Units |
| :--- | :--- | :--- | :--- |
| **Age** | `age` | Age of the patient in years | 29 – 77 yrs |
| **Sex** | `sex` | Biological sex | `1` = Male, `0` = Female |
| **Chest Pain Type** | `cp` | Anginal severity indicator | `0`: Asymptomatic, `1`: Typical, `2`: Atypical, `3`: Non-anginal |
| **Resting BP** | `trestbps` | Resting blood pressure on hospital admission | 94 – 200 mm Hg |
| **Serum Cholestoral** | `chol` | Fasting serum cholesterol level | 126 – 564 mg/dl |
| **Fasting Blood Sugar** | `fbs` | Fasting blood glucose test (> 120 mg/dl) | `1` = True, `0` = False |
| **Resting ECG** | `restecg` | Resting electrocardiographic evaluation | `0`: Normal, `1`: ST-T wave anomaly, `2`: LVH |
| **Max Heart Rate** | `thalach` | Maximum heart rate achieved during stress test | 71 – 202 bpm |
| **Exercise Angina** | `exang` | Exercise-induced angina | `1` = Yes, `0` = No |
| **ST Depression** | `oldpeak` | ST depression induced by exercise relative to rest | 0.0 – 6.2 |
| **ST Slope** | `slope` | Slope of the peak exercise ST segment | `0`: Upsloping, `1`: Flat, `2`: Downsloping |
| **Major Vessels** | `ca` | Number of major vessels colored by fluoroscopy | 0 – 3 vessels |
| **Thalassemia** | `thal` | Myocardial perfusion scintigraphy defect | `1` = Normal, `2` = Fixed defect, `3` = Reversible |
| **Target** | `target` | Presence of angiographic disease (> 50% stenosis) | `1` = Disease Detected, `0` = Normal |

---

## Architecture & ML Pipeline

```text
  +--------------------------+
  |    heart_database.csv    |  <-- 13 Biometric Patient Parameters
  +--------------------------+
               |
               v
  +--------------------------+
  | Preprocessing & Cleansing|  <-- Null value verification & outlier screening
  +--------------------------+
               |
               v
  +--------------------------+
  |  Stratified Train/Test   |  <-- 80% Training (Fit) / 20% Evaluation (Test)
  |     Split (seed=42)      |
  +--------------------------+
               |
               v
  +--------------------------+
  |  StandardScaler Z-Score  |  <-- Normalization: z = (x - mean) / std
  +--------------------------+
        /                  \
       v                    v
+------------------+  +--------------------+
| Logistic Model   |  | Random Forest      |
| max_iter = 1000  |  | 200 Trees, Depth 10|
+------------------+  +--------------------+
       \                    /
        v                  v
  +--------------------------+
  | Evaluation & Metrics     |  <-- Accuracy: ~88% - 91%
  | Confusion Matrix / Report|
  +--------------------------+
               |
               v
  +--------------------------+
  |  Single-Patient Predictor|  <-- Instant Clinical Screening Output
  +--------------------------+
```

---

## Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/pkpranav-projects/Heart_Disease_Detection.git
cd Heart_Disease_Detection
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Running the Models

### Option A: Logistic Regression Pipeline
```bash
python heart_disease_detection_logisticregression.py
```
**Sample Output:**
```text
   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target
0   63    1   3       145   233    1        0      150      0      2.3      0   0     1       1

Missing Values:
 0

Accuracy: 0.8852459016393442

Classification Report:
               precision    recall  f1-score   support
           0       0.89      0.86      0.88        29
           1       0.88      0.91      0.89        32
    accuracy                           0.89        61

Patient Sample: [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
Diagnosis Verdict: Heart Disease Detected
```

### Option B: Random Forest Ensemble Pipeline
```bash
python heart_disease_detection_randomforest.py
```
**Sample Output:**
```text
Accuracy: 0.9016393442622951

Confusion Matrix:
[[27  2]
 [ 4 28]]

Classification Report:
               precision    recall  f1-score   support
           0       0.87      0.93      0.90        29
           1       0.93      0.88      0.90        32
    accuracy                           0.90        61

Patient Sample: [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]
Diagnosis Verdict: No Heart Disease Detected
```

---

## Live Patient Inference

To predict risk for a new patient in code:

```python
import numpy as np

# Format: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
new_patient = np.array([[58, 1, 2, 140, 220, 0, 1, 160, 0, 1.4, 1, 0, 2]])

# Scale against fitted StandardScaler
scaled_patient = scaler.transform(new_patient)
prediction = model.predict(scaled_patient)

print("Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected")
```

---

## Repository Structure

```text
Heart_Disease_Detection/
├── heart_database.csv                           # Complete clinical dataset (UCI Cleveland)
├── heart_disease_detection_logisticregression.py # Logistic Regression pipeline & inference
├── heart_disease_detection_randomforest.py      # Random Forest ensemble pipeline (200 trees)
├── .gitignore                                   # Python & runtime environment exclusions
├── LICENSE                                      # MIT Open Source License
└── README.md                                    # Project documentation
```

---

## Author

Developed by **P K PRANAV** ([@pkpranav-projects](https://github.com/pkpranav-projects)).

## License

This project is licensed under the [MIT License](LICENSE).