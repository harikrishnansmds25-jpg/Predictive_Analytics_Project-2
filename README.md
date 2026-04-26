# Water Quality Classification for Safe Drinking Prediction

An end-to-end machine learning project for classifying water samples as **potable (safe to drink)** or **non-potable** using physicochemical water quality parameters.

##  Project Overview

Unsafe drinking water is a major global health concern. This project uses machine learning to predict water potability based on key chemical and physical characteristics of water. It supports rapid screening for environmental monitoring, public health, and water resource management.

##  Objectives

* Classify water samples as Safe or Not Safe for drinking
* Compare multiple machine learning models
* Improve prediction using feature engineering and SMOTE
* Deploy an interactive Streamlit web application

##  Dataset

* **Source:** Kaggle Water Potability Dataset
* **Samples:** 3,276 water samples
* **Original Features:** 9
* **Engineered Features:** 5
* **Total Features:** 14
* **Target Variable:** `Potability`

  * `1` = Potable
  * `0` = Non-potable

##  Features Used

**Original Features**

* pH
* Hardness
* Solids
* Chloramines
* Sulfate
* Conductivity
* Organic Carbon
* Trihalomethanes
* Turbidity

**Engineered Features**

* ph_deviation
* mineral_load
* chloramines_conductivity
* thm_per_carbon
* turbidity_solids_ratio

## Methodology

1. Data preprocessing and cleaning
2. Missing value imputation using KNN Imputer
3. Outlier treatment using Winsorization
4. Feature engineering
5. Class balancing using SMOTE
6. Feature selection using consensus ranking
7. Model training and hyperparameter tuning
8. Ensemble stacking
9. Streamlit deployment

##  Models Implemented

* Random Forest
* XGBoost
* LightGBM
* Stacking Ensemble (XGBoost + LightGBM + Random Forest)

##  Best Model Performance

* **Best Model:** Stacking Ensemble
* **Accuracy:** ~82%
* **F1-Macro:** ~0.81
* **ROC-AUC:** ~0.87

##  Streamlit Application Features

* Train models directly in the browser
* Upload CSV files for batch prediction
* Manual single-sample prediction
* Confidence score for each prediction
* Download prediction results as CSV

##  Repository Structure

```text
├── app.py
├── requirements.txt
├── Water_Quality_classification.ipynb
├── feature_engineering.py
├── models/
├── scaler.pkl
├── selector.pkl
├── README.md
└── water_quality_dataset.zip
```

##  Deployment

Deployed using Streamlit Cloud for easy online access.
App link: https://predictiveanalyticsproject-2app.streamlit.app/
##  Technologies Used

* Python 3.10+
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Imbalanced-learn (SMOTE)
* Matplotlib
* Seaborn
* Streamlit
* Joblib



