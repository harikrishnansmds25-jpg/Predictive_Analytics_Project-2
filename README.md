# Water Quality Classification for Safe Drinking Prediction

An end-to-end machine learning project for classifying water samples as **potable (safe to drink)** or **non-potable** using physicochemical water quality parameters.

 ## 👩‍💻 Team Members
- HARIKRISHNAN S.M  
- ABHITHA RAJ .S.R 
- AYANA SANTHOSH KHAN

## Course Details
Predictive Analytics
 
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
# Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) was performed to understand the structure, quality, and relationships within the water quality dataset before model building.

The analysis helped identify:

- Missing values
- Outliers
- Feature distributions
- Correlation between variables
- Class imbalance issues

---

## Dataset Overview

- Total Samples: 3276
- Original Features: 9
- Engineered Features: 5
- Target Variable: Potability

### Target Variable

- 1 → Potable (Safe Drinking Water)
- 0 → Non-potable (Unsafe Water)

---

# EDA Process

## 1. Data Inspection

Initial inspection was carried out using:

```python
df.head()
df.info()
df.describe()
```

### Observations

- All features were numerical.
- Some columns contained missing values.
- Feature scales varied significantly.

---

## 2. Missing Value Analysis

Missing values were identified using:

```python
df.isnull().sum()
```

### Features with Missing Values

- pH
- Sulfate
- Trihalomethanes

### Observation

Moderate missing values were present and later handled using KNN Imputation.

---

## 3. Target Variable Distribution

Class distribution was analyzed using count plots.

```python
sns.countplot(x='Potability', data=df)
```

### Observation

- The dataset was imbalanced.
- Non-potable samples were higher than potable samples.
- SMOTE was used to balance the classes.

---

## 4. Statistical Summary

Descriptive statistics were generated using:

```python
df.describe()
```

### Findings

- Features like Solids and Conductivity showed high variance.
- Several features contained extreme values.

---

## 5. Feature Distribution Analysis

Histograms were used to analyze feature distributions.

```python
df.hist(figsize=(15,10))
```

### Observation

- Many features were skewed.
- Some variables showed non-normal distribution.

---

## 6. Outlier Detection

Boxplots were used to detect outliers.

```python
sns.boxplot(data=df)
```

### Observation

Outliers were found in:

- Solids
- Sulfate
- Conductivity
- Trihalomethanes

### Handling Method

Winsorization was applied to reduce the impact of extreme values.

---

## 7. Correlation Analysis

Correlation between features was analyzed using a heatmap.

```python
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

### Observation

- Moderate relationships existed between several features.
- No severe multicollinearity was observed.

---

## 8. Feature Engineering

Additional features were created to improve model performance.

### Engineered Features

- ph_deviation
- mineral_load
- chloramines_conductivity
- thm_per_carbon
- turbidity_solids_ratio

### Observation

Engineered features improved class separation and prediction accuracy.

---

## 9. Pairplot Analysis

Pairplots were used to visualize class separation.

```python
sns.pairplot(df, hue='Potability')
```

### Observation

Some overlap existed between classes, but engineered features improved separability.

---

# EDA Conclusion

The Exploratory Data Analysis revealed:

- Presence of missing values and outliers
- Imbalanced target classes
- Skewed feature distributions
- Useful feature relationships
- Importance of preprocessing and feature engineering

EDA played a crucial role in improving data quality and enhancing machine learning model performance.

---

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


##  Deployment

Deployed using Streamlit Cloud for easy online access.
App link: https://predictiveanalyticsproject-2app.streamlit.app/
<img width="750" height="429" alt="image" src="https://github.com/user-attachments/assets/a7534694-1bb5-4b72-95ee-3c54849e7c12" />
<img width="690" height="534" alt="image" src="https://github.com/user-attachments/assets/e3dbdb31-7b05-4466-8f85-99a863841be4" />
<img width="675" height="493" alt="image" src="https://github.com/user-attachments/assets/5c8d12a8-3a16-46de-8ab4-cebe6e2ed484" />
<img width="1296" height="532" alt="image" src="https://github.com/user-attachments/assets/d752931f-b893-4d1a-a428-244edf1fc968" />




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
## Instructions for Setting Up and Running the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/water-quality-classification.git
cd water-quality-classification
```

### 2. Create a Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies

Make sure Python 3.10 or above is installed on your system.

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

After running the command, the application will automatically open in your browser.

If it does not open automatically, visit:

```bash
http://localhost:8501
```



