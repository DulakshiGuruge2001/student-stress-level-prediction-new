# Student Stress Level Prediction

A comprehensive machine learning project that predicts student stress levels using supervised learning and clustering techniques. This project analyzes various psychological and behavioral factors to classify stress into three categories: **No Stress**, **Eustress** (positive stress), and **Distress** (negative stress).

## 📊 Project Overview

This project implements a complete data science pipeline including:
- **Exploratory Data Analysis (EDA)** with statistical visualizations
- **Data Preprocessing** including outlier detection and K-Prototypes clustering
- **Machine Learning Models** with multiple classification algorithms
- **Comparative Analysis** between different predictive approaches

### Key Features
- Multi-class stress classification (3 categories)
- Hybrid clustering approach combining numerical and categorical features
- Multiple machine learning algorithms for robust predictions
- Comprehensive feature importance analysis
- Visualization of distributions and model performance

## 🗂️ Project Structure

```
student-stress-level-prediction-new/
├── codes/
│   ├── EDA.R                           # Exploratory Data Analysis
│   ├── Outlier.R                       # Outlier detection and handling
│   ├── Comparison 2 variables.R        # Bivariate analysis
│   ├── K-prototypeClustering.R         # Mixed-type clustering
│   ├── Logistic.R                      # Multinomial logistic regression
│   ├── SVM.R                           # Support Vector Machine models
│   ├── Random forest.R                 # Random Forest classification
│   ├── Xg boost.R                      # XGBoost classification
│   └── Final_Project.py                # Comprehensive Python implementation
├── Data set/                           # Input datasets
├── Report/                             # Analysis reports and findings
└── README.md                           # This file
```

## 💻 Technologies & Languages

- **Python** (68.5%) - NumPy, Pandas, Scikit-learn, XGBoost
- **R** (31.5%) - Caret, kernlab, clustMixType, ggplot2

## 📋 Dataset

The project uses a student stress dataset containing:
- **Psychological Indicators**: anxiety levels, depression scores, self-esteem measures
- **Behavioral Features**: categorical lifestyle and environmental factors
- **Target Variable**: stress_level (No_Stress, Eustress, Distress)

### Data Preprocessing
- Duplicate row removal
- Missing value handling with `na.omit()`
- Categorical variable encoding
- Numerical feature scaling (standardization)
- 80-20 train-test split

## 🔍 Key Components

### 1. Exploratory Data Analysis (`EDA.R`)
- Summary statistics and data shape analysis
- Histogram distributions for numerical variables
- Pie chart showing stress level proportions
- Boxplot analysis of features by stress category
- Train-test data visualization

### 2. Data Cleaning & Outliers (`Outlier.R`)
- Identifies and visualizes outliers
- Handles extreme values using statistical methods
- Preserves data integrity while removing anomalies

### 3. Clustering Analysis (`K-prototypeClustering.R`)
- **K-Prototypes Algorithm**: Handles mixed data types (numerical + categorical)
- Elbow method for optimal cluster selection (k=3)
- Silhouette analysis using Gower distance metric
- Cluster profiling with numerical means and categorical distributions
- PCA visualization of clusters

### 4. Classification Models

#### Multinomial Logistic Regression (`Logistic.R`)
- Elastic Net regularization (Ridge/Lasso/ElasticNet)
- 5-fold cross-validation with 2 repeats
- Feature importance with aggregated metrics
- Confusion matrix and F1-score evaluation

#### Support Vector Machines (`SVM.R`)
- **4 Kernel Types**:
  1. Linear kernel
  2. Radial Basis Function (RBF)
  3. Polynomial kernel (degrees 2-4)
  4. Sigmoid (tanh) kernel
- Hyperparameter tuning (C, sigma, scale)
- Permutation-based feature importance
- Performance metrics: Accuracy, F1-score

#### Random Forest (`Random forest.R`)
- Ensemble approach with multiple decision trees
- Feature importance ranking
- Robustness to outliers

#### XGBoost (`Xg boost.R`)
- Gradient boosting optimization
- Advanced feature importance analysis
- High predictive accuracy

### 5. Comparative Analysis (`Comparison 2 variables.R`)
- Bivariate relationships between key features
- Statistical significance testing
- Feature interaction analysis

### 6. Python Implementation (`Final_Project.py`)
- Comprehensive pipeline in Python
- Integration of all preprocessing and modeling steps
- Production-ready code with extensive documentation

## 🚀 Usage

### Running R Scripts
```bash
# EDA and data exploration
Rscript codes/EDA.R

# K-Prototypes clustering
Rscript codes/K-prototypeClustering.R

# Logistic Regression model
Rscript codes/Logistic.R

# SVM with multiple kernels
Rscript codes/SVM.R

# Random Forest
Rscript codes/Random\ forest.R

# XGBoost
Rscript codes/Xg\ boost.R
```

### Running Python Script
```bash
python codes/Final_Project.py
```

### Requirements
**R Packages:**
```r
library(caret)
library(ggplot2)
library(dplyr)
library(kernlab)
library(clustMixType)
library(cluster)
library(factoextra)
library(glmnet)
```

**Python Packages:**
```
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
```

## 📈 Model Performance

Models are evaluated using:
- **Accuracy**: Overall classification correctness
- **Precision**: Proportion of correct positive predictions
- **Recall (Sensitivity)**: True positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class classification breakdown

### Key Predictive Features
Based on feature importance analysis:
1. **Anxiety Levels** - Strong predictor of stress
2. **Depression Scores** - Highly correlated with distress
3. **Self-Esteem Measures** - Inverse relationship with stress
4. **Categorical Factors** - Lifestyle and environmental variables

## 🎯 Key Findings

1. **Three-Class Problem**: Successfully classifies students into three distinct stress categories
2. **Mixed-Type Data Handling**: K-Prototypes effectively captures both numerical and categorical patterns
3. **Model Comparison**: Multiple algorithms provide robust predictions with complementary strengths
4. **Feature Relationships**: Clear psychological indicators of stress levels
5. **Cluster Validation**: Well-separated clusters with positive silhouette scores

## 📊 Visualizations

The project generates numerous visualizations:
- Distribution histograms for each feature
- Stress level proportions (pie charts)
- Feature-by-stress boxplots
- Cluster PCA visualization
- Silhouette plots
- Feature importance bar charts
- Model comparison plots

## 🔄 Workflow

```
Raw Data
   ↓
Data Cleaning (EDA, Outliers)
   ↓
Feature Engineering & Clustering
   ↓
Train-Test Split (80-20)
   ↓
Model Training & Cross-Validation
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Feature Importance Analysis
   ↓
Comparative Results
