# 🚢 Titanic Survival Prediction (Kaggle)

This repository contains a complete machine learning pipeline for the Titanic survival prediction challenge on [Kaggle](https://www.kaggle.com/competitions/titanic/).

## 📌 Features

- Exploratory Data Analysis (EDA)
- Missing value handling
- Feature engineering:
  - Title extraction from name
  - Family size and solo traveler detection
- One-hot encoding for categorical variables
- Model training with:
  - Gradient Boosting
  - XGBoost (optional)
- Cross-validation
- Hyperparameter tuning
- Final submission file generation

## 📁 File Structure

- `model.py` – trains and saves the model
- `predict.py` – loads the model and predicts on test data
- `train.csv`, `test.csv` – input datasets from Kaggle
- `submission.csv` – output file for Kaggle submission

## ⚙️ Requirements
pip install -r requirements.txt
