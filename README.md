
# Titanic SVM Frontend

A simple Streamlit frontend for the Titanic Survival Prediction project.

## Run locally

Open terminal in this folder:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The browser will open the Streamlit application.

## Model workflow

The app follows the supplied notebook:
- Seaborn Titanic dataset
- Remove unused columns
- Fill missing age with mean
- Drop missing embarked rows
- Label encode `sex` and `embarked`
- StandardScaler
- SVC (Support Vector Classifier)
- Survival prediction

## Main frontend sections

- Passenger statistics
- Survived / not-survived overview
- Female vs male survival-rate chart
- Passenger input form
- SVM survival prediction
- Project workflow information
