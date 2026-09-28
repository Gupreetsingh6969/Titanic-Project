<<<<<<< HEAD

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
=======
# Titanic-Project
Titanic Survival Prediction using Machine Learning and SVM. This project includes data preprocessing, feature scaling, survival analysis, and an interactive Streamlit web application for predicting passenger survival.
>>>>>>> 11794d69be69df1d8e952aa20e737515d761a325
