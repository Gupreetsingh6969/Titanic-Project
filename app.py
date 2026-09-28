
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)


# =========================================================
# STYLING
# =========================================================

st.markdown("""
<style>

.main {
    background: #f6f8fb;
}

.hero {
    padding: 28px 32px;
    border-radius: 18px;
    background: linear-gradient(135deg, #172554, #2563eb);
    color: white;
    margin-bottom: 24px;
}

.hero h1 {
    margin: 0;
    font-size: 38px;
}

.hero p {
    margin-top: 8px;
    font-size: 16px;
    opacity: 0.9;
}

.card {
    padding: 20px;
    border-radius: 16px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0 3px 12px rgba(0,0,0,0.05);
}

.result-survive {
    padding: 24px;
    border-radius: 16px;
    background: #ecfdf5;
    border: 1px solid #86efac;
    text-align: center;
}

.result-not {
    padding: 24px;
    border-radius: 16px;
    background: #fef2f2;
    border: 1px solid #fca5a5;
    text-align: center;
}

.small {
    color: #64748b;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    # Load Titanic dataset
    df = sns.load_dataset("titanic")

    # Remove unused columns
    columns_to_drop = [
        "deck",
        "embark_town",
        "alive",
        "class",
        "who",
        "adult_male",
        "alone"
    ]

    df.drop(
        columns=columns_to_drop,
        inplace=True
    )

    # Fill missing age values
    df["age"] = df["age"].fillna(
        df["age"].mean()
    )

    # Remove rows with missing embarked values
    df.dropna(
        subset=["embarked"],
        inplace=True
    )

    # =====================================================
    # ENCODE SEX
    # Female = 0
    # Male = 1
    # =====================================================

    df["sex"] = df["sex"].map({
        "female": 0,
        "male": 1
    })

    # =====================================================
    # ENCODE EMBARKED
    # C = 0
    # Q = 1
    # S = 2
    # =====================================================

    df["embarked"] = df["embarked"].map({
        "C": 0,
        "Q": 1,
        "S": 2
    })

    # =====================================================
    # CONVERT NUMERIC COLUMNS
    # =====================================================

    df["pclass"] = df["pclass"].astype(int)
    df["sex"] = df["sex"].astype(int)
    df["age"] = df["age"].round().astype(int)
    df["sibsp"] = df["sibsp"].astype(int)
    df["parch"] = df["parch"].astype(int)
    df["fare"] = df["fare"].round().astype(int)
    df["embarked"] = df["embarked"].astype(int)
    df["survived"] = df["survived"].astype(int)

    return df


# =========================================================
# TRAIN SVM MODEL
# =========================================================

@st.cache_resource
def train_model():

    df = load_data()

    # Exact feature order
    feature_order = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    # Features
    X = df[feature_order]

    # Target
    y = df["survived"]

    # StandardScaler
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # SVM Model
    model = SVC()

    # Train model
    model.fit(
        X_scaled,
        y
    )

    return (
        model,
        scaler,
        df,
        feature_order
    )


# =========================================================
# LOAD MODEL
# =========================================================

model, scaler, df, feature_order = train_model()


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <h1>🚢 Titanic Survival Predictor</h1>

    <p>
        Machine Learning project using Support Vector Machine (SVM)
    </p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# DASHBOARD
# =========================================================

total = len(df)

survived = int(
    df["survived"].sum()
)

not_survived = total - survived

survival_rate = (
    survived / total * 100
)


# =========================================================
# DASHBOARD METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Passengers",
    total
)

c2.metric(
    "Survived",
    survived
)

c3.metric(
    "Did Not Survive",
    not_survived
)

c4.metric(
    "Survival Rate",
    f"{survival_rate:.1f}%"
)


# =========================================================
# SURVIVAL OVERVIEW
# =========================================================

st.markdown("### 📊 Survival Overview")

left, right = st.columns(2)


# =========================================================
# GENDER SURVIVAL RATE
# =========================================================

with left:

    gender_rate = (
        df.groupby("sex")["survived"]
        .mean() * 100
    )

    gender_table = pd.DataFrame({
        "Group": [
            "Female",
            "Male"
        ],
        "Survival Rate (%)": [
            gender_rate.get(0, 0),
            gender_rate.get(1, 0)
        ]
    })

    st.bar_chart(
        gender_table.set_index("Group")
    )


# =========================================================
# OVERALL SURVIVAL
# =========================================================

with right:

    overview = pd.DataFrame({
        "Status": [
            "Survived",
            "Did Not Survive"
        ],
        "Passengers": [
            survived,
            not_survived
        ]
    })

    st.bar_chart(
        overview.set_index("Status")
    )


# =========================================================
# PREDICTION SECTION
# =========================================================

st.markdown(
    "### 🔮 Predict Passenger Survival"
)

st.caption(
    "Enter passenger details to predict survival using the trained SVM model."
)


# =========================================================
# PREDICTION FORM
# =========================================================

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(3)

    # =====================================================
    # PASSENGER INFORMATION
    # =====================================================

    with col1:

        pclass = st.selectbox(
            "Passenger Class",
            [1, 2, 3],
            help=(
                "1 = First class, "
                "2 = Second class, "
                "3 = Third class"
            )
        )

        sex = st.selectbox(
            "Sex",
            [
                "Female",
                "Male"
            ]
        )

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=100,
            value=30,
            step=1
        )

    # =====================================================
    # FAMILY AND FARE INFORMATION
    # =====================================================

    with col2:

        sibsp = st.number_input(
            "Siblings / Spouses Aboard (sibsp)",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        parch = st.number_input(
            "Parents / Children Aboard (parch)",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        fare = st.number_input(
            "Fare",
            min_value=0.0,
            max_value=600.0,
            value=32.0,
            step=1.0
        )

    # =====================================================
    # EMBARKATION INFORMATION
    # =====================================================

    with col3:

        embarked = st.selectbox(
            "Port of Embarkation",
            [
                "Cherbourg (C)",
                "Queenstown (Q)",
                "Southampton (S)"
            ]
        )

        st.markdown(
            "**Model:** SVC (Support Vector Classifier)"
        )

        st.markdown(
            "**Preprocessing:** StandardScaler"
        )

        st.markdown(
            "**Features:** 7"
        )

        st.markdown(
            "**Target:** survived"
        )

    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    submitted = st.form_submit_button(
        "🚀 Predict Survival",
        use_container_width=True
    )


# =========================================================
# MAKE PREDICTION
# =========================================================

if submitted:

    # =====================================================
    # ENCODE SEX
    # =====================================================

    sex_value = {
        "Female": 0,
        "Male": 1
    }[sex]

    # =====================================================
    # ENCODE EMBARKED
    # =====================================================

    embarked_value = {
        "Cherbourg (C)": 0,
        "Queenstown (Q)": 1,
        "Southampton (S)": 2
    }[embarked]

    # =====================================================
    # CREATE INPUT DATA
    # =====================================================

    input_data = pd.DataFrame([{
        "pclass": int(pclass),
        "sex": int(sex_value),
        "age": int(age),
        "sibsp": int(sibsp),
        "parch": int(parch),
        "fare": int(round(fare)),
        "embarked": int(embarked_value)
    }])

    # =====================================================
    # EXACT FEATURE ORDER
    # =====================================================

    input_data = input_data[
        feature_order
    ]

    # =====================================================
    # SCALE INPUT
    # =====================================================

    input_scaled = scaler.transform(
        input_data
    )

    # =====================================================
    # MAKE PREDICTION
    # =====================================================

    prediction = int(
        model.predict(input_scaled)[0]
    )

    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.markdown("---")

    if prediction == 1:

        st.markdown("""
        <div class="result-survive">

            <h2>✅ Predicted: Survived</h2>

            <p>
                The SVM model predicts that this passenger
                would survive.
            </p>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-not">

            <h2>❌ Predicted: Did Not Survive</h2>

            <p>
                The SVM model predicts that this passenger
                would not survive.
            </p>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# ABOUT PROJECT
# =========================================================

st.markdown("---")

with st.expander("ℹ️ About this project"):

    st.write("""
    This Streamlit application follows the ML workflow
    used in the Titanic SVM project:

    1. Load Titanic dataset using Seaborn.
    2. Remove unused columns.
    3. Remove the "alone" feature.
    4. Fill missing age values with the mean.
    5. Remove rows with missing embarked values.
    6. Encode categorical values.
    7. Separate features (X) and target (y).
    8. Standardize features using StandardScaler.
    9. Train an SVC model.
    10. Take passenger input from the user.
    11. Apply the same feature order.
    12. Scale the input using StandardScaler.
    13. Use the trained SVM model to predict survival.
    """)


# =========================================================
# FOOTER
# =========================================================

st.caption(
    "Titanic Survival Prediction • SVM Machine Learning Project"
)

