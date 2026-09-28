
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
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f6f8fb;
}

/* =====================================================
   METRIC CARDS
   ===================================================== */

[data-testid="stMetric"] {
    background-color: white !important;
    padding: 18px !important;
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
}

/* Metric Label */
[data-testid="stMetricLabel"] {
    color: #374151 !important;
    font-weight: 600 !important;
}

/* Metric Value */
[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 800 !important;
    font-size: 30px !important;
}

/* Metric Delta */
[data-testid="stMetricDelta"] {
    color: #374151 !important;
}

/* Metric Value - all child elements */
[data-testid="stMetricValue"] * {
    color: #111827 !important;
}

/* Headings */
h1, h2, h3 {
    color: #111827;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TITANIC DATASET
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

    # Fill missing Age values
    df["age"] = df["age"].fillna(
        df["age"].mean()
    )

    # Remove rows where Embarked is missing
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
    # CONVERT DATA TYPES
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

    # Create SVM model
    model = SVC()

    # Train model
    model.fit(
        X_scaled,
        y
    )

    return model, scaler, df, feature_order


# =========================================================
# LOAD MODEL
# =========================================================

model, scaler, df, feature_order = train_model()


# =========================================================
# HEADER
# =========================================================

st.title("🚢 Titanic Survival Predictor")

st.write(
    "Machine Learning project using "
    "Support Vector Machine (SVM)"
)

st.markdown("---")


# =========================================================
# DASHBOARD CALCULATIONS
# =========================================================

total_passengers = len(df)

total_survived = int(
    df["survived"].sum()
)

total_not_survived = (
    total_passengers - total_survived
)

survival_rate = (
    total_survived / total_passengers
) * 100


# =========================================================
# TITANIC DATASET OVERVIEW
# =========================================================

st.subheader("📊 Titanic Dataset Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Passengers",
        total_passengers
    )


with col2:

    st.metric(
        "Survived",
        total_survived
    )


with col3:

    st.metric(
        "Did Not Survive",
        total_not_survived
    )


with col4:

    st.metric(
        "Survival Rate",
        f"{survival_rate:.1f}%"
    )


# =========================================================
# SURVIVAL OVERVIEW
# =========================================================

st.markdown("---")

st.subheader("📈 Survival Overview")

left_column, right_column = st.columns(2)


# =========================================================
# GENDER SURVIVAL
# =========================================================

with left_column:

    st.write("### 👨‍👩‍👧 Survival Rate by Gender")

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

with right_column:

    st.write("### 🚢 Overall Survival")

    overview = pd.DataFrame({
        "Status": [
            "Survived",
            "Did Not Survive"
        ],
        "Passengers": [
            total_survived,
            total_not_survived
        ]
    })

    st.bar_chart(
        overview.set_index("Status")
    )


# =========================================================
# PREDICTION SECTION
# =========================================================

st.markdown("---")

st.subheader("🔮 Predict Passenger Survival")

st.write(
    "Enter passenger details to predict survival "
    "using the trained SVM model."
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
                "1 = First Class | "
                "2 = Second Class | "
                "3 = Third Class"
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
    # FAMILY AND FARE
    # =====================================================

    with col2:

        sibsp = st.number_input(
            "Siblings / Spouses Aboard",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )

        parch = st.number_input(
            "Parents / Children Aboard",
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
    # EMBARKATION AND MODEL INFO
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

        st.info(
            "Model: SVC\n\n"
            "Preprocessing: StandardScaler\n\n"
            "Features: 7\n\n"
            "Target: survived"
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
    # DISPLAY PREDICTION
    # =====================================================

    st.markdown("---")

    st.subheader("🎯 Prediction Result")


    if prediction == 1:

        st.success(
            "✅ Predicted: Survived"
        )

        st.write(
            "The SVM model predicts that "
            "this passenger would survive."
        )


    else:

        st.error(
            "❌ Predicted: Did Not Survive"
        )

        st.write(
            "The SVM model predicts that "
            "this passenger would not survive."
        )


# =========================================================
# ABOUT PROJECT
# =========================================================

st.markdown("---")

with st.expander("ℹ️ About this Project"):

    st.write("""
    This Streamlit application uses a Support Vector Machine
    (SVM) to predict Titanic passenger survival.

    Machine Learning Workflow:

    1. Load Titanic dataset using Seaborn.
    2. Remove unused columns.
    3. Remove the "alone" feature.
    4. Handle missing Age values.
    5. Remove missing Embarked values.
    6. Encode categorical variables.
    7. Select 7 features.
    8. Apply StandardScaler.
    9. Train the SVC model.
    10. Take passenger information from the user.
    11. Apply the same feature order.
    12. Scale the user input.
    13. Predict passenger survival.
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Titanic Survival Prediction • "
    "SVM Machine Learning Project"
)

