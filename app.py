
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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS - ADVANCED UI
# =========================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

h1, h2, h3, h4 {
    color: #111827 !important;
}

p, label {
    color: #374151;
}


/* =========================================================
   HERO HEADER
   ========================================================= */

.hero {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e3a5f 55%,
        #2563eb 100%
    );
    padding: 38px 42px;
    border-radius: 24px;
    margin-bottom: 30px;
    box-shadow: 0 12px 35px rgba(15, 23, 42, 0.18);
}

.hero-title {
    color: white !important;
    font-size: 44px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: #dbeafe !important;
    font-size: 18px;
    margin-top: 10px;
    margin-bottom: 0;
}

.hero-badge {
    display: inline-block;
    margin-top: 20px;
    padding: 7px 14px;
    border-radius: 20px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: white !important;
    font-size: 13px;
    font-weight: 600;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    font-size: 26px;
    font-weight: 750;
    color: #111827 !important;
    margin-top: 10px;
    margin-bottom: 18px;
}

.section-description {
    color: #6b7280 !important;
    font-size: 15px;
    margin-top: -10px;
    margin-bottom: 22px;
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

[data-testid="stMetric"] {
    background: white !important;
    padding: 22px !important;
    border-radius: 18px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.07) !important;
    min-height: 125px;
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.11) !important;
}

[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 650 !important;
    font-size: 14px !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-weight: 850 !important;
    font-size: 32px !important;
}

[data-testid="stMetricValue"] * {
    color: #111827 !important;
}


/* =========================================================
   CHART CARDS
   ========================================================= */

.chart-card {
    background: white;
    padding: 22px 24px 16px 24px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
    min-height: 390px;
}

.chart-title {
    font-size: 18px;
    font-weight: 750;
    color: #111827;
    margin-bottom: 4px;
}

.chart-description {
    color: #6b7280;
    font-size: 13px;
    margin-bottom: 15px;
}


/* =========================================================
   PREDICTION CONTAINER
   ========================================================= */

.prediction-card {
    background: white;
    padding: 28px;
    border-radius: 22px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.07);
}


/* =========================================================
   INPUT FIELDS
   ========================================================= */

div[data-baseweb="select"] > div {
    border-radius: 10px !important;
}

input {
    border-radius: 10px !important;
}


/* =========================================================
   BUTTON
   ========================================================= */

.stFormSubmitButton button {
    background: linear-gradient(
        135deg,
        #2563eb,
        #1d4ed8
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 13px 25px !important;
    font-size: 16px !important;
    font-weight: 750 !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
}

.stFormSubmitButton button:hover {
    background: linear-gradient(
        135deg,
        #1d4ed8,
        #1e40af
    ) !important;
    transform: translateY(-1px);
}


/* =========================================================
   MODEL INFO CARD
   ========================================================= */

.model-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    padding: 22px;
    border-radius: 16px;
    margin-top: 8px;
}

.model-title {
    color: #1e3a8a;
    font-size: 17px;
    font-weight: 750;
    margin-bottom: 14px;
}

.model-row {
    display: flex;
    justify-content: space-between;
    padding: 9px 0;
    border-bottom: 1px solid #e5e7eb;
    font-size: 14px;
}

.model-row:last-child {
    border-bottom: none;
}

.model-label {
    color: #64748b;
}

.model-value {
    color: #111827;
    font-weight: 700;
}


/* =========================================================
   PREDICTION RESULT
   ========================================================= */

.result-survived {
    background: #ecfdf5;
    border: 2px solid #10b981;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin-top: 18px;
}

.result-not-survived {
    background: #fef2f2;
    border: 2px solid #ef4444;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    margin-top: 18px;
}

.result-icon {
    font-size: 48px;
}

.result-title {
    font-size: 28px;
    font-weight: 850;
    margin: 8px 0;
}

.result-text {
    font-size: 15px;
    color: #475569;
}


/* =========================================================
   INFO CARDS
   ========================================================= */

.info-card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
}

.info-title {
    font-size: 18px;
    font-weight: 750;
    color: #111827;
    margin-bottom: 10px;
}


/* =========================================================
   EXPANDER
   ========================================================= */

.streamlit-expanderHeader {
    font-weight: 700 !important;
    color: #111827 !important;
}


/* =========================================================
   DIVIDER
   ========================================================= */

hr {
    margin-top: 28px !important;
    margin-bottom: 28px !important;
    border-color: #e5e7eb !important;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    text-align: center;
    padding: 25px 10px;
    color: #64748b;
    font-size: 13px;
}

.footer-main {
    font-weight: 700;
    color: #334155;
    font-size: 14px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 768px) {

    .hero {
        padding: 28px 24px;
    }

    .hero-title {
        font-size: 32px;
    }

    .hero-subtitle {
        font-size: 15px;
    }

    [data-testid="stMetricValue"] {
        font-size: 26px !important;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TITANIC DATASET
# =========================================================

@st.cache_data
def load_data():

    df = sns.load_dataset("titanic")

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

    # Handle missing Age
    df["age"] = df["age"].fillna(
        df["age"].mean()
    )

    # Remove missing Embarked
    df.dropna(
        subset=["embarked"],
        inplace=True
    )

    # Encode Sex
    df["sex"] = df["sex"].map({
        "female": 0,
        "male": 1
    })

    # Encode Embarked
    df["embarked"] = df["embarked"].map({
        "C": 0,
        "Q": 1,
        "S": 2
    })

    # Data types
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

    feature_order = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    X = df[feature_order]
    y = df["survived"]

    # StandardScaler
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # SVM
    model = SVC()

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
# HEADER / HERO
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        🚢 Titanic Survival Predictor
    </div>

    <div class="hero-subtitle">
        Machine Learning application powered by
        Support Vector Machine (SVM)
    </div>

    <div class="hero-badge">
        🤖 SVM &nbsp; • &nbsp; StandardScaler &nbsp; • &nbsp; 7 Features
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# DATASET CALCULATIONS
# =========================================================

total_passengers = len(df)

total_survived = int(
    df["survived"].sum()
)

total_not_survived = (
    total_passengers - total_survived
)

survival_rate = (
    total_survived /
    total_passengers
) * 100


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">📊 Dataset Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Key statistics from the cleaned Titanic dataset.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "👥 Total Passengers",
        total_passengers
    )

with col2:
    st.metric(
        "✅ Survived",
        total_survived
    )

with col3:
    st.metric(
        "❌ Did Not Survive",
        total_not_survived
    )

with col4:
    st.metric(
        "📈 Survival Rate",
        f"{survival_rate:.1f}%"
    )


# =========================================================
# SURVIVAL ANALYTICS
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📈 Survival Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Explore survival patterns across the dataset.'
    '</div>',
    unsafe_allow_html=True
)

left_column, right_column = st.columns(2)


# =========================================================
# GENDER SURVIVAL
# =========================================================

with left_column:

    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">
            👨‍👩‍👧 Survival Rate by Gender
        </div>
        <div class="chart-description">
            Percentage of passengers who survived by gender.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        gender_table.set_index("Group"),
        height=300
    )


# =========================================================
# OVERALL SURVIVAL
# =========================================================

with right_column:

    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">
            🚢 Overall Survival
        </div>
        <div class="chart-description">
            Comparison between survived and non-survived passengers.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        overview.set_index("Status"),
        height=300
    )


# =========================================================
# PREDICTION SECTION
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔮 Predict Passenger Survival</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter passenger details and let the trained SVM model generate a prediction.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PREDICTION FORM
# =========================================================

with st.form("prediction_form"):

    col1, col2, col3 = st.columns(
        [1, 1, 1]
    )


    # =====================================================
    # PASSENGER DETAILS
    # =====================================================

    with col1:

        st.markdown(
            "### 👤 Passenger Details"
        )

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
    # FAMILY & FARE
    # =====================================================

    with col2:

        st.markdown(
            "### 👨‍👩‍👧 Family & Fare"
        )

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
    # EMBARKATION & MODEL
    # =====================================================

    with col3:

        st.markdown(
            "### 🚢 Travel & Model"
        )

        embarked = st.selectbox(
            "Port of Embarkation",
            [
                "Cherbourg (C)",
                "Queenstown (Q)",
                "Southampton (S)"
            ]
        )

        st.markdown("""
        <div class="model-card">

            <div class="model-title">
                🤖 Model Information
            </div>

            <div class="model-row">
                <span class="model-label">Algorithm</span>
                <span class="model-value">SVC</span>
            </div>

            <div class="model-row">
                <span class="model-label">Scaling</span>
                <span class="model-value">StandardScaler</span>
            </div>

            <div class="model-row">
                <span class="model-label">Features</span>
                <span class="model-value">7</span>
            </div>

            <div class="model-row">
                <span class="model-label">Target</span>
                <span class="model-value">Survived</span>
            </div>

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # BUTTON
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🚀 Predict Survival",
        use_container_width=True
    )


# =========================================================
# MAKE PREDICTION
# =========================================================

if submitted:

    # Encode Sex
    sex_value = {
        "Female": 0,
        "Male": 1
    }[sex]


    # Encode Embarked
    embarked_value = {
        "Cherbourg (C)": 0,
        "Queenstown (Q)": 1,
        "Southampton (S)": 2
    }[embarked]


    # Create input DataFrame
    input_data = pd.DataFrame([{

        "pclass": int(pclass),

        "sex": int(sex_value),

        "age": int(age),

        "sibsp": int(sibsp),

        "parch": int(parch),

        "fare": int(round(fare)),

        "embarked": int(embarked_value)

    }])


    # Exact feature order
    input_data = input_data[
        feature_order
    ]


    # Scale input
    input_scaled = scaler.transform(
        input_data
    )


    # Prediction
    prediction = int(
        model.predict(
            input_scaled
        )[0]
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )


    if prediction == 1:

        st.markdown("""
        <div class="result-survived">

            <div class="result-icon">
                ✅
            </div>

            <div class="result-title">
                Predicted: Survived
            </div>

            <div class="result-text">
                The SVM model predicts that this passenger
                would survive based on the entered information.
            </div>

        </div>
        """, unsafe_allow_html=True)


    else:

        st.markdown("""
        <div class="result-not-survived">

            <div class="result-icon">
                ❌
            </div>

            <div class="result-title">
                Predicted: Did Not Survive
            </div>

            <div class="result-text">
                The SVM model predicts that this passenger
                would not survive based on the entered information.
            </div>

        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # PASSENGER SUMMARY
    # =====================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "### 📋 Passenger Information"
    )

    summary1, summary2, summary3, summary4 = st.columns(4)

    with summary1:
        st.metric(
            "Class",
            pclass
        )

    with summary2:
        st.metric(
            "Sex",
            sex
        )

    with summary3:
        st.metric(
            "Age",
            age
        )

    with summary4:
        st.metric(
            "Fare",
            f"${fare:.0f}"
        )


# =========================================================
# MACHINE LEARNING WORKFLOW
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🧠 Machine Learning Workflow</div>',
    unsafe_allow_html=True
)

workflow1, workflow2, workflow3, workflow4 = st.columns(4)

with workflow1:

    st.markdown("""
    <div class="info-card">

        <div class="info-title">
            01. 📥 Data
        </div>

        Titanic dataset is loaded using
        Seaborn and prepared for machine learning.

    </div>
    """, unsafe_allow_html=True)


with workflow2:

    st.markdown("""
    <div class="info-card">

        <div class="info-title">
            02. 🧹 Preprocessing
        </div>

        Missing values are handled and
        categorical features are encoded.

    </div>
    """, unsafe_allow_html=True)


with workflow3:

    st.markdown("""
    <div class="info-card">

        <div class="info-title">
            03. ⚙️ Scaling
        </div>

        StandardScaler is applied to
        normalize the input features.

    </div>
    """, unsafe_allow_html=True)


with workflow4:

    st.markdown("""
    <div class="info-card">

        <div class="info-title">
            04. 🤖 Prediction
        </div>

        SVC model predicts whether the
        passenger survived or not.

    </div>
    """, unsafe_allow_html=True)


# =========================================================
# ABOUT PROJECT
# =========================================================

st.markdown("---")

with st.expander("ℹ️ About This Project"):

    st.markdown("""
    ### 🚢 Titanic Survival Prediction

    This Streamlit application uses a
    **Support Vector Machine (SVM)** machine learning
    model to predict Titanic passenger survival.

    ### 🔄 Machine Learning Pipeline

    **1. Load Dataset**
    
    Titanic dataset is loaded using Seaborn.

    **2. Data Cleaning**
    
    Unused columns are removed and missing values are handled.

    **3. Feature Engineering**
    
    Categorical variables such as Sex and Embarked
    are converted into numerical values.

    **4. Feature Selection**
    
    The model uses 7 features:

    - Passenger Class
    - Sex
    - Age
    - Siblings / Spouses
    - Parents / Children
    - Fare
    - Embarkation Port

    **5. Feature Scaling**
    
    StandardScaler is used before training the SVM model.

    **6. Model Training**
    
    SVC is trained using the processed Titanic dataset.

    **7. Prediction**
    
    User enters passenger information and the trained
    model predicts the survival outcome.
    """)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="footer">

    <div class="footer-main">
        🚢 Titanic Survival Predictor
    </div>

    <div>
        Machine Learning Project • Support Vector Machine (SVM)
    </div>

    <div style="margin-top: 8px;">
        Built with Python • Streamlit • Pandas • Scikit-learn
    </div>

</div>
""", unsafe_allow_html=True)

