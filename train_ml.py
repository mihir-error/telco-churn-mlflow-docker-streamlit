import streamlit as st
import pandas as pd
import joblib
import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_engineering import FeatureEngineer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import os

st.title("📊 Telco Churn Predictor & Business KPI Dashboard")

# Paths
MASTER_DATA_PATH = "data/master_dataset.csv"
PIPELINE_PATH = "models/saved_pipeline.pkl"

# Ensure data folder exists
os.makedirs("data", exist_ok=True)
if not os.path.isdir("data"):
    os.makedirs("data")


# =========================
# 1️⃣ Upload new dataset
# =========================
uploaded_file = st.file_uploader("Upload a CSV file for training/prediction", type=["csv"])
if uploaded_file is not None:
    new_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded dataset:", new_df.head())

    # Append to master dataset
    if os.path.exists(MASTER_DATA_PATH):
        master_df = pd.read_csv(MASTER_DATA_PATH)
        master_df = pd.concat([master_df, new_df], ignore_index=True)
    else:
        master_df = new_df

    master_df.to_csv(MASTER_DATA_PATH, index=False)
    st.success(f"Master dataset updated! Total rows: {len(master_df)}")

    # Select target column
    target_col = st.selectbox("Select target column", master_df.columns)

# =========================
# 2️⃣ Load pipeline if exists
# =========================
if os.path.exists(PIPELINE_PATH):
    pipeline = joblib.load(PIPELINE_PATH)
    st.info("Loaded existing pipeline")
else:
    pipeline = None
    st.warning("No trained pipeline found. Please upload data and retrain first.")

# =========================
# 3️⃣ Retrain Model Button
# =========================
if uploaded_file is not None and st.button("Retrain Model"):
    df = master_df.copy()
    df[target_col] = df[target_col].map({'No': 0, 'Yes': 1})
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Categorical and numeric features
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    categorical_features = [col for col in categorical_features if col != 'customerID']

    numeric_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'tenure_bin_code', 'is_new_customer',
        'total_per_month', 'high_monthly_flag'
    ]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
        ],
        remainder='drop'
    )

    # Create new pipeline if none exists
    rf_model = RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='log2',
        max_depth=10,
        bootstrap=True,
        random_state=42,
        class_weight={0:1.0, 1:2.0}
    )

    pipeline = Pipeline(steps=[
        ('fe', FeatureEngineer()),
        ('preprocess', preprocessor),
        ('model', rf_model)
    ])

    # Fit pipeline
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "model")

        st.success(f"✅ Model retrained! Test Accuracy: {acc:.2f}")
        joblib.dump(pipeline, PIPELINE_PATH)
        st.info("Updated pipeline saved!")

# =========================
# 4️⃣ Churn Prediction & KPI (Independent of retraining)
# =========================
if pipeline is not None and uploaded_file is not None:
    df = master_df.copy()
    df[target_col] = df[target_col].map({'No': 0, 'Yes': 1})
    X = df.drop(columns=[target_col])

    # Predict churn probabilities
    df_scores = X.copy()
    df_scores['Churn_Prob'] = pipeline.predict_proba(X)[:, 1]
    df_scores['Risk_Level'] = pd.cut(df_scores['Churn_Prob'], bins=[0, 0.3, 0.7, 1],
                                     labels=['Low', 'Medium', 'High'])

    # High-risk customers
    risk_threshold = 0.7
    high_risk = df_scores[df_scores['Churn_Prob'] >= risk_threshold]

    # Potential revenue saved
    revenue_saved = high_risk['MonthlyCharges'].sum() * 12

    # Discount adjustment slider
    discount = st.slider("Expected avg discount (%)", 0, 50, 10)
    adjusted_revenue_saved = revenue_saved * (1 - discount / 100)

    # Display KPI
    st.subheader("📈 Business KPI")
    st.write(f"🚀 Potential annual revenue saved: ${revenue_saved:.2f}")
    st.write(f"💡 Adjusted potential revenue saved after {discount}% discount: ${adjusted_revenue_saved:.2f}")

    # Show high-risk customer table
    st.subheader("High-Risk Customers")
    st.dataframe(high_risk[['customerID', 'Churn_Prob', 'Risk_Level', 'MonthlyCharges']])
