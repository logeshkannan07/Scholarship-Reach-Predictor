# scholarship_app_safe.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Load Dataset ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("TN_Scholarship_Reach_REALISTIC.csv")
    # Derived features
    df["income_to_infra"] = df["avg_family_income"] / df["school_infrastructure_index"].replace(0,1)
    df["awareness_index"] = (df["literacy_rate"] * df["schools_with_internet_percent"]) / 100
    return df

df = load_data()

# ---------------- Prepare Model Features ----------------
feature_cols = [
    "avg_family_income",
    "literacy_rate",
    "female_ratio",
    "rural_population_percent",
    "num_students",
    "schools_with_computer_lab_percent",
    "schools_with_internet_percent",
    "school_infrastructure_index",
    "income_to_infra",
    "awareness_index"
]

X = df[feature_cols]
y = df["scholarship_reach_percent"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}
trained_models = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    trained_models[name] = m

# ---------------- Dummy Areas ----------------
dummy_areas = {
    "Chennai": ["Adyar", "T. Nagar", "Velachery"],
    "Madurai": ["Simmakkal", "Alanganallur", "Thiruparankundram"],
    "Coimbatore": ["RS Puram", "Peelamedu", "Gandhipuram"]
}

# ---------------- Streamlit GUI ----------------
st.title("🎓 TN Scholarship Reach Predictor")
model_choice = st.selectbox("Choose Model", list(trained_models.keys()))
district = st.selectbox("Select District", df["district"].unique())
area = st.selectbox("Select Area / City", dummy_areas.get(district, ["Area 1", "Area 2"]))
st.write(f"Selected Area: **{area}, {district}**")

# Numeric Inputs pre-filled with district data
district_data = df[df["district"] == district].iloc[0]
avg_income = st.number_input("Average Family Income", value=float(district_data["avg_family_income"]))
literacy_rate = st.slider("Literacy Rate (%)", 0.0, 100.0, float(district_data["literacy_rate"]))
female_ratio = st.slider("Female Ratio", 800.0, 1100.0, float(district_data["female_ratio"]))
rural_percent = st.slider("Rural Population (%)", 0.0, 100.0, float(district_data["rural_population_percent"]))
num_students = st.number_input("Number of Students", value=int(district_data["num_students"]))
comp_lab_percent = st.slider("Schools with Computer Lab (%)", 0.0, 100.0, float(district_data["schools_with_computer_lab_percent"]))
internet_percent = st.slider("Schools with Internet (%)", 0.0, 100.0, float(district_data["schools_with_internet_percent"]))
infra_index = st.slider("School Infrastructure Index", 0.0, 100.0, float(district_data["school_infrastructure_index"]))

# Derived Features
income_to_infra = avg_income / (infra_index if infra_index != 0 else 1)
awareness_index = (literacy_rate * internet_percent) / 100

# Predict Button
if st.button("Predict Scholarship Reach"):
    features_array = np.array([[avg_income, literacy_rate, female_ratio, rural_percent,
                                num_students, comp_lab_percent, internet_percent,
                                infra_index, income_to_infra, awareness_index]])
    features_scaled = scaler.transform(features_array)
    model = trained_models[model_choice]
    pred = model.predict(features_scaled)[0]
    pred = float(np.clip(pred, 0, 100))
    st.success(f"🏆 Predicted Scholarship Reach: {pred:.2f}%")

# Optional: Heatmap
if st.checkbox("Show Correlation Heatmap"):
    st.subheader("Correlation Heatmap")
    corr = df[feature_cols + ["scholarship_reach_percent"]].corr()
    fig, ax = plt.subplots(figsize=(9,7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Optional: Model Performance
if st.checkbox("Show Model Performance"):
    res = []
    for name, m in trained_models.items():
        y_pred = m.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        res.append({"Model": name, "RMSE": round(rmse,2), "R²": round(r2,2)})
    st.subheader("Model Performance")
    st.table(pd.DataFrame(res))
