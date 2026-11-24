import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("data.csv")

y = data["Price"]

X = data[
    [
        "Mileage", "Cylinder", "Liter", "Doors", "Cruise", "Sound", "Leather",
        "Make", "Model", "Trim", "Type"
    ]
]

numeric_features = ["Mileage", "Cylinder", "Liter", "Doors", "Cruise", "Sound", "Leather"]
categorical_features = ["Make", "Model", "Trim", "Type"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model a: Baseline Linear Regression only numeric, no scaling

X_num = data[numeric_features]
y_num = data["Price"]

Xn_train, Xn_test, yn_train, yn_test = train_test_split(
    X_num, y_num, test_size=0.2, random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(Xn_train, yn_train)
yn_pred = baseline_model.predict(Xn_test)

r2_baseline = r2_score(yn_test, yn_pred)
print("R2 Score (Baseline - numeric only, no scaling):", r2_baseline)


# Model b) Linear Regression  scaling and one-hot

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model_scaled_ohe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

model_scaled_ohe.fit(X_train, y_train)
y_pred = model_scaled_ohe.predict(X_test)

r2_scaled_ohe = r2_score(y_test, y_pred)
print("R2 Score (Linear + Scaling + OneHot):", r2_scaled_ohe)


# Model c) Polynomial Features for numeric degree=2 and one-hot

preprocess_poly = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False))
        ]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model_poly = Pipeline(steps=[
    ("preprocess", preprocess_poly),
    ("regressor", LinearRegression())
])

model_poly.fit(X_train, y_train)
y_pred_poly = model_poly.predict(X_test)

r2_poly = r2_score(y_test, y_pred_poly)
print("R2 Score (Polynomial degree=2 + OneHot):", r2_poly)

