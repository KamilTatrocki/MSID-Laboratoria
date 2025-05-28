import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("../data/players_22.csv", dtype={25: str, 108: str})
data = data.drop(columns=[
    'player_url',
    'player_face_url',
    'club_logo_url',
    'club_flag_url',
    'nation_logo_url',
    'nation_flag_url'
])
Yvalue = "overall"     # "value_eur" #"overall"
testSize = 0.2
randomnumber = 21
data = data[data[Yvalue].notna()]
X = data.drop(Yvalue, axis=1)
y = data[Yvalue]


numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomnumber)

# Model regrsji liniowej (oryginalny)
lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regression', LinearRegression())
])

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lr)
r2 = r2_score(y_test, y_pred_lr)

print("Mean Squared Error (MSE) for Linear Regression:", mse)
print("R2 score for Linear Regression:", r2)

y_pred_lr = lr.predict(X_train)
mse = mean_squared_error(y_train, y_pred_lr)
r2 = r2_score(y_train, y_pred_lr)

print("Mean Squared Error (MSE) for Linear Regression(Training dataset):", mse)
print("R2 score for Linear Regression(Training dataset):", r2)

# Model regresji liniowej z cechami welomianowymi
lr_poly = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_regression, k=50)),
    ('polynomial', PolynomialFeatures(degree=2)),
    ('regression', LinearRegression())
])

lr_poly.fit(X_train, y_train)
y_pred_lr_poly = lr_poly.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lr_poly)
r2 = r2_score(y_test, y_pred_lr_poly)

print("Mean Squared Error (MSE) for Polynomial Regression:", mse)
print("R2 score for Polynomial Regression:", r2)

y_pred_lr_poly = lr_poly.predict(X_train)
mse = mean_squared_error(y_train, y_pred_lr_poly)
r2 = r2_score(y_train, y_pred_lr_poly)

print("Mean Squared Error (MSE) for Polynomial Regression(Training dataset):", mse)
print("R2 score for Polynomial Regression(Training dataset):", r2)

