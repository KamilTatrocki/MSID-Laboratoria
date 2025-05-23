import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge


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


# Model regresji liniowej z cechami wielomianowymi
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
linear_model = lr_poly.named_steps['regression']
print("\nWartości wag (Linear Regression):")
print(", ".join([f"{w:.4f}" for w in linear_model.coef_]))

# Model regresji liniowej z cechami wielomianowymi plus Lasso
lr_poly = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_regression, k=50)),
    ('polynomial', PolynomialFeatures(degree=2)),
    ('regression', Lasso(alpha=0.01, max_iter=10000))
])

lr_poly.fit(X_train, y_train)
y_pred_lr_poly = lr_poly.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lr_poly)
r2 = r2_score(y_test, y_pred_lr_poly)

print("Lasso Mean Squared Error (MSE) for Polynomial Regression:", mse)
print("Lasso R2 score for Polynomial Regression:", r2)

y_pred_lr_poly = lr_poly.predict(X_train)
mse = mean_squared_error(y_train, y_pred_lr_poly)
r2 = r2_score(y_train, y_pred_lr_poly)

print("Lasso Mean Squared Error (MSE) for Polynomial Regression(Training dataset):", mse)
print("Lasso R2 score for Polynomial Regression(Training dataset):", r2)


lasso_model = lr_poly.named_steps['regression']
print("\nWartości wag (Lasso) :")
print(", ".join([f"{w:.4f}" for w in lasso_model.coef_]))


# Model regresji liniowej z cechami wielomianowymi plus Ridge
lr_poly_ridge = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(f_regression, k=50)),
    ('polynomial', PolynomialFeatures(degree=2)),
    ('regression', Ridge(alpha=0.1, max_iter=10000))
])

lr_poly_ridge.fit(X_train, y_train)
y_pred_lr_poly_ridge = lr_poly_ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred_lr_poly_ridge)
r2 = r2_score(y_test, y_pred_lr_poly_ridge)

print("Ridge Mean Squared Error (MSE) for Polynomial Regression:", mse)
print("Ridge R2 score for Polynomial Regression:", r2)

y_pred_lr_poly_ridge = lr_poly_ridge.predict(X_train)
mse = mean_squared_error(y_train, y_pred_lr_poly_ridge)
r2 = r2_score(y_train, y_pred_lr_poly_ridge)

print("Ridge Mean Squared Error (MSE) for Polynomial Regression(Training dataset):", mse)
print("Ridge R2 score for Polynomial Regression(Training dataset):", r2)

ridge_model = lr_poly_ridge.named_steps['regression']
print("\nWartości wag (Ridge) :")
print(", ".join([f"{w:.4f}" for w in ridge_model.coef_]))
