import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import  RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error, r2_score


data = pd.read_csv("../data/players_22.csv", dtype={25: str, 108: str})
data = data.drop(columns=[
    'player_url',
    'player_face_url',
    'club_logo_url',
    'club_flag_url',
    'nation_logo_url',
    'nation_flag_url'
])
Yvalue =   "overall"     # "value_eur" #"overall"
testSize = 0.2
randomnumber=21
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

#nowe
from sklearn.model_selection import KFold, cross_validate
cv = KFold(n_splits=3, shuffle=True, random_state=randomnumber)
scoring = {
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2"
}

models = {
    "RandomForest": RandomForestRegressor(random_state=randomnumber),
    "LinearRegression": LinearRegression(),
    "SVR": SVR(kernel="rbf")
}

results = {}

for name, estimator in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", estimator)
    ])

    cv_res = cross_validate(
        pipe,
        X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )


    rmse_folds = -cv_res["test_RMSE"]
    r2_folds   =  cv_res["test_R2"]

    results[name] = {
        "rmse": rmse_folds,
        "r2":   r2_folds
    }
    print(name)
    print(
    f"RMSE foldy: {rmse_folds.round(2).tolist()} "
    f" średnia = {rmse_folds.mean():.2f}, "
    f"std = {rmse_folds.std():.2f}"
    )

    print(
    f" R2  foldy: {r2_folds.round(3).tolist()} "
    f" średnia = {r2_folds.mean():.3f}, "
    f"std = {r2_folds.std():.3f}"
    )
