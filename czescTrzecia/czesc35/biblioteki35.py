import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer
import numpy as np

data = pd.read_csv("../data/players_22.csv", dtype={25: str, 108: str})
data = data.drop(columns=[
    'player_url', 'player_face_url', 'club_logo_url',
    'club_flag_url', 'nation_logo_url', 'nation_flag_url'
])

Yvalue = "overall"  # lub "value_eur"
testSize = 0.20
randomnumber = 21

data = data[data[Yvalue].notna()]
X, y = data.drop(Yvalue, axis=1), data[Yvalue]

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns

num_tf = Pipeline([('imp', SimpleImputer(strategy='mean')),
                   ('sc', StandardScaler())])

cat_tf = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                   ('ohe', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num', num_tf, num_cols),
    ('cat', cat_tf, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=testSize, random_state=randomnumber
)

param_grid_rf = {
    'reg__n_estimators': [100, 200, 400],
    'reg__max_depth': [None, 10, 20],
}

param_grid_svr = {
    'reg__kernel': ['rbf'],
    'reg__C': [1, 10, 100],
    'reg__epsilon': [0.1, 0.2, 0.5]
}



def tune_model(regressor, param_grid, name):
    pipe = Pipeline([
        ('prep', preprocessor),
        ('reg', regressor)
    ])

    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=3,
        scoring={'MSE': mse_scorer, 'R2': 'r2'},
        refit='R2',
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_train, y_train)

    best_params = grid.best_params_

    y_pred_test = grid.predict(X_test)
    y_pred_train = grid.predict(X_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    print(f"\n{name}")
    print("-" * len(name))
    print("Najlepsze parametry:", best_params)
    print(f"R² (train) = {r2_train:.3f} | MSE (train) = {mse_train:.1f}")
    print(f"R² (test)  = {r2_test:.3f} | MSE (test)  = {mse_test:.1f}")
    return grid



grid_rf = tune_model(RandomForestRegressor(random_state=randomnumber),
                     param_grid_rf, name="Random Forest Regressor")

grid_svr = tune_model(SVR(),
                      param_grid_svr, name="Support Vector Regressor")



