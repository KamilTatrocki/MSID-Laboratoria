import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import  RandomForestRegressor, VotingRegressor, StackingRegressor
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
Yvalue = "overall"
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
        ('cat', categorical_transformer, categorical_cols)]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=randomnumber)

lr_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

rfr_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=randomnumber))
])

svr_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf'))
])

voting = VotingRegressor(estimators=[
    ('lr', lr_pipe),
    ('rfr', rfr_pipe),
    ('svr', svr_pipe)
])

voting.fit(X_train, y_train)
y_pred_voting_test = voting.predict(X_test)
y_pred_voting_train = voting.predict(X_train)

stacking = StackingRegressor(
    estimators=[
        ('lr', lr_pipe),
        ('rfr', rfr_pipe),
        ('svr', svr_pipe)
],
    final_estimator=LinearRegression()
)

stacking.fit(X_train, y_train)
y_pred_stacking_test = stacking.predict(X_test)
y_pred_stacking_train = stacking.predict(X_train)

metrics = {
    'Model': ['VotingRegressor Test', 'VotingRegressor Train', 'StackingRegressor Test', 'StackingRegressor Train'],
    'MSE': [
        mean_squared_error(y_test, y_pred_voting_test),
        mean_squared_error(y_train, y_pred_voting_train),
        mean_squared_error(y_test, y_pred_stacking_test),
        mean_squared_error(y_train, y_pred_stacking_train)
    ],
    'R2': [
        r2_score(y_test, y_pred_voting_test),
        r2_score(y_train, y_pred_voting_train),
        r2_score(y_test, y_pred_stacking_test),
        r2_score(y_train, y_pred_stacking_train)
    ]
}

results_df = pd.DataFrame(metrics)
player_names = X_test['short_name']
prediction_df = pd.DataFrame({
    'Player': player_names,
    'True Value': y_test,
    'Predicted Voting': y_pred_voting_test,
    'Predicted Stacking': y_pred_stacking_test
})
with pd.ExcelWriter('model_results.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Metrics', index=False)
    prediction_df.to_excel(writer, sheet_name='Predictions', index=False)
