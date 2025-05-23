import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

data = pd.read_csv("../data/players_22.csv", dtype={25: str, 108: str})
data = data.drop(columns=[
    'player_url',
    'player_face_url',
    'club_logo_url',
    'club_flag_url',
    'nation_logo_url',
    'nation_flag_url'
])
Yvalue = "work_rate"
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

# Random Forest Classifier
rfr = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=randomnumber)),
    ('classifier', RandomForestClassifier(random_state=randomnumber))
])

rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_test)
precision = precision_score(y_test, y_pred_rfr, average='weighted',zero_division=0)
recall = recall_score(y_test, y_pred_rfr, average='weighted',zero_division=0)
f1 = f1_score(y_test, y_pred_rfr, average='weighted',zero_division=0)

print("Random Forest Classifier (Test dataset):")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")



# Logistic Regression
lr = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=randomnumber)),
    ('classifier', LogisticRegression(max_iter=1000))
])

lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
precision = precision_score(y_test, y_pred_lr, average='weighted',zero_division=0)
recall = recall_score(y_test, y_pred_lr, average='weighted',zero_division=0)
f1 = f1_score(y_test, y_pred_lr, average='weighted',zero_division=0)

print("Logistic Regression (Test dataset):")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")



# SVC
svr = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=randomnumber)),
    ('classifier', SVC(kernel='rbf'))
])

svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
precision = precision_score(y_test, y_pred_svr, average='weighted',zero_division=0)
recall = recall_score(y_test, y_pred_svr, average='weighted',zero_division=0)
f1 = f1_score(y_test, y_pred_svr, average='weighted',zero_division=0)

print("SVC (Test dataset):")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

