import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

RANDOM_STATE   = 42
TARGET_COL     = "overall"
TEST_SIZE      = 0.2
LEARNING_RATE  = 0.01
EPOCHS         = 1000
BATCH_SIZE     = 512
PRINT_EVERY    = 100

MAX_OHE_CARD   = 50


csv = pd.read_csv("data/players_22.csv", dtype={25: str, 108: str})
csv = csv.drop(columns=[
    "player_url", "player_face_url", "club_logo_url",
    "club_flag_url", "nation_logo_url", "nation_flag_url"
])

csv = csv[csv[TARGET_COL].notna()]             # usuń wiersze bez y
X_full = csv.drop(columns=[TARGET_COL])
y_full = csv[TARGET_COL].values.reshape(-1, 1)


num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()

cat_candidates = X_full.select_dtypes(exclude=[np.number]).columns
cat_cols = [
    c for c in cat_candidates
    if X_full[c].nunique(dropna=True) <= MAX_OHE_CARD
]



X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
)


numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])



ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)


categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     ohe)
])

preprocessor = ColumnTransformer(
    [("num", numeric_pipe, num_cols),
     ("cat", categorical_pipe, cat_cols)]
)

X_train = preprocessor.fit_transform(X_train_df).astype(np.float32)
X_test  = preprocessor.transform(X_test_df).astype(np.float32)



X_train64 = X_train.astype(np.float64, copy=False)
y_train64 = y_train.astype(np.float64, copy=False)
X_test64  = X_test .astype(np.float64, copy=False)


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])

def closed_form_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_b = add_bias(X)
    return np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y



def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return add_bias(X) @ w







w_closed      = closed_form_fit(X_train64, y_train64)
y_pred_closed = predict(X_test64, w_closed)


def minibatches(X, y, batch_size, rng):
    idxs = np.arange(X.shape[0])
    rng.shuffle(idxs)
    for start in range(0, len(idxs), batch_size):
        batch_idx = idxs[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]

def gradient_descent_fit(
        X, y, lr=0.001, epochs=1_000, batch_size=256):

    rng = np.random.default_rng(RANDOM_STATE)
    n_samples, n_features = X.shape
    w = rng.normal(scale=0.01, size=(n_features + 1, 1)).astype(X.dtype)

    for epoch in range(epochs + 1):
        for Xb, yb in minibatches(X, y, batch_size, rng=rng):
            Xb_b = add_bias(Xb)
            grad = 2 / len(yb) * Xb_b.T @ (Xb_b @ w - yb)
            w   -= lr * grad

        if  epoch % PRINT_EVERY == 0:
            mse = mean_squared_error(y, predict(X, w))
            print(f"epoch {epoch:4d}/{epochs}  MSE={mse:.4f}")

    return w

w_gd      = gradient_descent_fit(
    X_train, y_train, lr=LEARNING_RATE,
    epochs=EPOCHS, batch_size=BATCH_SIZE
)
y_pred_gd = predict(X_test, w_gd)


linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_sklearn = linreg.predict(X_test)


def report(name, y_true, y_pred):
    print(f"{name:<16s}| MSE = {mean_squared_error(y_true, y_pred):10.4f}"
          f" | R² = {r2_score(y_true, y_pred):7.4f}")

report("Closed-form",     y_test, y_pred_closed)
report("GradientDescent", y_test, y_pred_gd)
report("sklearn",         y_test, y_pred_sklearn)
