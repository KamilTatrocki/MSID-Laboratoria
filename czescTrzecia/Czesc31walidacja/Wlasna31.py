import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

RANDOM_STATE   = 42
TARGET_COL     = "overall"
TEST_SIZE      = 0.2
LEARNING_RATE  = 0.01
EPOCHS         = 1000
BATCH_SIZE     = 512
PRINT_EVERY    = 100

MAX_OHE_CARD   = 50


csv = pd.read_csv("../data/players_22.csv", dtype={25: str, 108: str})
csv = csv.drop(columns=[
    "player_url", "player_face_url", "club_logo_url",
    "club_flag_url", "nation_logo_url", "nation_flag_url"
])

csv = csv[csv[TARGET_COL].notna()]
X_full = csv.drop(columns=[TARGET_COL])
y_full = csv[TARGET_COL].values.reshape(-1, 1)


num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()

cat_candidates = X_full.select_dtypes(exclude=[np.number]).columns
cat_cols = [
    c for c in cat_candidates
    if X_full[c].nunique(dropna=True) <= MAX_OHE_CARD
]



X_train_2, X_test_2, y_train, y_test = train_test_split(
    X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])

def closed_form_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_b = add_bias(X)
    return np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y



def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return add_bias(X) @ w

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
            grad = 2 / len(yb) * Xb_b.T @ (Xb_b @ w - yb) #pochodna mse wzgledem wag
            w   -= lr * grad

        if  epoch % PRINT_EVERY == 0:
            mse = mean_squared_error(y, predict(X, w))
            print(f"epoch {epoch:4d}/{epochs}  MSE={mse:.4f}")

    return w



kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
mse_scores, r2_scores = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_full), start=1):

    X_tr_raw, X_val_raw = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_tr,     y_val     = y_full[train_idx],      y_full[val_idx]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        [("num", numeric_pipe, num_cols),
         ("cat", categorical_pipe, cat_cols)]
    )

    X_tr  = pre.fit_transform(X_tr_raw)
    X_val = pre.transform(X_val_raw)


    w = gradient_descent_fit(X_tr, y_tr)
    y_pred = predict(X_val, w)

    mse = mean_squared_error(y_val, y_pred)
    r2  = r2_score(y_val, y_pred)

    mse_scores.append(mse)
    r2_scores.append(r2)

    print(f"Fold {fold}:  MSE={mse:.4f}  |  R²={r2:.4f}")
print(f"Średnia      MSE={np.mean(mse_scores):.4f}  |  R²={np.mean(r2_scores):.4f}")
print(f"Odch. std    MSE={np.std(mse_scores):.4f}  |  R²={np.std(r2_scores):.4f}")
