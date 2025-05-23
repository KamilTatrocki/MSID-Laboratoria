import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

RANDOM_STATE   = 42
TARGET_COL     = "overall"
TEST_SIZE      = 0.2
LEARNING_RATE  = 0.01
EPOCHS         = 1000
BATCH_SIZE     = 512
PRINT_EVERY    = 1

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


numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])






categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    [("num", numeric_pipe, num_cols),
     ("cat", categorical_pipe, cat_cols)]
)

X_train = preprocessor.fit_transform(X_train_2)
X_test  = preprocessor.transform(X_test_2)


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


def gradient_descent_fit_l2(
        X_train, y_train, X_test, y_test, lr=0.001, epochs=1_000, batch_size=256,  l2=0.01):
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples, n_features = X_train.shape
    w = rng.normal(scale=0.01, size=(n_features + 1, 1)).astype(X_train.dtype)
    reg_mask = np.vstack([np.zeros((1, 1)), np.ones((n_features, 1))])

    train_mse_history = []
    test_mse_history = []
    epoch_history = []

    for epoch in range(epochs + 1):
        for Xb, yb in minibatches(X_train, y_train, batch_size, rng=rng):
            Xb_b = add_bias(Xb)
            grad = 2 / len(yb) * Xb_b.T @ (Xb_b @ w - yb)
            grad += 2 * l2 * w * reg_mask
            w -= lr * grad

        if epoch % PRINT_EVERY == 0:
            train_mse = mean_squared_error(y_train, predict(X_train, w))
            test_mse = mean_squared_error(y_test, predict(X_test, w))

            #print(f"epoch {epoch:4d}/{epochs}  Train MSE={train_mse:.4f}  Test MSE={test_mse:.4f}")

            train_mse_history.append(train_mse)
            test_mse_history.append(test_mse)
            epoch_history.append(epoch)

    return w, train_mse_history, test_mse_history, epoch_history

w_gd, train_mse_history, test_mse_history, epoch_history = gradient_descent_fit_l2(
    X_train, y_train, X_test, y_test, lr=LEARNING_RATE,
    epochs=EPOCHS, batch_size=BATCH_SIZE
)

y_pred_gd = predict(X_test, w_gd)



plt.figure(figsize=(10, 6))
plt.plot(epoch_history, train_mse_history, 'b-', label='Zbiór treningowy')
plt.plot(epoch_history, test_mse_history, 'r-', label='Zbiór testowy')
plt.xlabel('Epoka')
plt.ylabel('MSE')
plt.title('Zbieżność funkcji kosztu względem epoki')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mse_zbieznosc.png')
plt.show()