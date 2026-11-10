# -*- coding: utf-8 -*-
"""poker_analysis.ipynb - Linear regression via Normal Equations"""

import numpy as np
import jax.numpy as jnp
import jax.numpy.linalg as jla
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

print("--- 1. Loading Local Dataset ---")
X_raw = np.load('X_raw.npy')
y_raw = np.load('y_raw.npy')
print(f"Loaded X_raw: {X_raw.shape}")   # expect (169, 4)
print(f"Loaded y_raw: {y_raw.shape}")   # expect (169,)
# Columns: [high_card, low_card, is_suited, is_pair]

print("\n--- 2. Feature Expansion ---")
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_expanded = poly.fit_transform(X_raw)
print(f"Expanded feature count: {X_expanded.shape[1]}")

def normal_equations(X, y, lambda_reg=0.001):
    """
    Ridge regression via normal equations — exact closed-form solution.
    Bias term (index 0) excluded from regularization penalty.
    """
    n_features = X.shape[1]
    reg_matrix = lambda_reg * jnp.eye(n_features)
    reg_matrix = reg_matrix.at[0, 0].set(0.0)
    XtX   = jnp.dot(X.T, X) + reg_matrix
    Xty   = jnp.dot(X.T, y)
    return jla.solve(XtX, Xty)

print("\n--- 3. K-Fold Cross-Validation ---")
K  = 5
kf = KFold(n_splits=K, shuffle=True, random_state=42)
fold_train_mse, fold_test_mse = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_expanded)):
    X_train, X_test = X_expanded[train_idx], X_expanded[test_idx]
    y_train, y_test = y_raw[train_idx],      y_raw[test_idx]

    mu    = np.mean(X_train, axis=0)
    sigma = np.std(X_train,  axis=0) + 1e-8

    X_train_b = np.hstack([np.ones((len(X_train), 1)), (X_train - mu) / sigma])
    X_test_b  = np.hstack([np.ones((len(X_test),  1)), (X_test  - mu) / sigma])

    theta = normal_equations(
        jnp.array(X_train_b, dtype=jnp.float32),
        jnp.array(y_train,   dtype=jnp.float32)
    )

    train_preds = np.clip(np.dot(X_train_b, theta), 0.0, 1.0)
    test_preds  = np.clip(np.dot(X_test_b,  theta), 0.0, 1.0)

    t_mse = np.mean((y_train - train_preds) ** 2)
    v_mse = np.mean((y_test  - test_preds)  ** 2)
    fold_train_mse.append(t_mse)
    fold_test_mse.append(v_mse)
    print(f"  Fold {fold+1}/{K} — Train MSE: {t_mse:.5f} | Test MSE: {v_mse:.5f}")

print(f"\nMean Train MSE : {np.mean(fold_train_mse):.5f} +/- {np.std(fold_train_mse):.5f}")
print(f"Mean Test  MSE : {np.mean(fold_test_mse):.5f}  +/- {np.std(fold_test_mse):.5f}")

print("\n--- 4. Final Model (Full Dataset) ---")
mu_full    = np.mean(X_expanded, axis=0)
sigma_full = np.std(X_expanded,  axis=0) + 1e-8
X_full_b   = np.hstack([np.ones((len(X_expanded), 1)),
                         (X_expanded - mu_full) / sigma_full])

final_theta = normal_equations(
    jnp.array(X_full_b, dtype=jnp.float32),
    jnp.array(y_raw,    dtype=jnp.float32)
)

all_preds = np.clip(np.dot(X_full_b, final_theta), 0.0, 1.0)
print(f"Full Dataset MSE: {np.mean((y_raw - all_preds) ** 2):.5f}")

print("\nSample Predictions vs Ground Truth:")
for i in range(5):
    print(f"  True: {y_raw[i]*100:.1f}% | Predicted: {all_preds[i]*100:.1f}%")

# ── Export for VS Code UI ─────────────────────────────────────────────────────
np.save('model_theta.npy', np.array(final_theta))
np.save('model_mu.npy',    mu_full)
np.save('model_sigma.npy', sigma_full)
print("\nModel exported: model_theta.npy, model_mu.npy, model_sigma.npy")
print("Download these 3 files from the Colab sidebar.")