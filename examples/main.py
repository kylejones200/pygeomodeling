# Geomodeling with GPR and Kriging in Python

"""
This script performs:
1. Gaussian Process Regression (GPR) on log-transformed PERMX from the real SPE9 dataset
2. K-Fold cross-validation of GPR
3. Ordinary Kriging (benchmark)
4. Export of predicted property and uncertainty to Eclipse-compatible GRDECL format
5. Visualization of GPR prediction, uncertainty, and training coverage
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from pykrige.ok import OrdinaryKriging
from grdecl_parser import load_spe9_data
import warnings
warnings.filterwarnings('ignore')

# Load real SPE9 data
print("Loading SPE9 dataset...")
data, _ = load_spe9_data()
nx, ny, nz = data['dimensions']
n_cells = nx * ny * nz
permx_3d = data['properties']['PERMX']

x_coords = np.linspace(0, 1, nx)
y_coords = np.linspace(0, 1, ny)
z_coords = np.linspace(0, 1, nz)
X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
permx = permx_3d.ravel()

valid_mask = permx > 1.0
X_valid = coords[valid_mask]
y_valid = np.log1p(permx[valid_mask])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_valid, y_valid, test_size=0.2, random_state=42)

# Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

# GPR Model
kernel = RBF([0.5, 0.5, 0.5], (1e-2, 1e1)) + Matern(length_scale=0.5, nu=1.5)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1.0, n_restarts_optimizer=5)
gpr.fit(X_train_scaled, y_train_scaled)

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores, cv_rmse = [], []
for train_idx, val_idx in kf.split(X_train_scaled):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train_scaled[train_idx], y_train_scaled[val_idx]
    g = GaussianProcessRegressor(kernel=kernel, alpha=1.0)
    g.fit(X_tr, y_tr)
    y_pred = g.predict(X_val)
    y_val_orig = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_orig = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    cv_scores.append(r2_score(y_val_orig, y_pred_orig))
    cv_rmse.append(np.sqrt(mean_squared_error(y_val_orig, y_pred_orig)))

# Predict
X_grid_scaled = x_scaler.transform(X_valid)
pred_log, sigma = gpr.predict(X_grid_scaled, return_std=True)
pred_perm = np.expm1(y_scaler.inverse_transform(pred_log.reshape(-1, 1)).flatten())

# Kriging (2D benchmark)
kx, ky, v = X_train[:, 0], X_train[:, 1], y_train
OK = OrdinaryKriging(kx, ky, v, variogram_model='spherical')
gx = np.linspace(0, 1, 50)
gy = np.linspace(0, 1, 50)
zk, ss = OK.execute('grid', gx, gy)

# Fill full grid with predictions
pred_3d = np.zeros(n_cells)
sigma_3d = np.zeros(n_cells)
pred_3d[valid_mask] = pred_perm
sigma_3d[valid_mask] = sigma
pred_3d = pred_3d.reshape((nx, ny, nz), order='F')
sigma_3d = sigma_3d.reshape((nx, ny, nz), order='F')

# Visualization
z_mid = nz // 2
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(permx_3d[:, :, z_mid].T, origin='lower', cmap='viridis')
ax[0].set_title("Original PERMX")
ax[1].imshow(pred_3d[:, :, z_mid].T, origin='lower', cmap='viridis')
ax[1].set_title("GPR Predicted PERMX")
ax[2].imshow(sigma_3d[:, :, z_mid].T, origin='lower', cmap='magma')
ax[2].set_title("Prediction Uncertainty σ")
plt.tight_layout()
plt.savefig("gpr_prediction_slices.png", dpi=150)
plt.show()

def write_grdecl_property(filename, values, keyword, nx, ny, nz):
    with open(filename, 'w') as f:
        f.write(f"{keyword}\n")
        for i, val in enumerate(values):
            f.write(f"{val:.6E} ")
            if (i + 1) % 5 == 0:
                f.write("\n")
        if (i + 1) % 5 != 0:
            f.write("\n")
        f.write("/\n")

write_grdecl_property("PERMX_GPR.GRDECL", pred_3d.ravel(order='F'), "PERMX", nx, ny, nz)
write_grdecl_property("SIGMA_GPR.GRDECL", sigma_3d.ravel(order='F'), "SIGMA", nx, ny, nz)

# Summary
print("\n=== SUMMARY ===")
print(f"Grid dimensions: {nx} x {ny} x {nz}")
print(f"Valid cells: {X_valid.shape[0]} / {n_cells}")
print(f"Predicted PERMX: {np.min(pred_perm):.2f} – {np.max(pred_perm):.2f} mD")
print(f"Mean σ: {np.mean(sigma):.2f}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Cross-validation R²: {np.mean(cv_scores):.3f}, RMSE: {np.mean(cv_rmse):.2f} mD")
