"""This code tests the Ensemble model combinations (Meta MLR, Meta RF, Meta SVR RBF, Meta SVR Linear, Meta SVR Polynomial)."""

import math
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# --- Level-0 Base Learners (Inputs stacked into the Meta-Learners) ---
ENABLE_BASE_RF         = True
ENABLE_BASE_MLR        = True
ENABLE_BASE_SVR_RBF    = True
ENABLE_BASE_SVR_LINEAR = True
ENABLE_BASE_SVR_POLY   = True

# --- Level-1 Meta-Learners (Ensemble models to train, evaluate, and plot) ---
ENABLE_META_RF         = True
ENABLE_META_MLR        = True
ENABLE_META_SVR_RBF    = True
ENABLE_META_SVR_LINEAR = True
ENABLE_META_SVR_POLY   = True

# 0. Directory Setup
saveModel_directory = "Output Models"
saveGraph_directory = "Graphs"
saveCSV_directory = "CSV_Results"

for folder in [saveModel_directory, saveGraph_directory, saveCSV_directory]:
    os.makedirs(folder, exist_ok=True)

# 1. Load Training and Testing Datasets
df_meta = pd.read_csv('Dataset/Meta_dataset.csv').dropna()
df_val = pd.read_csv('Dataset/Validation_dataset.csv').dropna()

# 2. Define Feature and Target Columns
feature_cols = [
    'temp',
    'RH',
    'PM10(Smoke)',
    'PM2.5(KCl)',
    'PM2.5(Smoke)',
    'PM1(KCl)',
    'PM1(Smoke)',
]
target_col = 'PM10(KCl)'

X_meta_train_raw = df_meta[feature_cols]
y_meta_train = df_meta[target_col]

X_val_raw = df_val[feature_cols]
y_val_test = df_val[target_col]

# 3. Scale Base Features (Only loaded/computed if at least one Base SVR is enabled)
if any([ENABLE_BASE_SVR_RBF, ENABLE_BASE_SVR_LINEAR, ENABLE_BASE_SVR_POLY]):
    scaler_X = joblib.load('Output Models/PM10_Base_svr_rbf_scaler_model.joblib')
    X_meta_train_scaled = scaler_X.transform(X_meta_train_raw)
    X_val_scaled = scaler_X.transform(X_val_raw)

# 4. Generate Level-0 Base Model Predictions (Only for enabled Base Learners)
meta_train_preds = []
val_preds = []
active_base_names = []

if ENABLE_BASE_MLR:
    mlr_model = joblib.load('Output Models/PM10_Base_mlr_model.joblib')
    meta_train_preds.append(mlr_model.predict(X_meta_train_raw))
    val_preds.append(mlr_model.predict(X_val_raw))
    active_base_names.append("Base MLR")

if ENABLE_BASE_RF:
    rf_model = joblib.load('Output Models/PM10_Base_rf_model.joblib')
    meta_train_preds.append(rf_model.predict(X_meta_train_raw))
    val_preds.append(rf_model.predict(X_val_raw))
    active_base_names.append("Base RF")

if ENABLE_BASE_SVR_RBF:
    svr_model_RBF = joblib.load('Output Models/PM10_Base_svr_rbf_model.joblib')
    meta_train_preds.append(svr_model_RBF.predict(X_meta_train_scaled))
    val_preds.append(svr_model_RBF.predict(X_val_scaled))
    active_base_names.append("Base SVR RBF")

if ENABLE_BASE_SVR_LINEAR:
    svr_model_LINEAR = joblib.load('Output Models/PM10_Base_svr_linear_model.joblib')
    meta_train_preds.append(svr_model_LINEAR.predict(X_meta_train_scaled))
    val_preds.append(svr_model_LINEAR.predict(X_val_scaled))
    active_base_names.append("Base SVR Linear")

if ENABLE_BASE_SVR_POLY:
    svr_model_POLY = joblib.load('Output Models/PM10_Base_svr_poly_model.joblib')
    meta_train_preds.append(svr_model_POLY.predict(X_meta_train_scaled))
    val_preds.append(svr_model_POLY.predict(X_val_scaled))
    active_base_names.append("Base SVR Poly")

if not meta_train_preds:
    raise ValueError("Error: At least one ENABLE_BASE_* switch must be set to True.")

print(f"Active Level-0 Base Learners ({len(active_base_names)}): {', '.join(active_base_names)}")

# Stack Level-1 matrices dynamically based on active base learners
X_meta_train_stacked = np.column_stack(meta_train_preds)
X_val_stacked = np.column_stack(val_preds)

# Scale Level-1 Stacked Features (Only computed if at least one Meta SVR is enabled)
if any([ENABLE_META_SVR_RBF, ENABLE_META_SVR_LINEAR, ENABLE_META_SVR_POLY]):
    meta_scaler_X = StandardScaler()
    X_meta_train_stacked_scaled = meta_scaler_X.fit_transform(X_meta_train_stacked)
    X_val_stacked_scaled = meta_scaler_X.transform(X_val_stacked)
    joblib.dump(
        meta_scaler_X,
        os.path.join(saveModel_directory, 'PM10_Combination_Meta_SVR_scaler.joblib'),
    )

# 5. Train & Evaluate Enabled Level-1 Meta-Learners
df_test_results = df_val.copy()
metrics = {}
plot_configs = []

# --- Meta RF ---
if ENABLE_META_RF:
    ensemble_RF_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=13,
        max_features=min(3, X_meta_train_stacked.shape[1]),  # Prevents crash if <3 base models active
        random_state=61,
    )
    ensemble_RF_model.fit(X_meta_train_stacked, y_meta_train)
    joblib.dump(
        ensemble_RF_model,
        os.path.join(saveModel_directory, 'PM10_Combination_Ensemble_RF_model.joblib'),
    )
    y_pred_rf = ensemble_RF_model.predict(X_val_stacked)
    df_test_results['RF Model'] = y_pred_rf
    metrics['Meta Random Forest'] = {
        'mae': mean_absolute_error(y_val_test, y_pred_rf),
        'rmse': root_mean_squared_error(y_val_test, y_pred_rf),
        'r2': r2_score(y_val_test, y_pred_rf),
    }
    plot_configs.append({
        'y_pred': y_pred_rf,
        'title': 'Meta Random Forest',
        'color': '#059669',
        'edge': '#047857',
        'm': metrics['Meta Random Forest'],
    })

# --- Meta MLR ---
if ENABLE_META_MLR:
    ensemble_MLR_model = LinearRegression()
    ensemble_MLR_model.fit(X_meta_train_stacked, y_meta_train)
    joblib.dump(
        ensemble_MLR_model,
        os.path.join(saveModel_directory, 'PM10_Combination_Ensemble_MLR_model.joblib'),
    )
    y_pred_mlr = ensemble_MLR_model.predict(X_val_stacked)
    df_test_results['MLR Model'] = y_pred_mlr
    metrics['Meta MLR'] = {
        'mae': mean_absolute_error(y_val_test, y_pred_mlr),
        'rmse': root_mean_squared_error(y_val_test, y_pred_mlr),
        'r2': r2_score(y_val_test, y_pred_mlr),
    }
    plot_configs.append({
        'y_pred': y_pred_mlr,
        'title': 'Meta MLR',
        'color': '#8B5CF6',
        'edge': '#6D28D9',
        'm': metrics['Meta MLR'],
    })

# --- Meta SVR RBF ---
if ENABLE_META_SVR_RBF:
    ensemble_SVR_RBF_model = SVR(kernel='rbf', C=20.0, epsilon=0.01, gamma=0.07)
    ensemble_SVR_RBF_model.fit(X_meta_train_stacked_scaled, y_meta_train)
    joblib.dump(
        ensemble_SVR_RBF_model,
        os.path.join(saveModel_directory, 'PM10_Combination_Ensemble_SVR_RBF_model.joblib'),
    )
    y_pred_svr_rbf = ensemble_SVR_RBF_model.predict(X_val_stacked_scaled)
    df_test_results['SVR RBF Model'] = y_pred_svr_rbf
    metrics['Meta SVR RBF'] = {
        'mae': mean_absolute_error(y_val_test, y_pred_svr_rbf),
        'rmse': root_mean_squared_error(y_val_test, y_pred_svr_rbf),
        'r2': r2_score(y_val_test, y_pred_svr_rbf),
    }
    plot_configs.append({
        'y_pred': y_pred_svr_rbf,
        'title': 'Meta SVR RBF Kernel',
        'color': '#0D9488',
        'edge': '#0F766E',
        'm': metrics['Meta SVR RBF'],
    })

# --- Meta SVR Linear ---
if ENABLE_META_SVR_LINEAR:
    ensemble_SVR_Linear_model = SVR(kernel='linear', C=10.0, epsilon=0.01)
    ensemble_SVR_Linear_model.fit(X_meta_train_stacked_scaled, y_meta_train)
    joblib.dump(
        ensemble_SVR_Linear_model,
        os.path.join(saveModel_directory, 'PM10_Combination_Ensemble_SVR_Linear_model.joblib'),
    )
    y_pred_svr_lin = ensemble_SVR_Linear_model.predict(X_val_stacked_scaled)
    df_test_results['SVR Linear Model'] = y_pred_svr_lin
    metrics['Meta SVR Linear'] = {
        'mae': mean_absolute_error(y_val_test, y_pred_svr_lin),
        'rmse': root_mean_squared_error(y_val_test, y_pred_svr_lin),
        'r2': r2_score(y_val_test, y_pred_svr_lin),
    }
    plot_configs.append({
        'y_pred': y_pred_svr_lin,
        'title': 'Meta SVR Linear Kernel',
        'color': '#0EA5E9',
        'edge': '#0284C7',
        'm': metrics['Meta SVR Linear'],
    })

# --- Meta SVR Polynomial ---
if ENABLE_META_SVR_POLY:
    ensemble_SVR_Poly_model = SVR(
        kernel='poly', C=10.0, epsilon=0.01, degree=3, coef0=1.0, gamma='scale'
    )
    ensemble_SVR_Poly_model.fit(X_meta_train_stacked_scaled, y_meta_train)
    joblib.dump(
        ensemble_SVR_Poly_model,
        os.path.join(saveModel_directory, 'PM10_Combination_Ensemble_SVR_Poly_model.joblib'),
    )
    y_pred_svr_poly = ensemble_SVR_Poly_model.predict(X_val_stacked_scaled)
    df_test_results['SVR Poly Model'] = y_pred_svr_poly
    metrics['Meta SVR Polynomial'] = {
        'mae': mean_absolute_error(y_val_test, y_pred_svr_poly),
        'rmse': root_mean_squared_error(y_val_test, y_pred_svr_poly),
        'r2': r2_score(y_val_test, y_pred_svr_poly),
    }
    plot_configs.append({
        'y_pred': y_pred_svr_poly,
        'title': 'Meta SVR Polynomial Kernel',
        'color': '#F59E0B',
        'edge': '#D97706',
        'm': metrics['Meta SVR Polynomial'],
    })

# 6. Export CSV & Print Metrics for Active Meta-Learners
df_test_results.to_csv(
    os.path.join(
        saveCSV_directory,
        'PM10_Combination_Ensemble_dataset_with_predictions.csv',
    ),
    index=False,
)

print('\n=== MODEL EVALUATION METRICS ===')
for name, m in metrics.items():
    print(f"{name:<22}: MAE={m['mae']:.4f} | RMSE={m['rmse']:.4f} | R²={m['r2']:.4f}")

# 7. Dynamically Plot Scatterplots for Enabled Meta-Learners
n_plots = len(plot_configs)
if n_plots > 0:
    n_cols = 1 if n_plots == 1 else 2
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()

    for idx, cfg in enumerate(plot_configs):
        ax = axes_flat[idx]
        y_t = y_val_test
        y_p = cfg['y_pred']

        ax.scatter(
            y_t,
            y_p,
            alpha=0.7,
            color=cfg['color'],
            edgecolors=cfg['edge'],
            label='Predictions',
        )

        # 1:1 Reference Line
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            linewidth=2,
            label='1:1 Ideal Line',
        )

        ax.set_title(cfg['title'], fontsize=12, fontweight='bold')
        ax.set_xlabel('PM10 Reference Concentration (µg/m³)', fontsize=10)
        ax.set_ylabel('PM10 Predicted Concentration (µg/m³)', fontsize=10)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Overlay metrics text box
        txt = (
            f"$R^2$ = {cfg['m']['r2']:.3f}\nRMSE = {cfg['m']['rmse']:.2f}\nMAE ="
            f" {cfg['m']['mae']:.2f}"
        )
        ax.text(
            0.60,
            0.08,
            txt,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                alpha=0.9,
                edgecolor='#CBD5E1',
            ),
        )

    # Remove any unused trailing subplots
    for unused_idx in range(n_plots, len(axes_flat)):
        fig.delaxes(axes_flat[unused_idx])

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            saveGraph_directory,
            'PM10_Combination_Ensemble_Models_Scatterplots.png',
        ),
        dpi=400,
    )
    plt.show()