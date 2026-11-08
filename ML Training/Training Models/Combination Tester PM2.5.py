"""This code tests the Ensemble model combinations (Meta MLR, Meta RF, Meta SVR RBF, Meta SVR Linear)."""

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

# 0. Directory Setup
saveModel_directory = "Output Models"
saveGraph_directory = "Graphs"
saveCSV_directory = "CSV_Results"

for folder in [saveModel_directory, saveGraph_directory, saveCSV_directory]:
  os.makedirs(folder, exist_ok=True)

# 1. Import / Load pre-trained base models from disk
rf_model            = joblib.load('Output Models/PM2.5_Base_rf_model.joblib')
mlr_model           = joblib.load('Output Models/PM2.5_Base_mlr_model.joblib')
svr_model_RBF       = joblib.load('Output Models/PM2.5_Base_svr_rbf_model.joblib')
svr_model_LINEAR    = joblib.load('Output Models/PM2.5_Base_svr_linear_model.joblib')
scaler_X            = joblib.load('Output Models/PM2.5_Base_svr_rbf_scaler_model.joblib')

# 2. Load Training and Testing Datasets
df_meta = pd.read_csv('Dataset/Meta_dataset.csv').dropna()
df_val = pd.read_csv('Dataset/Validation_dataset.csv').dropna()

# 3. Define Feature and Target Columns
feature_cols = [
    'temp',
    'RH',
    'PM10(Smoke)',
    'PM10(KCl)',
    'PM2.5(Smoke)',
    'PM1(KCl)',
    'PM1(Smoke)',
]
target_col = 'PM2.5(KCl)'

# Feature matrices for ensemble training (Meta_dataset)
X_meta_train_raw = df_meta[feature_cols]
y_meta_train = df_meta[target_col]

# Feature matrices for ensemble testing (Validation_dataset alone)
X_val_raw = df_val[feature_cols]
y_val_test = df_val[target_col]

# 4. Scale Base Features using the Pre-Fitted Scaler
X_meta_train_scaled = scaler_X.transform(X_meta_train_raw)
X_val_scaled = scaler_X.transform(X_val_raw)

# Generate base model predictions on Meta_dataset
pred_mlr_meta_train = mlr_model.predict(X_meta_train_raw)
pred_rf_meta_train = rf_model.predict(X_meta_train_raw)
pred_svr_RBF_meta_train = svr_model_RBF.predict(X_meta_train_scaled)
pred_svr_LINEAR_meta_train = svr_model_LINEAR.predict(X_meta_train_scaled)

# Stack Level-1 predictions for Meta-Learner training
X_meta_train_stacked = np.column_stack((
    pred_mlr_meta_train,
    pred_rf_meta_train,
    pred_svr_RBF_meta_train,
    pred_svr_LINEAR_meta_train,
))

# 5. Train Meta-Learners
# --- Meta MLR ---
ensemble_MLR_model = LinearRegression()
ensemble_MLR_model.fit(X_meta_train_stacked, y_meta_train)
joblib.dump(
    ensemble_MLR_model,
    os.path.join(saveModel_directory, 'PM2.5_Ensemble_MLR_model.joblib'),
)

# --- Meta RF ---
ensemble_RF_model = RandomForestRegressor(
    n_estimators=300,  # Adjusted to reasonable ensemble count
    max_depth=13,
    max_features=3,  
    random_state=61,
)
ensemble_RF_model.fit(X_meta_train_stacked, y_meta_train)
joblib.dump(
    ensemble_RF_model,
    os.path.join(saveModel_directory, 'PM2.5_Ensemble_RF_model.joblib'),
)

# --- Fit and Save Meta-Scaler for Level-1 Stacked Features ---
meta_scaler_X = StandardScaler()
X_meta_train_stacked_scaled = meta_scaler_X.fit_transform(X_meta_train_stacked)
joblib.dump(
    meta_scaler_X,
    os.path.join(saveModel_directory, 'PM2.5_Meta_SVR_scaler.joblib'),
)

# --- Meta SVR RBF ---
ensemble_SVR_RBF_model = SVR(kernel='rbf', C=20.0, epsilon=0.01, gamma=0.07)
ensemble_SVR_RBF_model.fit(X_meta_train_stacked_scaled, y_meta_train)
joblib.dump(
    ensemble_SVR_RBF_model,
    os.path.join(saveModel_directory, 'PM2.5_Ensemble_SVR_RBF_model.joblib'),
)

# --- Meta SVR Linear ---
ensemble_SVR_Linear_model = SVR(kernel='linear', C=10.0, epsilon=0.01)
ensemble_SVR_Linear_model.fit(X_meta_train_stacked_scaled, y_meta_train)
joblib.dump(
    ensemble_SVR_Linear_model,
    os.path.join(saveModel_directory, 'PM2.5_Ensemble_SVR_Linear_model.joblib'),
)

# 6. Generate Base Predictions for Validation_dataset
pred_mlr_val = mlr_model.predict(X_val_raw)
pred_rf_val = rf_model.predict(X_val_raw)
pred_svr_RBF_val = svr_model_RBF.predict(X_val_scaled)
pred_svr_LINEAR_val = svr_model_LINEAR.predict(X_val_scaled)

X_val_stacked = np.column_stack(
    (pred_mlr_val, pred_rf_val, pred_svr_RBF_val, pred_svr_LINEAR_val)
)

# FIX: Transform Level-1 Validation Features using Meta-Scaler
X_val_stacked_scaled = meta_scaler_X.transform(X_val_stacked)

# 7. Generate Ensemble Predictions on Validation_dataset
y_pred_ensemble_MLR_model = ensemble_MLR_model.predict(X_val_stacked)
y_pred_ensemble_RF_model = ensemble_RF_model.predict(X_val_stacked)

# FIX: Pass scaled Level-1 validation matrix into Meta SVR models
y_pred_ensemble_SVR_RBF_model = ensemble_SVR_RBF_model.predict(
    X_val_stacked_scaled
)
y_pred_ensemble_SVR_Linear_model = ensemble_SVR_Linear_model.predict(
    X_val_stacked_scaled
)

# Export Ensemble Validation CSV
df_test_results = df_val.copy()
df_test_results['MLR Model'] = y_pred_ensemble_MLR_model
df_test_results['RF Model'] = y_pred_ensemble_RF_model
df_test_results['SVR RBF Model'] = y_pred_ensemble_SVR_RBF_model
df_test_results['SVR Linear Model'] = y_pred_ensemble_SVR_Linear_model
df_test_results.to_csv(
    os.path.join(
        saveCSV_directory,
        'PM2.5_Ensemble_Combination_dataset_with_predictions.csv',
    ),
    index=False,
)

# 8. Compute Performance Metrics
metrics = {
    'RF': {
        'mae': mean_absolute_error(y_val_test, y_pred_ensemble_RF_model),
        'rmse': root_mean_squared_error(y_val_test, y_pred_ensemble_RF_model),
        'r2': r2_score(y_val_test, y_pred_ensemble_RF_model),
    },
    'SVR_RBF': {
        'mae': mean_absolute_error(y_val_test, y_pred_ensemble_SVR_RBF_model),
        'rmse': root_mean_squared_error(
            y_val_test, y_pred_ensemble_SVR_RBF_model
        ),
        'r2': r2_score(y_val_test, y_pred_ensemble_SVR_RBF_model),
    },
    'SVR_Linear': {
        'mae': mean_absolute_error(
            y_val_test, y_pred_ensemble_SVR_Linear_model
        ),
        'rmse': root_mean_squared_error(
            y_val_test, y_pred_ensemble_SVR_Linear_model
        ),
        'r2': r2_score(y_val_test, y_pred_ensemble_SVR_Linear_model),
    },
    'MLR': {
        'mae': mean_absolute_error(y_val_test, y_pred_ensemble_MLR_model),
        'rmse': root_mean_squared_error(y_val_test, y_pred_ensemble_MLR_model),
        'r2': r2_score(y_val_test, y_pred_ensemble_MLR_model),
    },
}

print('=== MODEL EVALUATION METRICS ===')
print(
    f"Meta Random Forest   : MAE={metrics['RF']['mae']:.4f} |"
    f" RMSE={metrics['RF']['rmse']:.4f} | R²={metrics['RF']['r2']:.4f}"
)
print(
    f"Meta SVR RBF        : MAE={metrics['SVR_RBF']['mae']:.4f} |"
    f" RMSE={metrics['SVR_RBF']['rmse']:.4f} |"
    f" R²={metrics['SVR_RBF']['r2']:.4f}"
)
print(
    f"Meta SVR Linear     : MAE={metrics['SVR_Linear']['mae']:.4f} |"
    f" RMSE={metrics['SVR_Linear']['rmse']:.4f} |"
    f" R²={metrics['SVR_Linear']['r2']:.4f}"
)
print(
    f"Meta MLR            : MAE={metrics['MLR']['mae']:.4f} |"
    f" RMSE={metrics['MLR']['rmse']:.4f} | R²={metrics['MLR']['r2']:.4f}"
)

# 9. Plot 4 Scatterplots in a Single 2x2 Image
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
((ax1, ax2), (ax3, ax4)) = axes

plot_configs = [
    {
        'ax': ax1,
        'y_true': y_val_test,
        'y_pred': y_pred_ensemble_RF_model,
        'title': 'Meta Random Forest',
        'color': '#059669',
        'edge': '#047857',
        'm': metrics['RF'],
    },
    {
        'ax': ax2,
        'y_true': y_val_test,
        'y_pred': y_pred_ensemble_SVR_RBF_model,
        'title': 'Meta SVR RBF Kernel',
        'color': '#0D9488',
        'edge': '#0F766E',
        'm': metrics['SVR_RBF'],
    },
    {
        'ax': ax3,
        'y_true': y_val_test,
        'y_pred': y_pred_ensemble_SVR_Linear_model,
        'title': 'Meta SVR Linear Kernel',
        'color': '#0EA5E9',
        'edge': '#0284C7',
        'm': metrics['SVR_Linear'],
    },
    {
        'ax': ax4,
        'y_true': y_val_test,
        'y_pred': y_pred_ensemble_MLR_model,
        'title': 'Meta MLR Base Learner',
        'color': '#8B5CF6',
        'edge': '#6D28D9',
        'm': metrics['MLR'],
    },
]

for cfg in plot_configs:
  ax = cfg['ax']
  y_t = cfg['y_true']
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
  ax.set_xlabel('PM2.5 Reference Concentration (µg/m³)', fontsize=10)
  ax.set_ylabel('PM2.5 Predicted Concentration (µg/m³)', fontsize=10)
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

plt.tight_layout()
plt.savefig(os.path.join(saveGraph_directory,'PM2.5_Ensemble_Combination_Models_Scatterplots.png',),dpi=400,)
plt.show()