"""
This code is for testing the basic learner models by fusing the meta dataset and meta validation dataset for the meta learner
"""
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# 0. Directory Setup
saveModel_directory = "Output Models"
saveGraph_directory = "Graphs"
saveCSV_directory = "CSV_Results"

for folder in [saveModel_directory, saveGraph_directory, saveCSV_directory]:
  os.makedirs(folder, exist_ok=True)

# 1. Import / Load pre-trained models from disk
rf_model = joblib.load('Output Models/Base_rf_model.joblib')
svr_model_RBF = joblib.load('Output Models/Base_svr_rbf_model.joblib')
svr_model_LINEAR = joblib.load('Output Models/Base_svr_linear_model.joblib')
scaler_X = joblib.load('Output Models/Base_svr_scaler_model.joblib')

# 2. Load Training and Testing Datasets
df_meta = pd.read_csv('Dataset/Meta_dataset.csv').dropna()
df_val = pd.read_csv('Dataset/Validation_dataset.csv').dropna()

# Combine Meta and Validation datasets ONLY for testing base learners
df_merged_base_test = pd.concat([df_meta, df_val], ignore_index=True)

# 3. Define Feature and Target Columns
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

# Feature matrices for MLR training (Meta_dataset)
X_meta_train_raw = df_meta[feature_cols]
y_meta_train = df_meta[target_col]

# Feature matrices for MLR testing (Validation_dataset alone)
X_val_raw = df_val[feature_cols]
y_val_test = df_val[target_col]

# Feature matrices for Base Learners testing (Merged dataset)
X_merged_raw = df_merged_base_test[feature_cols]
y_merged_test = df_merged_base_test[target_col]

# 4. Scale Features using the Pre-Fitted Scaler
X_meta_train_scaled = scaler_X.transform(X_meta_train_raw)
X_val_scaled = scaler_X.transform(X_val_raw)
X_merged_scaled = scaler_X.transform(X_merged_raw)

# 5. Train the Stacked MLR Meta-Learner
# Generate base model predictions on Meta_dataset
pred_rf_meta_train = rf_model.predict(X_meta_train_raw)
pred_svr_RBF_meta_train = svr_model_RBF.predict(X_meta_train_scaled)
pred_svr_LINEAR_meta_train = svr_model_LINEAR.predict(X_meta_train_scaled)

X_meta_train_stacked = np.column_stack((
    pred_rf_meta_train,
    pred_svr_RBF_meta_train,
    pred_svr_LINEAR_meta_train,
))

mlr_model = LinearRegression()
mlr_model.fit(X_meta_train_stacked, y_meta_train)
joblib.dump(
    mlr_model, os.path.join(saveModel_directory, 'Meta_MLR_model.joblib')
)

# 6. Generate MLR Predictions on Validation_dataset ALONE
pred_rf_val = rf_model.predict(X_val_raw)
pred_svr_RBF_val = svr_model_RBF.predict(X_val_scaled)
pred_svr_LINEAR_val = svr_model_LINEAR.predict(X_val_scaled)

X_val_stacked = np.column_stack(
    (pred_rf_val, pred_svr_RBF_val, pred_svr_LINEAR_val)
)
y_pred_mlr = mlr_model.predict(X_val_stacked)

# Export MLR Validation CSV
df_test_results = df_val.copy()
df_test_results['prediction'] = y_pred_mlr
df_test_results.to_csv(
    os.path.join(saveCSV_directory, 'MLR_dataset_with_predictions.csv'),
    index=False,
)

# 7. Generate Base Learner Predictions on MERGED Dataset
y_pred_rf = rf_model.predict(X_merged_raw)
y_pred_svr_rbf = svr_model_RBF.predict(X_merged_scaled)
y_pred_svr_linear = svr_model_LINEAR.predict(X_merged_scaled)

# 8. Compute Performance Metrics
# Base Learners (Tested on Merged Dataset)
metrics = {
    'RF': {
        'mae': mean_absolute_error(y_merged_test, y_pred_rf),
        'rmse': root_mean_squared_error(y_merged_test, y_pred_rf),
        'r2': r2_score(y_merged_test, y_pred_rf),
    },
    'SVR_RBF': {
        'mae': mean_absolute_error(y_merged_test, y_pred_svr_rbf),
        'rmse': root_mean_squared_error(y_merged_test, y_pred_svr_rbf),
        'r2': r2_score(y_merged_test, y_pred_svr_rbf),
    },
    'SVR_Linear': {
        'mae': mean_absolute_error(y_merged_test, y_pred_svr_linear),
        'rmse': root_mean_squared_error(y_merged_test, y_pred_svr_linear),
        'r2': r2_score(y_merged_test, y_pred_svr_linear),
    },
    # Meta Learner (Tested on Validation Dataset Alone)
    'MLR': {
        'mae': mean_absolute_error(y_val_test, y_pred_mlr),
        'rmse': root_mean_squared_error(y_val_test, y_pred_mlr),
        'r2': r2_score(y_val_test, y_pred_mlr),
    },
}

print('=== MODEL EVALUATION METRICS ===')
print(
    f"Random Forest (Merged Set)   : MAE={metrics['RF']['mae']:.4f} |"
    f" RMSE={metrics['RF']['rmse']:.4f} | R²={metrics['RF']['r2']:.4f}"
)
print(
    f"SVR RBF (Merged Set)         : MAE={metrics['SVR_RBF']['mae']:.4f} |"
    f" RMSE={metrics['SVR_RBF']['rmse']:.4f} |"
    f" R²={metrics['SVR_RBF']['r2']:.4f}"
)
print(
    f"SVR Linear (Merged Set)      :"
    f" MAE={metrics['SVR_Linear']['mae']:.4f} |"
    f" RMSE={metrics['SVR_Linear']['rmse']:.4f} |"
    f" R²={metrics['SVR_Linear']['r2']:.4f}"
)
print(
    f"Stacked MLR (Validation Set) : MAE={metrics['MLR']['mae']:.4f} |"
    f" RMSE={metrics['MLR']['rmse']:.4f} | R²={metrics['MLR']['r2']:.4f}"
)

# 9. Plot 4 Scatterplots in a Single 2x2 Image
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
((ax1, ax2), (ax3, ax4)) = axes

plot_configs = [
    {
        'ax': ax1,
        'y_true': y_merged_test,
        'y_pred': y_pred_rf,
        'title': 'Random Forest (Merged Test Set)',
        'color': '#059669',
        'edge': '#047857',
        'm': metrics['RF'],
    },
    {
        'ax': ax2,
        'y_true': y_merged_test,
        'y_pred': y_pred_svr_rbf,
        'title': 'SVR RBF Kernel (Merged Test Set)',
        'color': '#0D9488',
        'edge': '#0F766E',
        'm': metrics['SVR_RBF'],
    },
    {
        'ax': ax3,
        'y_true': y_merged_test,
        'y_pred': y_pred_svr_linear,
        'title': 'SVR Linear Kernel (Merged Test Set)',
        'color': '#0EA5E9',
        'edge': '#0284C7',
        'm': metrics['SVR_Linear'],
    },
    {
        'ax': ax4,
        'y_true': y_val_test,
        'y_pred': y_pred_mlr,
        'title': 'Stacked MLR (Validation Set Alone)',
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
  ax.set_xlabel('Reference Concentration (µg/m³)', fontsize=10)
  ax.set_ylabel('Predicted Concentration (µg/m³)', fontsize=10)
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
plt.savefig(
    os.path.join(saveGraph_directory, 'All_Models_Scatterplots.png'), dpi=300
)
plt.show()
