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

# 1. Import / Load pre-trained models from disk
rf_model            = joblib.load('Output Models/PM2.5_Base_rf_model.joblib')
mlr_model           = joblib.load('Output Models/PM2.5_Base_MLR_model.joblib')
svr_model_RBF       = joblib.load('Output Models/PM2.5_Base_svr_rbf_model.joblib')
svr_model_LINEAR    = joblib.load('Output Models/PM2.5_Base_svr_linear_model.joblib')
scaler_X            = joblib.load('Output Models/PM2.5_Base_svr_rbf_scaler_model.joblib') # linear and rbf models use the same scaler, so we can load either one
#ensemble_model      = joblib.load('Output Models/PM2.5_Ensemble_model.joblib')

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
    'PM10(KCl)',
    'PM2.5(Smoke)',
    'PM1(KCl)',
    'PM1(Smoke)',
]
target_col = 'PM2.5(KCl)'

# Feature matrices for Base Learners testing (Merged dataset)
X_merged_raw = df_merged_base_test[feature_cols]
y_merged_test = df_merged_base_test[target_col]

X_merged_scaled = scaler_X.transform(X_merged_raw)

#Generate Base Learner Predictions on MERGED Dataset
y_pred_mlr          = mlr_model.predict(X_merged_raw)
y_pred_rf           = rf_model.predict(X_merged_raw)
y_pred_svr_rbf      = svr_model_RBF.predict(X_merged_scaled)
y_pred_svr_linear   = svr_model_LINEAR.predict(X_merged_scaled)

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
    'MLR': {
        'mae': mean_absolute_error(y_merged_test, y_pred_mlr),
        'rmse': root_mean_squared_error(y_merged_test, y_pred_mlr),
        'r2': r2_score(y_merged_test, y_pred_mlr),
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
    f"MLR (Merged Set) : MAE={metrics['MLR']['mae']:.4f} |"
    f" RMSE={metrics['MLR']['rmse']:.4f} | R²={metrics['MLR']['r2']:.4f}"
)

# 9. Plot 4 Scatterplots in a Single 2x2 Image
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
((ax1, ax2), (ax3, ax4)) = axes

plot_configs = [
    {
        "ax": ax1,
        "y_true": y_merged_test,
        "y_pred": y_pred_rf,
        "title": "PM2.5 Random Forest (Merged Test Set)",
        "color": "#059669",
        "edge": "#047857",
        "m": metrics["RF"],
    },
    {
        "ax": ax2,
        "y_true": y_merged_test,
        "y_pred": y_pred_svr_rbf,
        "title": "PM2.5 SVR RBF Kernel (Merged Test Set)",
        "color": "#0D9488",
        "edge": "#0F766E",
        "m": metrics["SVR_RBF"],
    },
    {
        "ax": ax3,
        "y_true": y_merged_test,
        "y_pred": y_pred_svr_linear,
        "title": "PM2.5 SVR Linear Kernel (Merged Test Set)",
        "color": "#0EA5E9",
        "edge": "#0284C7",
        "m": metrics["SVR_Linear"],
    },
    {
        "ax": ax4,
        "y_true": y_merged_test,  # Fixed: Match y_pred_mlr dataset
        "y_pred": y_pred_mlr,
        "title": "PM2.5 MLR Base Learner (Merged Test Set)",
        "color": "#8B5CF6",
        "edge": "#6D28D9",
        "m": metrics["MLR"],
    },
]

for cfg in plot_configs:
  ax = cfg["ax"]
  y_t = cfg["y_true"]
  y_p = cfg["y_pred"]

  ax.scatter(
      y_t,
      y_p,
      alpha=0.7,
      color=cfg["color"],
      edgecolors=cfg["edge"],
      label="Predictions",
  )

  # 1:1 Reference Line
  min_val = min(y_t.min(), y_p.min())
  max_val = max(y_t.max(), y_p.max())
  ax.plot(
      [min_val, max_val],
      [min_val, max_val],
      "r--",
      linewidth=2,
      label="1:1 Ideal Line",
  )

  ax.set_title(cfg["title"], fontsize=12, fontweight="bold")
  ax.set_xlabel("PM2.5 Reference Concentration (µg/m³)", fontsize=10)
  ax.set_ylabel("PM2.5 Predicted Concentration (µg/m³)", fontsize=10)
  ax.legend(loc="upper left")
  ax.grid(True, linestyle="--", alpha=0.5)

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
          boxstyle="round,pad=0.5",
          facecolor="white",
          alpha=0.9,
          edgecolor="#CBD5E1",
      ),
  )

plt.tight_layout()
plt.savefig(
    os.path.join(saveGraph_directory, 'PM2.5_Base_Models_Full_Set_Validation_Scatterplots.png'), dpi=400
)
plt.show()
