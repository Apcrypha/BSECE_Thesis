import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV

saveModel_directory = "Output Models"
saveGraph_directory = "Graphs"
os.makedirs(saveModel_directory, exist_ok=True)
os.makedirs(saveGraph_directory, exist_ok=True)

# 1. Load separate Training and Testing CSV datasets
df_train = pd.read_csv("Dataset/PM2.5_Basic.csv").dropna().copy()
df_test = pd.read_csv("Dataset/PM2.5_Meta.csv").dropna().copy()


# 2. Physics-Informed Feature Engineering (Crucial for small datasets)
def add_engineered_features(df):
  eps = 1e-5  # Prevent division by zero
  # Hygroscopic aerosol growth proxy (captures humidity swelling)
  rh_clipped = df["Humidity"].clip(upper=99.0)
  df["Hygro_Factor"] = rh_clipped / (100.0 - rh_clipped)

  # Sensor channel averages (reduces optical noise)
  df["PM2.5_Mean"] = (df["PM2.5(KCl)"] + df["PM2.5(Smoke)"]) / 2.0
  df["PM10_Mean"] = (df["PM10(KCl)"] + df["PM10(Smoke)"]) / 2.0

  # Particle size distribution ratios
  df["Ratio_PM10_PM25"] = df["PM10(Smoke)"] / (df["PM2.5(Smoke)"] + eps)
  df["Ratio_PM25_PM1"] = df["PM2.5(Smoke)"] / (df["PM1(Smoke)"] + eps)
  df["Temp_RH_Interaction"] = df["Temperature"] * df["Humidity"]
  return df


df_train = add_engineered_features(df_train)
df_test = add_engineered_features(df_test)

feature_cols_PM2_5 = [
    "Temperature",
    "Humidity",
    "PM10(KCl)",
    "PM10(Smoke)",
    "PM2.5(KCl)",
    "PM2.5(Smoke)",
    "PM1(KCl)",
    "PM1(Smoke)",
    "Hygro_Factor",
    "PM2.5_Mean",
    "PM10_Mean",
    "Ratio_PM10_PM25",
    "Ratio_PM25_PM1",
    "Temp_RH_Interaction",
]
target_col_PM2_5 = "BAM_PM2.5"

X_train = df_train[feature_cols_PM2_5]
y_train = df_train[target_col_PM2_5]

X_test = df_test[feature_cols_PM2_5]
y_test = df_test[target_col_PM2_5]

# 3. Focused Hyperparameter Grid Tailored for Small Datasets (N = 235)
# Using ExtraTreesRegressor (or swap back to RandomForestRegressor to compare)
param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [6, 10, 15, None],
    "max_features": [0.7, 0.85, 1.0],  # Ensure primary PM features are evaluated
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3],
}

base_model = ExtraTreesRegressor(random_state=89, n_jobs=-1)

tuner = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

tuner.fit(X_train, y_train)
rf_model = tuner.best_estimator_

print("\n--- Best Hyperparameters Found ---")
for param, val in tuner.best_params_.items():
  print(f"{param:20s}: {val}")

joblib.dump(
    rf_model, os.path.join(saveModel_directory, "PM2.5_Base_rf_model.joblib")
)

# 4. Predict & Evaluate
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Tuned PM2.5 Base Model Evaluation Metrics ---")
print(f"R² Score                      : {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE)     : {mae:.4f}")

# 5. Plot Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    color="#0EA5E9",
    edgecolors="#0369A1",
    label="Validation Samples",
)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax1.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    linewidth=2,
    label="1:1 Ideal Line",
)

ax1.set_title("Tuned PM2.5 Base Model", fontsize=12, fontweight="bold")
ax1.set_xlabel("Reference BAM PM2.5 (µg/m³)", fontsize=11)
ax1.set_ylabel("Predicted PM2.5 (µg/m³)", fontsize=11)
ax1.legend(loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.5)

metrics_text = (
    f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} µg/m³\nMAE = {mae:.2f} µg/m³"
)
ax1.text(
    0.60,
    0.08,
    metrics_text,
    transform=ax1.transAxes,
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        alpha=0.9,
        edgecolor="#CBD5E1",
    ),
)

samples_to_plot = 100
sample_index = np.arange(1, min(len(y_test), samples_to_plot) + 1)

ax2.plot(
    sample_index,
    np.array(y_test)[:samples_to_plot],
    label="Actual (Reference)",
    color="#1E293B",
    linewidth=2,
    marker="o",
    markersize=3,
)
ax2.plot(
    sample_index,
    np.array(y_pred)[:samples_to_plot],
    label="Model Prediction",
    color="#F97316",
    linestyle="--",
    linewidth=2,
    marker="x",
    markersize=3,
)

ax2.set_title(
    f"PM2.5 Time-Series Validation (First {samples_to_plot} Samples)",
    fontsize=12,
    fontweight="bold",
)
ax2.set_xlabel("Sequence / Time Index", fontsize=11)
ax2.set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=11)
ax2.legend(loc="upper right")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(
    os.path.join(saveGraph_directory, "PM2.5_RFR_Validation_Results.png"),
    dpi=300,
)
plt.show()