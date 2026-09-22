import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 1. Load separate Training and Testing CSV datasets
df_train = pd.read_csv("..\Basic_output.csv").dropna()
df_test = pd.read_csv("..\Validation_dataset.csv").dropna()

# 2. Define Base Features and Ground-Truth Target
feature_cols = ['temp', 'RH', 'raw_PM25']  # Input features for base learners
target_col = 'reference_BAM'  # Target label

X_train = df_train[feature_cols]
y_train = df_train[target_col]

X_test = df_test[feature_cols]
y_test = df_test[target_col]

# 3. Train Level-0 Base Learners (RF & Scaled SVR)
# --- Base Learner 1: Random Forest ---
rf_model = RandomForestRegressor(
    n_estimators=150, max_depth=12, max_features='sqrt', random_state=42
)
rf_model.fit(X_train, y_train)

# --- Base Learner 2: Support Vector Regression (Scaled) ---
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

svr_model = SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma=0.1)
svr_model.fit(X_train_scaled, y_train)

# 4. Generate Base Model Predictions to Form Level-1 Feature Matrices
# Training set meta-features (Level-0 predictions on train set)
pred_rf_train = rf_model.predict(X_train)
pred_svr_train = svr_model.predict(X_train_scaled)
X_meta_train = np.column_stack((pred_rf_train, pred_svr_train))

# Testing set meta-features (Level-0 predictions on test set)
pred_rf_test = rf_model.predict(X_test)
pred_svr_test = svr_model.predict(X_test_scaled)
X_meta_test = np.column_stack((pred_rf_test, pred_svr_test))

# 5. Train Level-1 Meta-Learner (Multiple Linear Regression)
mlr_model = LinearRegression()
mlr_model.fit(X_meta_train, y_train)

# 6. Predict Final Ensembled Concentrations on Test Dataset
y_pred = mlr_model.predict(X_meta_test)

# 7. Model Parameter Inspection & Evaluation Metrics
w_rf = mlr_model.coef_[0]
w_svr = mlr_model.coef_[1]
bias = mlr_model.intercept_

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- MLR Meta-Learner Equation Parameters ---")
print(f"Random Forest Weight (w1) : {w_rf:.4f}")
print(f"SVR Weight           (w2) : {w_svr:.4f}")
print(f"Bias Offset          (b)  : {bias:.4f}\n")

print("--- Stacked MLR Validation Metrics ---")
print(f"Mean Absolute Error (MAE) : {mae:.4f} µg/m³")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} µg/m³")
print(f"R² Score                      : {r2:.4f}")

# 8. Graph Validated Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# GRAPH 1: Actual vs. Predicted Scatter Plot
ax1.scatter(
    y_test,
    y_pred,
    alpha=0.7,
    color='#8B5CF6',
    edgecolors='#6D28D9',
    label='Validation Samples',
)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax1.plot(
    [min_val, max_val],
    [min_val, max_val],
    'r--',
    linewidth=2,
    label='1:1 Ideal Line',
)

ax1.set_title(
    'Stacked MLR: Actual vs. Predicted PM₂.₅', fontsize=12, fontweight='bold'
)
ax1.set_xlabel('Reference BAM (µg/m³)', fontsize=11)
ax1.set_ylabel('Stacked MLR Predicted PM₂.₅ (µg/m³)', fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

metrics_text = (
    f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} µg/m³\nMAE = {mae:.2f} µg/m³'
)
ax1.text(
    0.60,
    0.08,
    metrics_text,
    transform=ax1.transAxes,
    fontsize=11,
    bbox=dict(
        boxstyle='round,pad=0.5',
        facecolor='white',
        alpha=0.9,
        edgecolor='#CBD5E1',
    ),
)

# GRAPH 2: Time-Series Tracking Plot
samples_to_plot = 100
sample_index = np.arange(1, min(len(y_test), samples_to_plot) + 1)

ax2.plot(
    sample_index,
    np.array(y_test)[:samples_to_plot],
    label='Actual (Reference)',
    color='#1E293B',
    linewidth=2,
    marker='o',
    markersize=3,
)
ax2.plot(
    sample_index,
    np.array(y_pred)[:samples_to_plot],
    label='Stacked MLR Prediction',
    color='#8B5CF6',
    linestyle='--',
    linewidth=2,
    marker='x',
    markersize=3,
)

ax2.set_title(
    f'Stacked MLR Time-Series Tracking (First {samples_to_plot} Samples)',
    fontsize=12,
    fontweight='bold',
)
ax2.set_xlabel('Sequence / Time Index', fontsize=11)
ax2.set_ylabel('PM Concentration (µg/m³)', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('mlr_stacked_validation_results.png', dpi=300)
plt.show()