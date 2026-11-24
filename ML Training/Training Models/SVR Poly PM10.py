import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib
import os

saveModel_directory = "Output Models"
saveGraph_directory = "Graphs"


# 1. Load separate Training and Testing CSV datasets
df_train = pd.read_csv("Dataset/PM10_Basic.csv").dropna()
df_test = pd.read_csv("Dataset/PM10_Meta.csv").dropna()

# Define Features and Target (must match exact CSV column headers)
feature_cols_PM10 = ['Temperature', 'Humidity', 'PM10(KCl)', 'PM10(Smoke)', 'PM2.5(KCl)', 'PM2.5(Smoke)', 'PM1(KCl)', 'PM1(Smoke)'] 
target_col_PM10 = 'BAM_PM10'                  

X_train_PM10 = df_train[feature_cols_PM10]
y_train_PM10 = df_train[target_col_PM10]

X_test_PM10 = df_test[feature_cols_PM10]
y_test_PM10 = df_test[target_col_PM10]

# 3. Feature Scaling (Crucial for SVR distance calculations)
scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train_PM10)
X_test_scaled = scaler_X.transform(X_test_PM10)

joblib.dump(scaler_X, os.path.join(saveModel_directory, 'PM10_Base_svr_poly_scaler_model.joblib'))

# 4. Initialize SVR with explicit hyperparameter control

SVR_model_Poly = SVR(
    kernel='poly',    #Polynomial kernel
    degree=3,       # Degree of the polynomial kernel function
    gamma='scale',  # Kernel coefficient for 'poly'
    coef0=1.0,      # Independent term in polynomial kernel
    C=10.0,          # Penalty parameter C (controls support vector force cap)
    epsilon=.01,     # Epsilon-insensitive tube margin
)

# 5. Train the SVR model on the scaled training dataset
SVR_model_Poly.fit(X_train_scaled, y_train_PM10)

joblib.dump(SVR_model_Poly, os.path.join(saveModel_directory, 'PM10_Base_svr_poly_model.joblib')) # Save the SVR model to a file for future use

# 6. Predict on the separate scaled test dataset
y_pred_POLY = SVR_model_Poly.predict(X_test_scaled)

# 7. Evaluate model performance metrics
mae_POLY = mean_absolute_error(y_test_PM10, y_pred_POLY)
rmse_POLY = root_mean_squared_error(y_test_PM10, y_pred_POLY)
r2_POLY = r2_score(y_test_PM10, y_pred_POLY)

print("\n--- PM10 Polynomial SVR Model Evaluation Metrics ---")
print(f"R² Score                      : {r2_POLY:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_POLY:.4f}")
print(f"Mean Absolute Error (MAE) : {mae_POLY:.4f}")

# 8. Graph Validated Results (2x2 Layout: Top Row = RBF, Bottom Row = Linear)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# GRAPH 1: Actual vs. Predicted Scatter Plot Linear SVR
ax1.scatter(y_test_PM10, y_pred_POLY, alpha=0.7, color='#0D9488', edgecolors='#0F766E', label='Validation Samples')

min_val = min(y_test_PM10.min(), y_pred_POLY.min())
max_val = max(y_test_PM10.max(), y_pred_POLY.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Ideal Line')

ax1.set_title('PM10 SVR: Polynomial', fontsize=12, fontweight='bold')
ax1.set_xlabel('Reference BAM PM10 (µg/m³)', fontsize=11)
ax1.set_ylabel('SVR Predicted PM10 (µg/m³)', fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

metrics_text = f"$R^2$ = {r2_POLY:.3f}\nRMSE = {rmse_POLY:.2f} µg/m³\nMAE = {mae_POLY:.2f} µg/m³"
ax1.text(0.60, 0.08, metrics_text, transform=ax1.transAxes, fontsize=11,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CBD5E1'))

# GRAPH 2: Time-Series Tracking Plot
samples_to_plot = 100 
sample_index = np.arange(1, min(len(y_test_PM10), samples_to_plot) + 1)

ax2.plot(sample_index, np.array(y_test_PM10)[:samples_to_plot], label='Actual (Reference)', color='#1E293B', linewidth=2, marker='o', markersize=3)
ax2.plot(sample_index, np.array(y_pred_POLY)[:samples_to_plot], label='SVR Prediction', color='#0EA5E9', linestyle='--', linewidth=2, marker='x', markersize=3)

ax2.set_title(f'PM10 Polynomial SVR Time-Series Validation (First {samples_to_plot} Samples)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Sequence / Time Index', fontsize=11)
ax2.set_ylabel('PM10 Concentration (µg/m³)', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)


plt.tight_layout()
plt.savefig(os.path.join(saveGraph_directory, 'PM10_SVR_Polynomial_Validation_Results.png'), dpi=300)
plt.show()