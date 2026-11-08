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
saveCSV_directory = "CSV_Results"


# 1. Load separate Training and Testing CSV datasets
df_train = pd.read_csv("Dataset/Basic_dataset.csv").dropna()
df_test = pd.read_csv("Dataset/Meta_dataset.csv").dropna()
df_test_results = df_test.copy() #Create a copy of the original test DataFrame

# 2. Define Features and Target (must match exact CSV column headers)
feature_cols_PM2_5 = ['temp', 'RH', 'PM10(Smoke)', 'PM10(KCl)', 'PM2.5(Smoke)', 'PM1(KCl)', 'PM1(Smoke)'] 
target_col_PM2_5 = 'PM2.5(KCl)'                 

X_train_PM2_5 = df_train[feature_cols_PM2_5]
y_train_PM2_5 = df_train[target_col_PM2_5]

X_test_PM2_5 = df_test[feature_cols_PM2_5]
y_test_PM2_5 = df_test[target_col_PM2_5]

# 3. Feature Scaling (Crucial for SVR distance calculations)
scaler_X = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train_PM2_5)
X_test_scaled = scaler_X.transform(X_test_PM2_5)

joblib.dump(scaler_X, os.path.join(saveModel_directory, 'PM2.5_Base_svr_linear_scaler_model.joblib'))

# 4. Initialize SVR with explicit hyperparameter control

SVR_model_LINEAR = SVR(
    kernel='linear',    # Radial Basis Function kernel
    C=10.0,          # Penalty parameter C (controls support vector force cap)
    epsilon=.01,     # Epsilon-insensitive tube margin
)


# 5. Train the SVR model on the scaled training dataset
SVR_model_LINEAR.fit(X_train_scaled, y_train_PM2_5)

joblib.dump(SVR_model_LINEAR, os.path.join(saveModel_directory, 'PM2.5_Base_svr_linear_model.joblib')) # Save the SVR model to a file for future use

# 6. Predict on the separate scaled test dataset
y_pred_LINEAR = SVR_model_LINEAR.predict(X_test_scaled)


df_test_results['prediction_linear'] = y_pred_LINEAR #Add the predictions as a new column

df_test_results.to_csv(os.path.join(saveCSV_directory, "PM2.5_SVR_dataset_with_predictions_linear.csv"), index=False) # Save the new DataFrame to a CSV file

# 7. Evaluate model performance metrics
mae_LINEAR = mean_absolute_error(y_test_PM2_5, y_pred_LINEAR)
rmse_LINEAR = root_mean_squared_error(y_test_PM2_5, y_pred_LINEAR)
r2_LINEAR = r2_score(y_test_PM2_5, y_pred_LINEAR)

print("\n--- PM2.5 Linear SVR Model Evaluation Metrics ---")
print(f"R² Score                      : {r2_LINEAR:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_LINEAR:.4f}")
print(f"Mean Absolute Error (MAE) : {mae_LINEAR:.4f}")

# 8. Graph Validated Results (2x2 Layout: Top Row = RBF, Bottom Row = Linear)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# GRAPH 1: Actual vs. Predicted Scatter Plot Linear SVR
ax1.scatter(y_test_PM2_5, y_pred_LINEAR, alpha=0.7, color='#0D9488', edgecolors='#0F766E', label='Validation Samples')

min_val = min(y_test_PM2_5.min(), y_pred_LINEAR.min())
max_val = max(y_test_PM2_5.max(), y_pred_LINEAR.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Ideal Line')

ax1.set_title('PM2.5 SVR: Linear', fontsize=12, fontweight='bold')
ax1.set_xlabel('Reference BAM PM2.5 (µg/m³)', fontsize=11)
ax1.set_ylabel('SVR Predicted PM2.5 (µg/m³)', fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

metrics_text = f"$R^2$ = {r2_LINEAR:.3f}\nRMSE = {rmse_LINEAR:.2f} µg/m³\nMAE = {mae_LINEAR:.2f} µg/m³"
ax1.text(0.60, 0.08, metrics_text, transform=ax1.transAxes, fontsize=11,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CBD5E1'))

# GRAPH 2: Time-Series Tracking Plot
samples_to_plot = 100 
sample_index = np.arange(1, min(len(y_test_PM2_5), samples_to_plot) + 1)

ax2.plot(sample_index, np.array(y_test_PM2_5)[:samples_to_plot], label='Actual (Reference)', color='#1E293B', linewidth=2, marker='o', markersize=3)
ax2.plot(sample_index, np.array(y_pred_LINEAR)[:samples_to_plot], label='SVR Prediction', color='#0EA5E9', linestyle='--', linewidth=2, marker='x', markersize=3)

ax2.set_title(f'PM2.5 Linear SVR Time-Series Validation (First {samples_to_plot} Samples)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Sequence / Time Index', fontsize=11)
ax2.set_ylabel('PM2.5 Concentration (µg/m³)', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)


plt.tight_layout()
plt.savefig(os.path.join(saveGraph_directory, 'PM2.5_Linear_SVR_Validation_Results.png'), dpi=300)
plt.show()