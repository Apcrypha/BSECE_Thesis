import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib
import os

saveModel_directory = "Models"
saveGraph_directory = "Graphs"
saveCSV_directory = "CSV_Results"

# Import / Load the pre-trained models from disk
rf_model = joblib.load('Models/Base_rf_model.joblib')
svr_model_RBF = joblib.load('Models/Base_svr_rbf_model.joblib')
svr_model_LINEAR = joblib.load('Models/Base_svr_linear_model.joblib')
scaler_X = joblib.load('Models/Base_svr_scaler_model.joblib')


# Load separate Training and Testing CSV datasets
df_train = pd.read_csv("Dataset/Meta_dataset.csv").dropna()
df_test = pd.read_csv("Dataset/Validation_dataset.csv").dropna()
df_test_results = df_test.copy() #Create a copy of the original test DataFrame

# Define Features and Target (must match exact CSV column headers)
feature_cols = ['S3', 'S4', 'S1','S2'] 
target_col = 'Ref'                

# Assign the features and target for meta-model training and testing
X_meta_train_raw = df_train[feature_cols]
y_meta = df_train[target_col]

X_meta_test_raw = df_test[feature_cols]
y_test = df_test[target_col]

# Scale the meta-model features using the pre-trained scaler
X_meta_scaled = scaler_X.transform(X_meta_train_raw)
X_meta_test_scaled = scaler_X.transform(X_meta_test_raw)

# Predict using the base models
pred_rf_meta_train = rf_model.predict(X_meta_train_raw)
pred_svr_RBF_meta_train = svr_model_RBF.predict(X_meta_scaled)
pred_svr_LINEAR_meta_train = svr_model_LINEAR.predict(X_meta_scaled)

pred_rf_meta_test = rf_model.predict(X_meta_test_raw)
pred_svr_RBF_meta_test = svr_model_RBF.predict(X_meta_test_scaled)
pred_svr_LINEAR_meta_test = svr_model_LINEAR.predict(X_meta_test_scaled)

# Stack the predictions from the base models to create the meta-model training and testing datasets
X_meta_train_stacked = np.column_stack((pred_rf_meta_train, pred_svr_RBF_meta_train, pred_svr_LINEAR_meta_train))
X_meta_test_stacked = np.column_stack((pred_rf_meta_test, pred_svr_RBF_meta_test, pred_svr_LINEAR_meta_test))

mlr_model = LinearRegression()
mlr_model.fit(X_meta_train_stacked, y_meta)

joblib.dump(mlr_model, os.path.join(saveModel_directory, 'Meta_MLR_model.joblib'))

y_pred = mlr_model.predict(X_meta_test_stacked)

df_test_results['prediction'] = y_pred #Add the predictions as a new column

df_test_results.to_csv(os.path.join(saveCSV_directory, "MLR_dataset_with_predictions.csv"), index=False) # Save the new DataFrame to a CSV file

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

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
plt.savefig(os.path.join(saveGraph_directory, 'MLR_validation_results.png'), dpi=300)
plt.show()