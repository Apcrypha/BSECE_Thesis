import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# 1. Load separate Training and Testing CSV datasets
df_train = pd.read_csv("Dataset\Basic_dataset.csv").dropna()
df_test = pd.read_csv("Dataset\Meta_dataset.csv").dropna()
df_test_results = df_test.copy() #Create a copy of the original test DataFrame

# 2. Define Features and Target (must match exact CSV column headers)
feature_cols = ['S3', 'S4', 'S1','S2'] 
target_col = 'Ref'                

X_train = df_train[feature_cols]
y_train = df_train[target_col]

X_test = df_test[feature_cols]
y_test = df_test[target_col]

# 3. Feature Scaling (Crucial for SVR distance calculations)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# 4. Initialize SVR with explicit hyperparameter control
svr_model = SVR(
    kernel='rbf',    # Radial Basis Function kernel
    C=25.0,          # Penalty parameter C (controls support vector force cap)
    epsilon=0.01,     # Epsilon-insensitive tube margin
    gamma=0.07        # Kernel coefficient (controls influence radius of each SV; or 'scale'/'auto')
)

# 5. Train the SVR model on the scaled training dataset
svr_model.fit(X_train_scaled, y_train)

# 6. Predict on the separate scaled test dataset
y_pred = svr_model.predict(X_test_scaled)

df_test_results['prediction'] = y_pred #Add the predictions as a new column

df_test_results.to_csv("SVR_dataset_with_predictions.csv", index=False) # Save the new DataFrame to a CSV file

# 7. Evaluate model performance metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- SVR Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE) : {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score                      : {r2:.4f}")

# 8. Graph Validated Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# GRAPH 1: Actual vs. Predicted Scatter Plot
ax1.scatter(y_test, y_pred, alpha=0.7, color='#0D9488', edgecolors='#0F766E', label='Validation Samples')

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Ideal Line')

ax1.set_title('SVR: Actual vs. Predicted PM₂.₅', fontsize=12, fontweight='bold')
ax1.set_xlabel('Reference BAM (µg/m³)', fontsize=11)
ax1.set_ylabel('SVR Predicted PM₂.₅ (µg/m³)', fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

metrics_text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} µg/m³\nMAE = {mae:.2f} µg/m³"
ax1.text(0.60, 0.08, metrics_text, transform=ax1.transAxes, fontsize=11,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CBD5E1'))

# GRAPH 2: Time-Series Tracking Plot
samples_to_plot = 100 
sample_index = np.arange(1, min(len(y_test), samples_to_plot) + 1)

ax2.plot(sample_index, np.array(y_test)[:samples_to_plot], label='Actual (Reference)', color='#1E293B', linewidth=2, marker='o', markersize=3)
ax2.plot(sample_index, np.array(y_pred)[:samples_to_plot], label='SVR Prediction', color='#0EA5E9', linestyle='--', linewidth=2, marker='x', markersize=3)

ax2.set_title(f'SVR Time-Series Validation (First {samples_to_plot} Samples)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Sequence / Time Index', fontsize=11)
ax2.set_ylabel('PM Concentration (µg/m³)', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('svr_validation_results.png', dpi=300)
plt.show()