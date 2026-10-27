import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib
import os

saveModel_directory = "Models"
saveGraph_directory = "Graphs"
saveCSV_directory = "CSV_Results"


# 1. Load separate Training and Testing CSV datasets
df_train = pd.read_csv("Dataset/Basic_dataset.csv").dropna()
df_test = pd.read_csv("Dataset/Meta_dataset.csv").dropna()
df_test_results = df_test.copy() #Create a copy of the original test DataFrame


# 2. Define Features and Target (must match exact CSV column headers)
feature_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income'] 
target_col = 'median_house_value'               

# 3. Assign 100% of train CSV to training set, test CSV to testing set
X_train = df_train[feature_cols]
y_train = df_train[target_col]

X_test = df_test[feature_cols]
y_test = df_test[target_col]

# 4. Initialize Random Forest with explicit hyperparameter control
rf_model = RandomForestRegressor(
    n_estimators=1000,     # Total number of decision trees in the forest
    max_depth=17,         # Maximum depth limit for each tree (prevents overfitting)
    max_features=3,  # Features considered at each split ('sqrt', 'log2', or integer/float)
    random_state=61     # Seed for reproducible results
)

# 5. Train the model on the full training dataset
rf_model.fit(X_train, y_train)

joblib.dump(rf_model, os.path.join(saveModel_directory, 'Base_rf_model.joblib')) # Save the RF model to a file for future use

# 6. Predict on the separate test dataset
y_pred = rf_model.predict(X_test)

df_test_results['prediction'] = y_pred #Add the predictions as a new column

df_test_results.to_csv(os.path.join(saveCSV_directory, "RF_dataset_with_predictions.csv"), index=False) # Save the new DataFrame to a CSV file

# 7. Evaluate model performance including MAE
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Evaluation Metrics ---")
print(f"Mean Absolute Error (MAE) : {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score                      : {r2:.4f}")


# 8. Create side-by-side plot layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ----------------------------------------------------
# GRAPH 1: Actual vs. Predicted Scatter Plot
# ----------------------------------------------------
ax1.scatter(y_test, y_pred, alpha=0.7, color='#0EA5E9', edgecolors='#0369A1', label='Validation Samples')

# Add 1:1 Ideal Line (Where predictions perfectly match reference)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Ideal Line')

ax1.set_title('Actual vs. Predicted Concentration', fontsize=12, fontweight='bold')
ax1.set_xlabel('Reference BAM (µg/m³)', fontsize=11)
ax1.set_ylabel('Model Predicted PM (µg/m³)', fontsize=11)
ax1.legend(loc='upper left')
ax1.grid(True, linestyle='--', alpha=0.5)

# Metrics Box Overlay
metrics_text = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} µg/m³\nMAE = {mae:.2f} µg/m³"
ax1.text(0.60, 0.08, metrics_text, transform=ax1.transAxes, fontsize=11,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#CBD5E1'))

# ----------------------------------------------------
# GRAPH 2: Time-Series Tracking Plot
# ----------------------------------------------------
# Convert to numpy arrays or plot a subset (e.g., first 100 samples) for visual clarity
samples_to_plot = 100 
sample_index = np.arange(1, min(len(y_test), samples_to_plot) + 1)

ax2.plot(sample_index, np.array(y_test)[:samples_to_plot], label='Actual (Reference)', color='#1E293B', linewidth=2, marker='o', markersize=3)
ax2.plot(sample_index, np.array(y_pred)[:samples_to_plot], label='Model Prediction', color='#F97316', linestyle='--', linewidth=2, marker='x', markersize=3)

ax2.set_title(f'Time-Series Tracking (First {samples_to_plot} Samples)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Sequence / Time Index', fontsize=11)
ax2.set_ylabel('PM Concentration (µg/m³)', fontsize=11)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

# Adjust layout and display / save graph
plt.tight_layout()
plt.savefig(os.path.join(saveGraph_directory, 'rf_validation_results.png'), dpi=300)
plt.show()