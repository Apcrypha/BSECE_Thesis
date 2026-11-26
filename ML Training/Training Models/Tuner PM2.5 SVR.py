import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# 1. Load separate Training and Testing CSV datasets
df_train = pd.read_csv("Dataset/PM2.5_Basic.csv").dropna()
df_test = pd.read_csv("Dataset/PM2.5_Meta.csv").dropna()

# 2. Define Features and Target
feature_cols_PM2_5 = [
    'Temperature',
    'Humidity',
    'PM10(KCl)',
    'PM10(Smoke)',
    'PM2.5(KCl)',
    'PM2.5(Smoke)',
    'PM1(KCl)',
    'PM1(Smoke)',
]
target_col_PM2_5 = 'BAM_PM2.5'

X_train_PM2_5 = df_train[feature_cols_PM2_5]
y_train_PM2_5 = df_train[target_col_PM2_5]

X_test_PM2_5 = df_test[feature_cols_PM2_5]
y_test_PM2_5 = df_test[target_col_PM2_5]

# 3. Feature Scaling (Fit ONLY on X_train, transform X_test)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_PM2_5)
X_test_scaled = scaler_X.transform(X_test_PM2_5)

# 4. Fine-Grained Hyperparameter Grids for Each SVR Kernel
# (Includes Sigmoid as well if you are tuning all 4 scikit-learn SVR kernels)
kernel_grids = {
    'linear': {
        'kernel': ['linear'],
        'C': [
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
            40.0,
            50.0,
            75.0,
            100.0,
        ],
        'epsilon': [
            0.0005,
            0.001,
            0.0025,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.075,
            0.1,
            0.15,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
        ],
    },
    'rbf': {
        'kernel': ['rbf'],
        'C': [
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
            40.0,
            50.0,
            75.0,
            100.0,
            150.0,
            200.0,
        ],
        'gamma': [
            'scale',
            'auto',
            0.0005,
            0.001,
            0.0025,
            0.005,
            0.0075,
            0.01,
            0.02,
            0.03,
            0.05,
            0.07,
            0.1,
            0.15,
            0.2,
            0.3,
            0.5,
        ],
        'epsilon': [
            0.001,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.075,
            0.1,
            0.15,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
        ],
    },
    'poly': {
        'kernel': ['poly'],
        'degree': [2, 3, 4],
        'C': [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 50.0],
        'gamma': [
            'scale',
            'auto',
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.2,
        ],
        'coef0': [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
        'epsilon': [0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5],
    },
    'sigmoid': {
        'kernel': ['sigmoid'],
        'C': [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0],
        'gamma': ['scale', 'auto', 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05],
        'coef0': [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
        'epsilon': [0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5],
    },
}

# 5. Exhaustively Tune Every Kernel & Print Best Hyperparameters
for kernel_name, param_grid in kernel_grids.items():
  total_combos = np.prod([len(v) for v in param_grid.values()])
  print(f"\n{'='*60}")
  print(
      f"Tuning {kernel_name.upper()} SVR ({total_combos:,} hyperparameter"
      " combinations)..."
  )
  print(f"{'='*60}")

  # max_iter caps non-converging high-degree poly/sigmoid combinations so GridSearch finishes
  base_svr = SVR(max_iter=100000)

  tuner = GridSearchCV(
      estimator=base_svr,
      param_grid=param_grid,
      cv=5,  # 5-fold cross-validation strictly on X_train_scaled
      scoring='neg_root_mean_squared_error',
      n_jobs=-1,  # Uses all CPU cores
      verbose=1,
  )

  tuner.fit(X_train_scaled, y_train_PM2_5)
  best_model = tuner.best_estimator_

  # Evaluate on PM2.5_Meta.csv to verify generalization score alongside CV score
  y_pred = best_model.predict(X_test_scaled)
  r2 = r2_score(y_test_PM2_5, y_pred)
  rmse = root_mean_squared_error(y_test_PM2_5, y_pred)
  mae = mean_absolute_error(y_test_PM2_5, y_pred)

  print(f"\n>>> BEST {kernel_name.upper()} SVR HYPERPARAMETERS <<<")
  for param, val in tuner.best_params_.items():
    print(f"  {param:10s} = {val}")

  print(f"\n  Best 5-Fold CV RMSE : {-tuner.best_score_:.4f} µg/m³")
  print(
      f"  Test Set Metrics    : R² = {r2:.4f} | RMSE = {rmse:.4f} µg/m³ | MAE"
      f" = {mae:.4f} µg/m³"
  )