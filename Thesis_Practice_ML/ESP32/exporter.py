"""
# Exporter for Random Forest Model to C Code
#This script loads a pre-trained Random Forest model and exports each decision tree as a C function.

import joblib

# 1. Load your original trained model
print('Loading original model...')
rf_model = joblib.load('Models/Base_rf_model.joblib')


# 2. Function to convert a single decision tree into C code
def tree_to_c(tree, tree_id):
  tree_ = tree.tree_

  def recurse(node, depth):
    indent = '  ' * depth
    # Check if leaf node
    if tree_.children_left[node] == tree_.children_right[node]:
      return f'{indent}return {tree_.value[node][0][0]:.6f};\n'
    else:
      feature = tree_.feature[node]
      threshold = tree_.threshold[node]
      left = recurse(tree_.children_left[node], depth + 1)
      right = recurse(tree_.children_right[node], depth + 1)
      return (
          f'{indent}if (x[{feature}] <= {threshold:.6f}f) {{\n'
          f'{left}'
          f'{indent}}} else {{\n'
          f'{right}'
          f'{indent}}}\n'
      )

  code = f'static double predict_tree_{tree_id}(const double* x) {{\n'
  code += recurse(0, 1)
  code += '}\n\n'
  return code


# 3. Export all 1,000 trees sequentially
total_trees = len(rf_model.estimators_)
print(f'Exporting {total_trees} trees to rf_model.h...')

with open('rf_model.h', 'w') as f:
  f.write('// Auto-generated C Header for 1,000-Tree Random Forest\n')
  f.write('#ifndef RF_MODEL_H\n#define RF_MODEL_H\n\n')

  # Write individual tree functions
  for i, estimator in enumerate(rf_model.estimators_):
    f.write(tree_to_c(estimator, i))
    if (i + 1) % 100 == 0:
      print(f'Progress: {i + 1} / {total_trees} trees exported...')

  # Write top-level wrapper function to average all tree outputs
  f.write('double predict_rf(const double* x) {\n')
  f.write('    double sum = 0.0;\n')
  for i in range(total_trees):
    f.write(f'    sum += predict_tree_{i}(x);\n')
  f.write(f'    return sum / {total_trees}.0;\n')
  f.write('}\n\n')
  f.write('#endif\n')

print('Export finished successfully! File saved as rf_model.h')
"""

"""
# Exporter

import sys
import joblib
import m2cgen as m2c

# Increase Python's recursion limit for large tree ensembles
sys.setrecursionlimit(30000)    #has to be set to a high value for large tree ensembles (e.g., 10000) to avoid RecursionError during model export

# 1. Load trained models and scaler
rf_model = joblib.load('Models/Base_rf_model.joblib')
svr_model = joblib.load('Models/Base_svr_model.joblib')
scaler_X = joblib.load('Models/Base_svr_scaler_model.joblib')
mlr_model = joblib.load('Models/Meta_MLR_model.joblib')

# 2. Export RF and SVR models to pure C code
c_code_rf = m2c.export_to_c(rf_model)
c_code_svr = m2c.export_to_c(svr_model)

with open('rf_model.h', 'w') as f:
  f.write('double predict_rf(double* x) {\n' + c_code_rf + '\n}')

with open('svr_model.h', 'w') as f:
  f.write('double predict_svr(double* x) {\n' + c_code_svr + '\n}')

# 3. Extract StandardScaler parameters (Mean & Scale/Std Dev)
print("// Copy these into your ESP32 Arduino code:")
print("--- SCALER PARAMETERS ---")
print(f"const float SCALER_MEAN[] = {{{', '.join(map(str, scaler_X.mean_))}}};")
print(
    f"const float SCALER_SCALE[] = {{{', '.join(map(str, scaler_X.scale_))}}};"
)

# 4. Extract MLR Weights and Intercept
print("\n--- MLR META-LEARNER PARAMETERS ---")
print(f"const float W_RF = {mlr_model.coef_[0]}f;")
print(f"const float W_SVR = {mlr_model.coef_[1]}f;")
print(f"const float MLR_BIAS = {mlr_model.intercept_}f;")

"""