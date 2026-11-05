# ==================== Test set: R² / RMSE Evaluation ====================
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# ==================== 1. Reload data and model ====================
# Please ensure these paths match exactly with those set when you ran the model previously
DATA_PATH = "/Users/sunzhuo/Sun/data.xlsx"
OUTPUT_DIR = "/Users/sunzhuo/Sun/MLP/NSGA2_stacking_results"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_nsga2_stacking_model.joblib")

# Read raw data
df = pd.read_excel(DATA_PATH)
X_raw = df.iloc[:, 1:-1].copy()
y_raw = pd.to_numeric(df.iloc[:, -1], errors="coerce")

# Filter out invalid rows
valid = y_raw.notna()
X_raw = X_raw.loc[valid].reset_index(drop=True)
y = y_raw.loc[valid].reset_index(drop=True)

# Split again using the original random seed to ensure X_test and y_test are exactly the same as before
TEST_SIZE = 0.20
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Load the trained final stacking model (which already includes the Preprocessing Pipeline)
final_stacking_model = joblib.load(MODEL_PATH)

# ==================== 2. Prediction and Evaluation ====================
# Use the trained final stacking model to directly predict on the 20% test set
y_test_pred_final = final_stacking_model.predict(X_test)

# Calculate R² and RMSE
r2_final = r2_score(y_test, y_test_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test, y_test_pred_final))

print("\n" + "=" * 72)
print("Final Stacking Model — Test Set Performance")
print("=" * 72)
print(f"Test R²   = {r2_final:.4f}")
print(f"Test RMSE = {rmse_final:.4f} °C")
print("=" * 72)