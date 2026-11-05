import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

FONT_SIZE_AXIS_LABEL  = 24
FONT_SIZE_TITLE       = 26
FONT_SIZE_TICK        = 18
FONT_SIZE_ANNOTATION  = 24
FONT_WEIGHT           = "bold"
FONT_FAMILY           = "Times New Roman"

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         [FONT_FAMILY, "Liberation Serif",
                           "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "axes.unicode_minus": False,
    "font.weight":        FONT_WEIGHT,
    "axes.titlesize":     FONT_SIZE_TITLE,
    "axes.titleweight":   FONT_WEIGHT,
    "axes.labelsize":     FONT_SIZE_AXIS_LABEL,
    "axes.labelweight":   FONT_WEIGHT,
    "xtick.labelsize":    FONT_SIZE_TICK,
    "ytick.labelsize":    FONT_SIZE_TICK,
})


print("\n" + "=" * 72)
print("Plotting OOF scatter plot for the best base learner")
print("=" * 72)

if "oof_residuals" not in globals():
    raise NameError("Cannot find oof_residuals, please ensure this appended code runs at the end of the original script.")
if "y_train" not in globals():
    raise NameError("Cannot find y_train.")

available_names = list(oof_residuals.keys())
print(f"Base learners contained in oof_residuals: {available_names}")

if len(available_names) == 0:
    raise ValueError("oof_residuals is empty, cannot calculate OOF metrics.")

y_train_array = np.asarray(y_train, dtype=float).reshape(-1)

base_oof_preds = {}
base_oof_metrics = {}

for _name in available_names:
    _residual = np.asarray(oof_residuals[_name], dtype=float).reshape(-1)
    if len(_residual) != len(y_train_array):
        raise ValueError(
            f"OOF residual length {len(_residual)} for {_name} "
            f"does not match y_train length {len(y_train_array)}."
        )
    oof_pred_array = y_train_array - _residual
    base_oof_preds[_name] = oof_pred_array
    base_oof_metrics[_name] = {
        "rmse": float(np.sqrt(mean_squared_error(y_train_array, oof_pred_array))),
        "r2":   float(r2_score(y_train_array, oof_pred_array)),
    }

best_base_name = min(base_oof_metrics, key=lambda x: base_oof_metrics[x]["rmse"])

y_true_best = y_train_array
y_pred_best = base_oof_preds[best_base_name]

rmse_best = base_oof_metrics[best_base_name]["rmse"]
r2_best   = base_oof_metrics[best_base_name]["r2"]

_display_name_map = {
    "rf": "RF", "xgb": "XGBoost", "lgbm": "LightGBM", "gbr": "GBR",
    "catboost": "CatBoost", "knn": "KNN", "dt": "DT", "svr": "SVR",
    "bpnn": "BPNN", "extra_trees": "ExtraTrees", "gpr": "GPR",
    "krr": "KRR", "adaboost": "AdaBoost", "hist_gbr": "HistGBR", "ebm": "EBM",
}
title_name = _display_name_map.get(best_base_name.lower(), best_base_name.upper())

fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    y_true_best, y_pred_best,
    alpha=0.6, s=40,
    color="#75b7ee", edgecolors="white"
)

ax.plot(
    [y_true_best.min(), y_true_best.max()],
    [y_true_best.min(), y_true_best.max()],
    "k--", lw=1
)

ax.set_xlabel(
    "Actual",
    fontsize=FONT_SIZE_AXIS_LABEL,
    fontweight=FONT_WEIGHT
)
ax.set_ylabel(
    "OOF Predicted",
    fontsize=FONT_SIZE_AXIS_LABEL,
    fontweight=FONT_WEIGHT
)

ax.set_title(
    f"{title_name}-Best single model  ",
    fontsize=FONT_SIZE_TITLE,
    fontweight=FONT_WEIGHT
)

ax.text(
    0.95, 0.05,
    f"R² = {r2_best:.3f}\nRMSE = {rmse_best:.3f}",
    transform=ax.transAxes,
    verticalalignment="bottom",
    horizontalalignment="right",
    fontsize=FONT_SIZE_ANNOTATION,
    fontweight=FONT_WEIGHT,
    color="#75b7ee"
)

ax.tick_params(labelsize=FONT_SIZE_TICK, width=2)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(FONT_SIZE_TICK)
    label.set_fontweight(FONT_WEIGHT)

plt.tight_layout()

_best_oof_png = os.path.join(OUTPUT_DIR, "b_best_base_oof.png")
plt.savefig(_best_oof_png, dpi=600, bbox_inches="tight")
plt.show()
plt.close()

print(f"Best base learner: {title_name}  (original name: {best_base_name})")
print(f"OOF R²       = {r2_best:.4f}")
print(f"OOF RMSE     = {rmse_best:.4f}")
print(f"Number of valid samples   = {len(y_true_best)}")
print(f"OOF scatter plot saved to: {_best_oof_png}")

_oof_rows = []
for _name in available_names:
    _oof_rows.append({
        "Model": _name,
        "Display_Name": _display_name_map.get(_name.lower(), _name.upper()),
        "OOF_R2": base_oof_metrics[_name]["r2"],
        "OOF_RMSE": base_oof_metrics[_name]["rmse"],
        "Is_Best": bool(_name == best_base_name),
    })

_oof_df = pd.DataFrame(_oof_rows).sort_values("OOF_RMSE").reset_index(drop=True)
_oof_csv = os.path.join(OUTPUT_DIR, "base_oof_metrics.csv")
_oof_df.to_csv(_oof_csv, index=False, encoding="utf-8-sig")
print(f"OOF metrics for each base learner saved to: {_oof_csv}")
print("=" * 72)