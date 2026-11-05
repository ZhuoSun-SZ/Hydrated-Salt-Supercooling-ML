import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, mean_squared_error

import seaborn as sns


FONT_SIZE_TITLE      = 20           
FONT_SIZE_AXIS_LABEL = 20           
FONT_SIZE_TICK       = 18           
FONT_WEIGHT          = "bold"       
FONT_FAMILY          = "Times New Roman"  

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         [FONT_FAMILY, "Liberation Serif",
                           "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",       
    "axes.unicode_minus": False,        
    "font.size":          FONT_SIZE_TICK,
    "font.weight":        FONT_WEIGHT,
    "axes.titlesize":     FONT_SIZE_TITLE,
    "axes.titleweight":   FONT_WEIGHT,
    "axes.labelsize":     FONT_SIZE_AXIS_LABEL,
    "axes.labelweight":   FONT_WEIGHT,
    "xtick.labelsize":    FONT_SIZE_TICK,
    "ytick.labelsize":    FONT_SIZE_TICK,
})


FIGSIZE = (12, 5)
FIG_DPI = 300
SAVE_DPI = 600

RANDOM_STATE_CV = 42

cv_plot = cv5

N_CV_FOLDS = cv_plot.get_n_splits(X_train, y_train)

print("\n" + "=" * 72)
print("Cross-validation stability evaluation of all candidate base learners")
print("=" * 72)
print(f"Number of candidate models: {len(tuned_models)}")
print(f"Number of CV folds: {N_CV_FOLDS}")
print(f"Evaluation data: training set, {len(X_train)} samples in total")
print("Note: This step does not use the independent test set.")


base_fold_scores = {}

fold_rows = []

for model_name, model in tuned_models.items():

    print(f"\nPerforming CV evaluation: {model_name}")

    scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv_plot,
        scoring={
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
        },
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    fold_r2 = np.asarray(scores["test_r2"], dtype=float)
    fold_rmse = -np.asarray(scores["test_rmse"], dtype=float)

    base_fold_scores[model_name] = {
        "r2": fold_r2,
        "rmse": fold_rmse,
    }

    for fold_idx, (r2_value, rmse_value) in enumerate(
        zip(fold_r2, fold_rmse), start=1
    ):
        fold_rows.append({
            "Model": model_name,
            "Fold": fold_idx,
            "R2": float(r2_value),
            "RMSE": float(rmse_value),
        })

    print(
        f"  R²:   {np.mean(fold_r2):.4f} "
        f"± {np.std(fold_r2, ddof=1):.4f}"
    )
    print(
        f"  RMSE: {np.mean(fold_rmse):.4f} "
        f"± {np.std(fold_rmse, ddof=1):.4f}"
    )


fold_df = pd.DataFrame(fold_rows)

fold_csv_path = os.path.join(
    OUTPUT_DIR,
    "all_base_learners_cv_fold_scores.csv"
)

fold_df.to_csv(
    fold_csv_path,
    index=False,
    encoding="utf-8-sig"
)


summary_rows = []

for model_name, values in base_fold_scores.items():

    r2_values = np.asarray(values["r2"], dtype=float)
    rmse_values = np.asarray(values["rmse"], dtype=float)

    summary_rows.append({
        "Model": model_name,

        "CV_folds": len(r2_values),

        "R2_mean": np.mean(r2_values),
        "R2_std": np.std(r2_values, ddof=1),

        "R2_median": np.median(r2_values),
        "R2_min": np.min(r2_values),
        "R2_max": np.max(r2_values),

        "RMSE_mean": np.mean(rmse_values),
        "RMSE_std": np.std(rmse_values, ddof=1),

        "RMSE_median": np.median(rmse_values),
        "RMSE_min": np.min(rmse_values),
        "RMSE_max": np.max(rmse_values),
    })

summary_df = pd.DataFrame(summary_rows)

summary_df = summary_df.sort_values(
    by="RMSE_mean",
    ascending=True
).reset_index(drop=True)

summary_csv_path = os.path.join(
    OUTPUT_DIR,
    "all_base_learners_cv_summary.csv"
)

summary_df.to_csv(
    summary_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print("\nCV summary results:")
print(
    summary_df[
        ["Model", "R2_mean", "R2_std", "RMSE_mean", "RMSE_std"]
    ].round(4).to_string(index=False)
)


model_names_cv = list(base_fold_scores.keys())

r2_data = [
    base_fold_scores[m]["r2"]
    for m in model_names_cv
]

rmse_data = [
    base_fold_scores[m]["rmse"]
    for m in model_names_cv
]


def format_model_label(model_name):

    special_names = {
        "rf": "RF",
        "xgb": "XGBoost",
        "lgbm": "LightGBM",
        "gbr": "GBR",
        "catboost": "CatBoost",
        "knn": "KNN",
        "dt": "DT",
        "svr": "SVR",
        "bpnn": "BPNN",
        "extra_trees": "ExtraTrees",
        "gpr": "GPR",
        "krr": "KRR",
        "adaboost": "AdaBoost",
        "hist_gbr": "HistGBR",
        "ebm": "EBM",
    }

    return special_names.get(
        model_name.lower(),
        model_name.upper()
    )


labels = [
    format_model_label(m)
    for m in model_names_cv
]

model_colors = sns.color_palette(
    "tab10",
    n_colors=len(model_names_cv)
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=FIGSIZE,
    dpi=FIG_DPI
)


box1 = axes[0].boxplot(
    r2_data,
    labels=labels,
    patch_artist=True,
    widths=0.55,
    showfliers=True,

    medianprops={
        "color": "black",
        "linewidth": 1.2,
    },

    whiskerprops={
        "color": "black",
        "linewidth": 1.0,
    },

    capprops={
        "color": "black",
        "linewidth": 1.0,
    },

    flierprops={
        "marker": "o",
        "markersize": 3,
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "alpha": 0.6,
    },
)

for patch, color in zip(
    box1["boxes"],
    model_colors
):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)


rng = np.random.default_rng(RANDOM_STATE_CV)

for i, data in enumerate(r2_data, start=1):

    x = rng.normal(
        loc=i,
        scale=0.025,
        size=len(data)
    )

    axes[0].plot(
        x,
        data,
        "o",
        color="gray",
        markersize=4,
        alpha=0.85,
        markeredgewidth=0,
        zorder=3,
    )


axes[0].set_title(
    f"{N_CV_FOLDS}-Fold CV Stability",
    fontsize=FONT_SIZE_TITLE,
    fontweight=FONT_WEIGHT
)

axes[0].set_ylabel(
    r"$R^2$",
    fontsize=FONT_SIZE_AXIS_LABEL,
    fontweight=FONT_WEIGHT
)


box2 = axes[1].boxplot(
    rmse_data,
    labels=labels,
    patch_artist=True,
    widths=0.55,
    showfliers=True,

    medianprops={
        "color": "black",
        "linewidth": 1.2,
    },

    whiskerprops={
        "color": "black",
        "linewidth": 1.0,
    },

    capprops={
        "color": "black",
        "linewidth": 1.0,
    },

    flierprops={
        "marker": "o",
        "markersize": 3,
        "markerfacecolor": "black",
        "markeredgecolor": "black",
        "alpha": 0.6,
    },
)

for patch, color in zip(
    box2["boxes"],
    model_colors
):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)


for i, data in enumerate(rmse_data, start=1):

    x = rng.normal(
        loc=i,
        scale=0.025,
        size=len(data)
    )

    axes[1].plot(
        x,
        data,
        "o",
        color="gray",
        markersize=4,
        alpha=0.85,
        markeredgewidth=0,
        zorder=3,
    )


axes[1].set_title(
    f"{N_CV_FOLDS}-Fold CV Stability",
    fontsize=FONT_SIZE_TITLE,
    fontweight=FONT_WEIGHT
)

axes[1].set_ylabel(
    "RMSE (°C)",
    fontsize=FONT_SIZE_AXIS_LABEL,
    fontweight=FONT_WEIGHT
)


for ax in axes:

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="x",
        rotation=45,
        labelsize=FONT_SIZE_TICK
    )

    ax.tick_params(
        axis="y",
        labelsize=FONT_SIZE_TICK
    )

    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("right")

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35
    )

plt.subplots_adjust(
    wspace=0.35,
    left=0.09,
    right=0.98,
    bottom=0.25,
    top=0.90
)


fig.canvas.draw()   

for ax in axes:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontsize(FONT_SIZE_TICK)
        lbl.set_fontweight(FONT_WEIGHT)


figure_path = os.path.join(
    OUTPUT_DIR,
    "all_base_learners_cv_stability.png"
)

plt.savefig(
    figure_path,
    dpi=SAVE_DPI,
    bbox_inches="tight"
)

plt.show()
plt.close()


print("\n" + "=" * 72)
print("CV evaluation of all candidate base learners completed.")
print(f"Detailed results per fold: {fold_csv_path}")
print(f"Summary statistics results: {summary_csv_path}")
print(f"CV stability boxplot: {figure_path}")
print("=" * 72)