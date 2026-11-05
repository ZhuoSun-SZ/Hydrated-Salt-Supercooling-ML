import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("\nPlotting Optuna TPE optimization process visualization (independent output)...")

FONTSIZE_TITLE  = 26
FONTSIZE_LABEL  = 24
FONTSIZE_TICK   = 22
FONTSIZE_LEGEND = 20
FONTSIZE_CBAR   = 20
FONT_WEIGHT     = "bold"
FONT_FAMILY     = "Times New Roman"

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         [FONT_FAMILY, "Liberation Serif", "DejaVu Serif"],
    "axes.unicode_minus": False,
    "font.weight":        FONT_WEIGHT,
    "axes.titlesize":     FONTSIZE_TITLE,
    "axes.titleweight":   FONT_WEIGHT,
    "axes.labelsize":     FONTSIZE_LABEL,
    "axes.labelweight":   FONT_WEIGHT,
    "xtick.labelsize":    FONTSIZE_TICK,
    "ytick.labelsize":    FONTSIZE_TICK,
})

df_optuna = study_meta.trials_dataframe()
alphas = df_optuna['params_alpha'].values
values = df_optuna['value'].values

best_idx = df_optuna['value'].idxmin()
best_alpha_opt = df_optuna.loc[best_idx, 'params_alpha']
best_value_opt = df_optuna.loc[best_idx, 'value']

trial_numbers = df_optuna.index.values
best_so_far = np.minimum.accumulate(values)

fig1, ax1 = plt.subplots(figsize=(8, 6))

scatter = ax1.scatter(alphas, values, c=df_optuna.index, cmap='viridis', alpha=0.6, s=60, edgecolors='white')
ax1.scatter(best_alpha_opt, best_value_opt, color='red', marker='*', s=300, edgecolors='black', zorder=5, label='Best Trial')

ax1.set_xscale('log')
ax1.set_xlabel("Regularization Parameter (Alpha)")
ax1.set_ylabel("CV RMSE (°C)")
ax1.set_title("TPE Search Space Exploration")
ax1.grid(alpha=0.3, linestyle="--")
ax1.legend(prop={'size': FONTSIZE_LEGEND, 'weight': FONT_WEIGHT})

cbar = fig1.colorbar(scatter, ax=ax1, pad=0.02)
cbar.set_label('Trial Order', fontsize=FONTSIZE_CBAR, fontweight=FONT_WEIGHT)
cbar.ax.tick_params(labelsize=FONTSIZE_CBAR)
for lbl in cbar.ax.get_yticklabels():
    lbl.set_fontweight(FONT_WEIGHT)

for lbl in ax1.get_xticklabels() + ax1.get_yticklabels():
    lbl.set_fontweight(FONT_WEIGHT)

plt.tight_layout()
save_path_1 = os.path.join(OUTPUT_DIR, "optuna_tpe_exploration.png")
plt.savefig(save_path_1, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.plot(trial_numbers, values, 'o', color='gray', alpha=0.4, label='Each Trial')
ax2.plot(trial_numbers, best_so_far, '-', color='#d62728', linewidth=2.5, label='Best RMSE so far')

ax2.set_xlabel("Trial Number")
ax2.set_ylabel("CV RMSE (°C)")
ax2.set_title("TPE Convergence History")
ax2.grid(alpha=0.3, linestyle="--")
ax2.legend(prop={'size': FONTSIZE_LEGEND, 'weight': FONT_WEIGHT}, loc='upper right')

for lbl in ax2.get_xticklabels() + ax2.get_yticklabels():
    lbl.set_fontweight(FONT_WEIGHT)

plt.tight_layout()
save_path_2 = os.path.join(OUTPUT_DIR, "optuna_tpe_convergence.png")
plt.savefig(save_path_2, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig2)

print(f"Figure 1 (Exploration Scatter Plot) saved to: {save_path_1}")
print(f"Figure 2 (Convergence History Plot) saved to: {save_path_2}")