import os
import matplotlib.pyplot as plt

XLABEL_TEXT = "Mean Pairwise OOF Residual Correlation (lower is better)"
YLABEL_TEXT = "10-fold CV RMSE (lower is better)"
TITLE_TEXT  = "NSGA-II: Prediction Error vs Residual Complementarity"

FONTSIZE_TITLE  = 22           
FONTSIZE_LABEL  = 19           
FONTSIZE_TICK   = 17           
FONTSIZE_LEGEND = 16           
FONT_WEIGHT     = "bold"       
FONT_FAMILY     = "Times New Roman" 

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         [FONT_FAMILY, "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "axes.unicode_minus": False,
    "font.weight":        FONT_WEIGHT,
    "axes.titlesize":     FONTSIZE_TITLE,
    "axes.titleweight":   FONT_WEIGHT,
    "axes.labelsize":     FONTSIZE_LABEL,
    "axes.labelweight":   FONT_WEIGHT,
    "xtick.labelsize":    FONTSIZE_TICK,
    "ytick.labelsize":    FONTSIZE_TICK,
    "legend.fontsize":    FONTSIZE_LEGEND,
})

STAR_COLOR    = "#d62728"      
DIAMOND_COLOR = "#4C7F52"      

SIZE_EVALUATED = 105            
SIZE_PARETO    = 185            
SIZE_SELECTED  = 550           

fig, ax = plt.subplots(figsize=(8, 6))

all_rmse = [fit[0] for fit in all_fit]
all_corr = [fit[1] for fit in all_fit]

pareto_corr = [fit[1] for _, fit in pareto_items]
pareto_rmse = [fit[0] for _, fit in pareto_items]

ax.scatter(all_corr, all_rmse, s=SIZE_EVALUATED, alpha=0.45, label="Evaluated combinations", zorder=2)
ax.scatter(pareto_corr, pareto_rmse, s=SIZE_PARETO, marker="D", color=DIAMOND_COLOR, 
           label="Pareto front", zorder=3)
ax.scatter([chosen_fit[1]], [chosen_fit[0]], s=SIZE_SELECTED, marker="*", color=STAR_COLOR, 
           label="Selected", zorder=4)

ax.set_xlabel(XLABEL_TEXT, fontsize=FONTSIZE_LABEL)
ax.set_ylabel(YLABEL_TEXT, fontsize=FONTSIZE_LABEL)
ax.set_title(TITLE_TEXT, fontsize=FONTSIZE_TITLE, pad=12)

ax.grid(alpha=0.25)
ax.legend(prop={'size': FONTSIZE_LEGEND, 'weight': FONT_WEIGHT})

for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight(FONT_WEIGHT)

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, "nsga2_pareto_front.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

