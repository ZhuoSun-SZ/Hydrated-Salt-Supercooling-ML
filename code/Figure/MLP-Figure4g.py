import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
import os

print("\n" + "=" * 72)
print("=" * 72)

bg_color_start = "#fbdddd"       
bg_color_end   = "#ceeff6"       
gradient_direction = 'vertical'  

TITLE_FONTSIZE      = 26         
LABEL_FONTSIZE      = 26         
TICK_FONTSIZE       = 22         
LEGEND_FONTSIZE     = 22         
ANNOTATION_FONTSIZE = 20         

font_family = 'Times New Roman'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [font_family, 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False


name_map = {
    'rf': 'RF', 'xgb': 'XGBoost', 'lgbm': 'LightGBM', 'gbr': 'GBR',
    'catboost': 'CatBoost', 'knn': 'KNN', 'dt': 'DT', 'svr': 'SVR',
    'bpnn': 'BPNN', 'extra_trees': 'ExtraTrees', 'gpr': 'GPR',
    'krr': 'KRR', 'adaboost': 'AdaBoost', 'hist_gbr': 'HistGBR', 'ebm': 'EBM'
}

base_r2 = []
base_rmse = []
display_names = []

for name in selected_names:
    if name not in tuned_models:
        print(f"[警告] tuned_models 中不存在 {name}，跳过。")
        continue

    model = tuned_models[name]  
    
    oof_pred = cross_val_predict(clone(model), X_train, y_train, cv=cv5, n_jobs=-1)
    r2 = r2_score(y_train, oof_pred)
    rmse = np.sqrt(mean_squared_error(y_train, oof_pred))
    
    base_r2.append(r2)
    base_rmse.append(rmse)
    display_names.append(name_map.get(name, name.upper()))

hidden_sizes = best_meta_params['hidden_layer_sizes']
if isinstance(hidden_sizes, list):
    hidden_sizes = tuple(hidden_sizes)

meta_model_for_plot = MLPRegressor(
    hidden_layer_sizes=hidden_sizes,
    activation=best_meta_params['activation'],
    solver=best_meta_params['solver'],
    alpha=best_meta_params['alpha'],
    learning_rate_init=best_meta_params['learning_rate_init'],
    batch_size=best_meta_params['batch_size'],
    max_iter=best_meta_params['max_iter'],
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=RANDOM_STATE
)

stack_oof_pred = cross_val_predict(meta_model_for_plot, meta_oof, y_train, cv=cv5, n_jobs=-1)
stack_r2 = r2_score(y_train, stack_oof_pred)
stack_rmse = np.sqrt(mean_squared_error(y_train, stack_oof_pred))

all_rmse = base_rmse + [stack_rmse]
all_r2 = base_r2 + [stack_r2]

x_min, x_max = min(all_rmse), max(all_rmse)
y_min, y_max = min(all_r2), max(all_r2)

x_margin = (x_max - x_min) * 0.15 if x_max > x_min else 0.5
y_margin = (y_max - y_min) * 0.15 if y_max > y_min else 0.05

x_limits = (x_min - x_margin, x_max + x_margin * 2.0)
y_limits = (y_min - y_margin, y_max + y_margin * 1.2)

fig, ax = plt.subplots(figsize=(9, 6.5))
fig.patch.set_facecolor('white')

ax.set_xlim(x_limits)
ax.set_ylim(y_limits)

if gradient_direction == 'vertical':
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    gradient = np.vstack([gradient, gradient])
else:
    gradient = np.linspace(0, 1, 100).reshape(-1, 1)
    gradient = np.hstack([gradient, gradient])

cmap = LinearSegmentedColormap.from_list('gradient', [bg_color_start, bg_color_end])
ax.imshow(gradient, aspect='auto', extent=[x_limits[0], x_limits[1], y_limits[0], y_limits[1]],
          origin='lower', cmap=cmap, zorder=-1)

ax.scatter(base_rmse, base_r2, color="#90c7f5", s=100, alpha=0.8, edgecolors='white', linewidth=2,
           label='Base Learners ', zorder=2)
ax.scatter(stack_rmse, stack_r2, color='#d62728', s=300, marker='*', edgecolors='black', 
           linewidth=1.5, label='Stacking Model ', zorder=3)

sorted_indices = np.argsort(base_rmse)
texts = []
for i, idx in enumerate(sorted_indices):
    disp_name = display_names[idx]
    x = base_rmse[idx]
    y = base_r2[idx]
    
    t = ax.text(x, y, disp_name, 
                fontsize=ANNOTATION_FONTSIZE, fontweight='bold', 
                color="#131a1f", ha='center', va='center', zorder=4)
    texts.append(t)

adjust_text(
    texts, 
    ax=ax,
    only_move={'points': 'y', 'text': 'y'},
    expand_text=(1.2, 1.2),
    arrowprops=dict(arrowstyle="-", color='gray', lw=0.8, alpha=0.7)
)

ax.annotate('Stacking', (stack_rmse, stack_r2), 
            xytext=(15, 0), textcoords='offset points',
            fontsize=ANNOTATION_FONTSIZE+2, fontweight='bold', ha='left', va='center', color='#d62728',
            zorder=4)

ax.set_xlabel('CV RMSE ', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax.set_ylabel('CV R² ', fontsize=LABEL_FONTSIZE, fontweight='bold')
ax.set_title('Stacking vs Selected Base Learners', 
             fontsize=TITLE_FONTSIZE, fontweight='bold', pad=15)

ax.legend(loc='upper right', prop={'size': LEGEND_FONTSIZE, 'weight': 'bold'}, frameon=False)

ax.tick_params(labelsize=TICK_FONTSIZE, width=1.5)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight('bold')

plt.tight_layout()

save_path = os.path.join(OUTPUT_DIR, 'cv_performance_comparison_selected.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

