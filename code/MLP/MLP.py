import os
import json
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import optuna
from optuna.samplers import TPESampler

from scipy.stats import randint, uniform, loguniform

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, KFold, cross_val_score,
    cross_val_predict
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import (
    StackingRegressor, RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, Matern, RationalQuadratic
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_SPLITS = 10

DATA_PATH = "/Users/sunzhuo/Sun/0.1.xlsx"
OUTPUT_DIR = "/Users/sunzhuo/Sun/MLP/NSGA2_stacking_results"

TEST_SIZE = 0.20
N_ITER_BASE_SEARCH = 100
POPULATION_SIZE = 16
N_GENERATIONS = 12
N_TRIALS_META = 300
CROSSOVER_PROB = 0.90
MUTATION_PROB = None

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

base_models = {
    "rf": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
    "gbr": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "knn": KNeighborsRegressor(),
    "dt": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "svr": SVR(kernel="rbf"),
    "bpnn": MLPRegressor(random_state=RANDOM_STATE, max_iter=2000,
                         early_stopping=True, n_iter_no_change=20),
}

param_distributions = {
    "rf": {
        "model__n_estimators": randint(200, 1001),
        "model__max_depth": [None, 4, 6, 8, 10, 15, 20],
        "model__min_samples_split": randint(2, 16),
        "model__min_samples_leaf": randint(1, 9),
        "model__max_features": [0.5, 0.7, 0.9, 1.0],
    },
    "xgb": {
        "model__n_estimators": randint(100, 1001),
        "model__max_depth": randint(2, 8),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__min_child_weight": loguniform(0.1, 20),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_alpha": loguniform(1e-5, 10),
        "model__reg_lambda": loguniform(1e-3, 100),
    },
    "lgbm": {
        "model__n_estimators": randint(150, 1001),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": [-1, 3, 4, 5, 6, 8, 10],
        "model__num_leaves": randint(5, 33),
        "model__min_child_samples": randint(5, 31),
        "model__reg_alpha": loguniform(1e-5, 10),
        "model__reg_lambda": loguniform(1e-5, 100),
    },
    "gbr": {
        "model__n_estimators": randint(100, 1001),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__max_depth": randint(2, 6),
        "model__min_samples_leaf": randint(1, 16),
        "model__subsample": uniform(0.6, 0.4),
        "model__max_features": [None, "sqrt", 0.7, 1.0],
    },
    "catboost": {
        "model__iterations": randint(300, 1801),
        "model__depth": randint(4, 11),
        "model__learning_rate": loguniform(0.005, 0.15),
        "model__l2_leaf_reg": loguniform(1, 30),
    },
    "knn": {
        "model__n_neighbors": randint(2, 20),
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    "dt": {
        "model__max_depth": randint(3, 25),
    },
    "svr": {
        "model__C": uniform(0.5, 100),
        "model__gamma": uniform(0.001, 5),
        "model__kernel": ["rbf", "linear", ],
        "model__epsilon": uniform(0.01, 0.6),
        "model__degree": randint(2, 5),
    },
    "bpnn": {
        "model__hidden_layer_sizes": [
            (8,), (16,), (32,), (64,),
            (16, 8), (32, 16), (64, 32)
        ],
        "model__activation": ["relu", "tanh"],
        "model__alpha": loguniform(1e-6, 1e-1),
        "model__learning_rate_init": loguniform(1e-4, 3e-2),
        "model__batch_size": [8, 16, 32, 64],
        "model__max_iter": [2000, 3000],
        "model__early_stopping": [True],
        "model__validation_fraction": [0.1, 0.15],
        "model__n_iter_no_change": [20, 30],
    },
    "extra_trees": {
        "model__n_estimators": randint(300, 1201),
        "model__max_depth": [None, 5, 8, 12, 16, 20],
        "model__min_samples_split": randint(2, 16),
        "model__min_samples_leaf": randint(1, 9),
        "model__max_features": [0.5, 0.7, 0.9, 1.0],
    },
    "gpr": {
        "model__kernel": [
            ConstantKernel(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2)),
            ConstantKernel(1.0, (1e-2, 1e2)) * Matern(1.0, (1e-2, 1e2), nu=1.5),
            ConstantKernel(1.0, (1e-2, 1e2)) * Matern(1.0, (1e-2, 1e2), nu=2.5),
            ConstantKernel(1.0, (1e-2, 1e2)) * RationalQuadratic(length_scale=1.0),
        ],
        "model__alpha": loguniform(1e-8, 1),
        "model__normalize_y": [True, False],
    },
    "krr": {
        "model__alpha": loguniform(1e-5, 1e1),
        "model__kernel": ["rbf", "poly", "laplacian"],
        "model__gamma": loguniform(1e-4, 1e1),
        "model__degree": randint(2, 5),
        "model__coef0": uniform(0.0, 2.0),
    },
    "adaboost": {
        "model__n_estimators": randint(30, 400),
        "model__learning_rate": loguniform(1e-2, 1.0),
        "model__loss": ["linear", "square", "exponential"],
    },
    "hist_gbr": {
        "model__max_iter": randint(50, 400),
        "model__learning_rate": loguniform(1e-2, 2e-1),
        "model__max_leaf_nodes": randint(7, 32),
        "model__max_depth": [None, 2, 3, 4, 5, 6],
        "model__min_samples_leaf": randint(5, 25),
        "model__l2_regularization": loguniform(1e-5, 10.0),
    },
    "ebm": {
        "model__max_bins": [128, 256, 512],
        "model__max_leaves": [2, 3, 4, 5],
        "model__learning_rate": loguniform(0.01, 0.20),
        "model__min_samples_leaf": randint(3, 16),
        "model__max_rounds": randint(2000, 8001),
    },
}

try:
    from xgboost import XGBRegressor
    base_models["xgb"] = XGBRegressor(
        random_state=RANDOM_STATE, n_jobs=1, objective="reg:squarederror", verbosity=0
    )
except ImportError:
    print("[Info] xgboost not installed, skipping xgb.")

try:
    from lightgbm import LGBMRegressor
    base_models["lgbm"] = LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, n_jobs=1)
except ImportError:
    print("[Info] lightgbm not installed, skipping lgbm.")

try:
    from catboost import CatBoostRegressor
    base_models["catboost"] = CatBoostRegressor(
        random_seed=RANDOM_STATE, verbose=0, allow_writing_files=False, thread_count=1
    )
except ImportError:
    print("[Info] catboost not installed, skipping catboost.")

base_models["extra_trees"] = ExtraTreesRegressor(
    random_state=RANDOM_STATE, n_jobs=1
)
base_models["gpr"] = GaussianProcessRegressor(
    random_state=RANDOM_STATE, n_restarts_optimizer=0
)
base_models["krr"] = KernelRidge()
base_models["adaboost"] = AdaBoostRegressor(random_state=RANDOM_STATE)
base_models["hist_gbr"] = HistGradientBoostingRegressor(
    random_state=RANDOM_STATE, early_stopping=False
)

try:
    from interpret.glassbox import ExplainableBoostingRegressor
    base_models["ebm"] = ExplainableBoostingRegressor(
        random_state=RANDOM_STATE, n_jobs=1, interactions=5,
        outer_bags=8, validation_size=0.15
    )
except Exception as exc:
    print(f"[Info] EBM not available, skipping ebm. Install/update interpret if needed. Reason: {exc}")

_original_model_order = [
    "rf", "xgb", "lgbm", "gbr", "catboost", "knn", "dt", "svr", "bpnn",
    "extra_trees", "gpr", "krr", "adaboost", "hist_gbr", "ebm"
]
base_models = {name: base_models[name] for name in _original_model_order if name in base_models}
param_distributions = {name: param_distributions[name] for name in _original_model_order if name in param_distributions}

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Data file not found: {DATA_PATH}\nPlease modify DATA_PATH at the top of the script."
    )

df = pd.read_excel(DATA_PATH)
if df.shape[1] < 3:
    raise ValueError("Data must contain at least: one ID column, one or more feature columns, and the last column as the target.")

X_raw = df.iloc[:, 1:-1].copy()
y_raw = pd.to_numeric(df.iloc[:, -1], errors="coerce")

valid = y_raw.notna()
X_raw = X_raw.loc[valid].reset_index(drop=True)
y = y_raw.loc[valid].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

categorical_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

def make_preprocessor():
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def make_model_pipeline(model):
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", clone(model)),
    ])

cv5 = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

print("=" * 72)
print("Data and cross-validation settings")
print("=" * 72)
print(f"Total samples: {len(y)}")
print(f"Training set: {len(y_train)} | Independent test set: {len(y_test)}")
print(f"Numerical features: {len(numeric_cols)} | Categorical features: {len(categorical_cols)}")
print(f"Candidate base learners: {list(base_models.keys())}")
print(f"Unified cross-validation: {N_SPLITS}-fold")
print("The independent test set is not used for hyperparameter tuning or NSGA-II selection.")

print("\n" + "=" * 72)
print("Step 1: Hyperparameter tuning using 10-fold CV within the training set")
print("=" * 72)

tuned_models = {}
model_scores = {}
search_summaries = []

for name, model in base_models.items():
    print(f"\nTuning: {name}")
    pipe = make_model_pipeline(model)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions[name],
        n_iter=N_ITER_BASE_SEARCH,
        scoring="neg_root_mean_squared_error",
        cv=cv5,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    tuned_models[name] = search.best_estimator_
    model_scores[name] = float(-search.best_score_)
    search_summaries.append({
        "model": name,
        "cv_rmse": float(-search.best_score_),
        "best_params": str(search.best_params_),
    })
    print(f"  CV RMSE = {-search.best_score_:.4f}")
    print(f"  Best parameters = {search.best_params_}")

sorted_scores = sorted(model_scores.items(), key=lambda item: item[1])

model_names = list(tuned_models.keys())
n_models = len(model_names)
if n_models < 2:
    raise RuntimeError("Fewer than 2 candidate models available. Cannot perform Stacking.")

if MUTATION_PROB is None:
    MUTATION_PROB = 1.0 / n_models

def make_meta_model(params=None):
    defaults = {
        "hidden_layer_sizes": (16, 8),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.01,
        "learning_rate_init": 0.001,
        "batch_size": 16,
        "max_iter": 2000,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "random_state": RANDOM_STATE,
    }
    if params:
        defaults.update(params)
    if isinstance(defaults.get("hidden_layer_sizes"), list):
        defaults["hidden_layer_sizes"] = tuple(defaults["hidden_layer_sizes"])
    return MLPRegressor(**defaults)

fitness_cache = {}
rmse_se_cache = {}

def repair_individual(ind):
    ind = np.asarray(ind, dtype=int).copy()
    if ind.sum() < 2:
        zeros = np.where(ind == 0)[0]
        need = 2 - int(ind.sum())
        if len(zeros) > 0:
            chosen = np.random.choice(zeros, size=min(need, len(zeros)), replace=False)
            ind[chosen] = 1
    return ind

print("\nComputing 10-fold OOF residual correlations for each base learner to assess model complementarity...")
model_names = list(tuned_models.keys())
n_models = len(model_names)
if n_models < 2:
    raise RuntimeError("Fewer than 2 candidate models available. Cannot perform Stacking.")

oof_residuals = {}
for name in model_names:
    oof_pred = cross_val_predict(
        clone(tuned_models[name]), X_train, y_train, cv=cv5, n_jobs=1
    )
    oof_residuals[name] = np.asarray(y_train, dtype=float) - np.asarray(oof_pred, dtype=float)

residual_matrix = np.column_stack([oof_residuals[name] for name in model_names])
if n_models >= 2:
    residual_corr_matrix = np.corrcoef(residual_matrix, rowvar=False)
    residual_corr_matrix = np.atleast_2d(residual_corr_matrix)
    residual_corr_matrix = np.nan_to_num(residual_corr_matrix, nan=1.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(residual_corr_matrix, 1.0)
else:
    residual_corr_matrix = np.ones((1, 1), dtype=float)

def mean_pairwise_residual_correlation(selected_indices):
    if len(selected_indices) < 2:
        return 1.0
    vals = []
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            vals.append(residual_corr_matrix[selected_indices[i], selected_indices[j]])
    return float(np.mean(vals)) if vals else 1.0

def evaluate_individual(ind):
    ind = repair_individual(ind)
    key = tuple(ind.tolist())
    if key in fitness_cache:
        return fitness_cache[key]

    selected_indices = [i for i, bit in enumerate(ind) if bit == 1]
    selected = [model_names[i] for i in selected_indices]
    estimators = [(name, clone(tuned_models[name])) for name in selected]

    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=make_meta_model(),
        cv=KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        passthrough=False,
        n_jobs=1,
    )
    try:
        scores = cross_val_score(
            stack, X_train, y_train,
            cv=cv5,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
            error_score="raise",
        )
        fold_rmse = -np.asarray(scores, dtype=float)
        rmse = float(np.mean(fold_rmse))
        rmse_se = (
            float(np.std(fold_rmse, ddof=1) / np.sqrt(len(fold_rmse)))
            if len(fold_rmse) > 1 else 0.0
        )
        if not np.isfinite(rmse):
            rmse = 1e12
        if not np.isfinite(rmse_se):
            rmse_se = 0.0

    except Exception as exc:
        print(f"[Warning] Evaluation failed for combination {selected}: {exc}")
        rmse = 1e12
        rmse_se = 0.0
    residual_corr = mean_pairwise_residual_correlation(selected_indices)
    result = (rmse, residual_corr)
    fitness_cache[key] = result
    rmse_se_cache[key] = rmse_se

    print(
        f"  Combination={selected} | CV RMSE={rmse:.4f} | "
        f"Mean residual correlation={residual_corr:.4f} | Number of models={len(selected)}"
    )
    return result

def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def fast_non_dominated_sort(population, fitnesses):
    n = len(population)
    domination_count = [0] * n
    dominated_sets = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(fitnesses[p], fitnesses[q]):
                dominated_sets[p].append(q)
            elif dominates(fitnesses[q], fitnesses[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    return fronts

def crowding_distance(front, fitnesses):
    distance = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distance[idx] = float("inf")
        return distance

    n_obj = len(fitnesses[0])
    for obj in range(n_obj):
        ordered = sorted(front, key=lambda idx: fitnesses[idx][obj])
        distance[ordered[0]] = float("inf")
        distance[ordered[-1]] = float("inf")
        low = fitnesses[ordered[0]][obj]
        high = fitnesses[ordered[-1]][obj]
        if high == low:
            continue
        for j in range(1, len(ordered) - 1):
            if np.isinf(distance[ordered[j]]):
                continue
            prev_val = fitnesses[ordered[j - 1]][obj]
            next_val = fitnesses[ordered[j + 1]][obj]
            distance[ordered[j]] += (next_val - prev_val) / (high - low)
    return distance

def tournament_select(population, fitnesses, ranks, distances):
    i, j = np.random.randint(0, len(population), size=2)
    if ranks[i] < ranks[j]:
        return population[i].copy()
    if ranks[j] < ranks[i]:
        return population[j].copy()
    if distances[i] > distances[j]:
        return population[i].copy()
    if distances[j] > distances[i]:
        return population[j].copy()
    return population[np.random.choice([i, j])].copy()

def crossover(parent1, parent2):
    if np.random.rand() > CROSSOVER_PROB:
        return parent1.copy(), parent2.copy()
    mask = np.random.rand(n_models) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return repair_individual(child1), repair_individual(child2)

def mutate(ind):
    child = ind.copy()
    flips = np.random.rand(n_models) < MUTATION_PROB
    child[flips] = 1 - child[flips]
    return repair_individual(child)

def nsga2_select(population, fitnesses, n_keep):
    fronts = fast_non_dominated_sort(population, fitnesses)
    selected_pop, selected_fit = [], []

    for front in fronts:
        if len(selected_pop) + len(front) <= n_keep:
            selected_pop.extend([population[i].copy() for i in front])
            selected_fit.extend([fitnesses[i] for i in front])
        else:
            distances = crowding_distance(front, fitnesses)
            front_sorted = sorted(front, key=lambda idx: distances[idx], reverse=True)
            remaining = n_keep - len(selected_pop)
            chosen = front_sorted[:remaining]
            selected_pop.extend([population[i].copy() for i in chosen])
            selected_fit.extend([fitnesses[i] for i in chosen])
            break
    return selected_pop, selected_fit

print("\n" + "=" * 72)
print("Step 2: NSGA-II multi-objective base learner selection")
print("=" * 72)
np.random.seed(RANDOM_STATE)

population = []
for _ in range(POPULATION_SIZE):
    ind = (np.random.rand(n_models) < np.random.uniform(0.25, 0.75)).astype(int)
    population.append(repair_individual(ind))

full_ind = np.ones(n_models, dtype=int)
population[0] = full_ind.copy()

ranked_names = [name for name, _ in sorted_scores]
ranked_indices = [model_names.index(name) for name in ranked_names]
seed_ind = np.zeros(n_models, dtype=int)
seed_ind[ranked_indices[:min(3, n_models)]] = 1
if POPULATION_SIZE > 1:
    population[1] = repair_individual(seed_ind)

fitnesses = [evaluate_individual(ind) for ind in population]
history = []

for generation in range(N_GENERATIONS):
    fronts = fast_non_dominated_sort(population, fitnesses)
    rank_map = {}
    distance_map = {}
    for rank, front in enumerate(fronts):
        d = crowding_distance(front, fitnesses)
        for idx in front:
            rank_map[idx] = rank
            distance_map[idx] = d[idx]

    offspring = []
    while len(offspring) < POPULATION_SIZE:
        p1 = tournament_select(population, fitnesses, rank_map, distance_map)
        p2 = tournament_select(population, fitnesses, rank_map, distance_map)
        c1, c2 = crossover(p1, p2)
        offspring.append(mutate(c1))
        if len(offspring) < POPULATION_SIZE:
            offspring.append(mutate(c2))

    offspring_fit = [evaluate_individual(ind) for ind in offspring]
    combined_pop = population + offspring
    combined_fit = fitnesses + offspring_fit
    population, fitnesses = nsga2_select(combined_pop, combined_fit, POPULATION_SIZE)

    best_rmse = min(f[0] for f in fitnesses)
    best_residual_corr = min(f[1] for f in fitnesses)
    history.append({
        "generation": generation + 1,
        "best_cv_rmse": best_rmse,
        "best_mean_residual_correlation": best_residual_corr,
        "unique_combinations_evaluated": len(fitness_cache),
    })
    print(f"Generation {generation + 1}/{N_GENERATIONS}: "
          f"best RMSE={best_rmse:.4f}, best residual correlation={best_residual_corr:.4f}, "
          f"unique combinations={len(fitness_cache)}")

all_items = []
for key, (rmse, residual_corr) in fitness_cache.items():
    all_items.append((np.array(key, dtype=int), (rmse, residual_corr)))

all_pop = [x[0] for x in all_items]
all_fit = [x[1] for x in all_items]
fronts_all = fast_non_dominated_sort(all_pop, all_fit)
pareto_idx = fronts_all[0]
pareto_items = [all_items[i] for i in pareto_idx]
pareto_items.sort(key=lambda item: (item[1][0], item[1][1]))

if not pareto_items:
    raise RuntimeError("NSGA-II did not produce a valid Pareto solution.")

best_pareto_ind, best_pareto_fit = min(
    pareto_items,
    key=lambda item: item[1][0]
)
best_pareto_key = tuple(best_pareto_ind.tolist())
best_pareto_se = float(rmse_se_cache.get(best_pareto_key, 0.0))
reverse_1se_threshold = float(best_pareto_fit[0] + best_pareto_se)

eligible_items = [
    (ind, fit) for ind, fit in pareto_items
    if fit[0] <= reverse_1se_threshold
]
if not eligible_items:
    eligible_items = [(best_pareto_ind, best_pareto_fit)]

chosen_ind, chosen_fit = max(
    eligible_items,
    key=lambda item: (
        int(item[0].sum()),
        -float(item[1][0]),
        -float(item[1][1])
    )
)
selected_names = [model_names[i] for i, bit in enumerate(chosen_ind) if bit == 1]
chosen_key = tuple(chosen_ind.tolist())
chosen_se = float(rmse_se_cache.get(chosen_key, 0.0))

print(f"Final selection (most models within threshold): K={len(selected_names)}")
print(f"Final base learners: {selected_names}")
print(f"Selected combination CV RMSE: {chosen_fit[0]:.4f}")
print(f"Selected combination SE: {chosen_se:.4f}")
print(f"Mean OOF residual correlation of selected combination: {chosen_fit[1]:.4f}")
print(f"Number of base learners: {len(selected_names)}")