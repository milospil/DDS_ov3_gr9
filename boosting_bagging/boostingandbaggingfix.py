# file: run_boosting_bagging.py
"""
Part 7: Boosting and Bagging Implementation

BOOSTING: AdaBoost, Gradient Boosting
BAGGING: Decision Trees (with early stopping), Logistic Regression
"""

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("PART 7: BOOSTING AND BAGGING MODELS")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading preprocessed data...")

X_train = pd.read_csv("competition/x_train.csv", header=0, index_col="id").astype(
    np.float32
)
y_train = pd.read_csv("competition/y_train.csv", header=0, index_col="id").squeeze(
    "columns"
)
X_test = pd.read_csv("competition/x_test.csv", header=0, index_col="id").astype(
    np.float32
)
y_test = pd.read_csv("competition/y_test.csv", header=0, index_col="id").squeeze(
    "columns"
)

# Convert target to integers
y_train = pd.to_numeric(y_train, errors="raise").astype(int)
y_test = pd.to_numeric(y_test, errors="raise").astype(int)

# Verify alignment
assert (X_train.index == y_train.index).all(), "Train indices don't match"
assert (X_test.index == y_test.index).all(), "Test indices don't match"

print(f"✓ Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"✓ Target distribution:\n{y_train.value_counts().sort_index()}")

# ============================================================================
# STEP 2: DEFINE EVALUATION FUNCTION
# ============================================================================


def evaluate_model(name, model, X_train, y_train, X_test, y_test, description=""):
    """
    Train and evaluate a model, printing detailed metrics.
    """
    print("\n" + "=" * 80)
    print(f"MODEL: {name}")
    if description:
        print(f"Description: {description}")
    print("=" * 80)

    # Train
    print("Training...")
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"\n✓ Test Accuracy:     {acc:.4f}")
    print(f"✓ Macro F1-score:    {f1_macro:.4f}")
    print(f"✓ Weighted F1-score: {f1_weighted:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    return {
        "Model": name,
        "Accuracy": acc,
        "Macro_F1": f1_macro,
        "Weighted_F1": f1_weighted,
    }


# ============================================================================
# BOOSTING MODELS
# ============================================================================
print("\n" + "#" * 80)
print("# BOOSTING METHODS")
print("#" * 80)

results = []

# ----------------------------------------------------------------------------
# MODEL 1: ADABOOST (Adaptive Boosting)
# ----------------------------------------------------------------------------
print("\n" + "-" * 80)
print("BOOSTING MODEL 1: AdaBoost")
print("-" * 80)

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42,
)

result = evaluate_model(
    "AdaBoost",
    adaboost,
    X_train,
    y_train,
    X_test,
    y_test,
    "Adaptive Boosting with Decision Stumps",
)
results.append(result)

# ----------------------------------------------------------------------------
# MODEL 2: GRADIENT BOOSTING
# ----------------------------------------------------------------------------
print("\n" + "-" * 80)
print("BOOSTING MODEL 2: Gradient Boosting")
print("-" * 80)

gradient_boost = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=42,
    verbose=0,
)

result = evaluate_model(
    "Gradient Boosting",
    gradient_boost,
    X_train,
    y_train,
    X_test,
    y_test,
    "Gradient Boosting with shallow trees",
)
results.append(result)

# ============================================================================
# BAGGING MODELS
# ============================================================================
print("\n" + "#" * 80)
print("# BAGGING METHODS")
print("#" * 80)

# ----------------------------------------------------------------------------
# MODEL 3: BAGGING WITH DECISION TREES
# ----------------------------------------------------------------------------
print("\n" + "-" * 80)
print("BAGGING MODEL 1: Bagging with Decision Trees")
print("-" * 80)


def fit_bagging_with_early_stop(
    X_train,
    y_train,
    base_depth=12,
    start_trees=100,
    step=100,
    max_trees=600,
    patience=3,
    min_improve=1e-4,
):
    """
    Fit bagging with early stopping based on OOB score.
    """
    print("\nTuning number of estimators with OOB score...")
    best_oob = -np.inf
    best_model = None
    no_improve_steps = 0

    for n in range(start_trees, max_trees + 1, step):
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=base_depth, random_state=42),
            n_estimators=n,
            max_samples=0.8,
            max_features=0.8,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )
        bag.fit(X_train, y_train)
        oob = bag.oob_score_
        print(f"  Trees={n:3d} | OOB Score={oob:.4f}")

        if oob > best_oob + min_improve:
            best_oob = oob
            best_model = deepcopy(bag)
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            if no_improve_steps >= patience:
                print(f"  ✓ Early stopping: No improvement for {patience} steps")
                break

    print(f"\n✓ Best OOB Score: {best_oob:.4f} with {best_model.n_estimators} trees")
    return best_model, best_oob


best_bagging, best_oob = fit_bagging_with_early_stop(X_train, y_train)

# Evaluate on test set
y_pred = best_bagging.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"\n✓ Test Accuracy:     {acc:.4f}")
print(f"✓ Macro F1-score:    {f1_macro:.4f}")
print(f"✓ Weighted F1-score: {f1_weighted:.4f}")
print(f"\nOOB Score (training):     {best_oob:.4f}")
print(f"Test Accuracy (held-out): {acc:.4f}")
print(f"Difference (OOB - Test):  {best_oob - acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

results.append(
    {
        "Model": "Bagging (Decision Trees)",
        "Accuracy": acc,
        "Macro_F1": f1_macro,
        "Weighted_F1": f1_weighted,
    }
)

# ----------------------------------------------------------------------------
# MODEL 4: BAGGING WITH LOGISTIC REGRESSION
# ----------------------------------------------------------------------------
print("\n" + "-" * 80)
print("BAGGING MODEL 2: Bagging with Logistic Regression")
print("-" * 80)

# Use Pipeline instead of custom class for sklearn compatibility
from sklearn.pipeline import Pipeline

bagging_lr = BaggingClassifier(
    estimator=Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="ovr",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
    n_estimators=50,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)

result = evaluate_model(
    "Bagging (Logistic Regression)",
    bagging_lr,
    X_train,
    y_train,
    X_test,
    y_test,
    "Bootstrap aggregating with linear models",
)
results.append(result)

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("FINAL COMPARISON: BOOSTING vs BAGGING")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Macro_F1", ascending=False)

print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv("boosting_bagging_results.csv", index=False)
print("\n✓ Results saved to: boosting_bagging_results.csv")

# ============================================================================
# VISUALIZATION: Performance Comparison
# ============================================================================
print("\n[Generating visualization...]")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy comparison
ax1 = axes[0]
sns.barplot(data=results_df, x="Model", y="Accuracy", ax=ax1, palette="viridis")
ax1.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
ax1.set_ylabel("Accuracy", fontsize=12)
ax1.set_xlabel("")
ax1.set_ylim([results_df["Accuracy"].min() - 0.02, 1.0])
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

# Add value labels
for i, v in enumerate(results_df["Accuracy"]):
    ax1.text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

# Plot 2: F1-score comparison
ax2 = axes[1]
x_pos = np.arange(len(results_df))
width = 0.35
ax2.bar(x_pos - width / 2, results_df["Macro_F1"], width, label="Macro F1", alpha=0.8)
ax2.bar(
    x_pos + width / 2, results_df["Weighted_F1"], width, label="Weighted F1", alpha=0.8
)
ax2.set_title("F1-Score Comparison", fontsize=14, fontweight="bold")
ax2.set_ylabel("F1-Score", fontsize=12)
ax2.set_xlabel("")
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_df["Model"], rotation=45, ha="right")
ax2.legend()
ax2.set_ylim([results_df[["Macro_F1", "Weighted_F1"]].min().min() - 0.02, 1.0])

plt.tight_layout()
plt.savefig("boosting_bagging_comparison.png", dpi=300, bbox_inches="tight")
print("✓ Visualization saved to: boosting_bagging_comparison.png")
plt.show()

print("\n" + "=" * 80)
print("PART 7 COMPLETE!")
print("=" * 80)
