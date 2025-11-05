from copy import deepcopy
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# TODO: Legge inn riktig test, train datasett
# Helst unscaled datasett, fordi decision trees ikke bruker avstander, men vet ikke om det er
# krise å bruke et scaled dataset heller

# 1) Les med header og id som index (ingen aligning nødvendig)
X_train = pd.read_csv("competition/x_train.csv", header=0, index_col="id").astype(np.float32)
y_train = pd.read_csv("competition/y_train.csv", header=0, index_col="id").squeeze("columns")
X_test  = pd.read_csv("competition/x_test.csv",  header=0, index_col="id").astype(np.float32)
y_test  = pd.read_csv("competition/y_test.csv",  header=0, index_col="id").squeeze("columns")

# y til heltall 1/2/3
y_train = pd.to_numeric(y_train, errors="raise").astype(int)
y_test  = pd.to_numeric(y_test,  errors="raise").astype(int)

# valgfritt: kjapp sjekk at indexene matcher
assert (X_train.index == y_train.index).all()
assert (X_test.index  == y_test.index ).all()

"""
def fit_bagging_with_early_stop(
    X_train, y_train,
    base_depth=None,
    start_trees=50,
    step=50,
    max_trees=800,
    patience=3,           # hvor mange "ingen forbedring"-steg vi tolererer
    min_improve=1e-3      # minste forbedring i OOB for å fortsette
):
    best_oob = -np.inf
    best_model = None
    no_improve_steps = 0

    # init
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=base_depth, random_state=42),
        n_estimators=start_trees,
        max_samples=0.8,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=42,
        warm_start=True  # gjør at vi kan "legge til" trær i samme modell
    )

    while bag.n_estimators <= max_trees:
        bag.fit(X_train, y_train)
        oob = bag.oob_score_
        print(f"Trees={bag.n_estimators:3d} | OOB={oob:.4f}")

        if oob > best_oob + min_improve:
            best_oob = oob
            best_model = deepcopy(bag)
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            if no_improve_steps >= patience:
                print("Early stop: OOB plateau.")
                break

        bag.n_estimators += step  # legg til flere trær

    return best_model, best_oob
"""

def fit_bagging_with_early_stop(
    X_train, y_train,
    base_depth=None,
    start_trees=100,
    step=200,
    max_trees=800,
    patience=3,
    min_improve=1e-3,
    random_state=42
):
    best_oob = -np.inf
    best_model = None
    no_improve_steps = 0

    for n in range(start_trees, max_trees + 1, step):
        bag = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=base_depth, random_state=random_state),
            n_estimators=n,
            max_samples=0.8,
            bootstrap=True,
            oob_score=True,      # ✔ tillatt fordi warm_start ikke brukes
            n_jobs=-1,
            random_state=random_state,
            # warm_start=False er default
        )
        bag.fit(X_train, y_train)
        oob = bag.oob_score_
        print(f"Trees={n:3d} | OOB={oob:.4f}")

        if oob > best_oob + min_improve:
            best_oob = oob
            best_model = deepcopy(bag)
            no_improve_steps = 0
        else:
            no_improve_steps += 1
            if no_improve_steps >= patience:
                print("Early stop: OOB plateau.")
                break

    return best_model, best_oob


best_bag, best_oob = fit_bagging_with_early_stop(X_train, y_train)
print(f"\nBest OOB: {best_oob:.4f} with {best_bag.n_estimators} trees")


# --- Prediksjoner på testsett ---
y_pred = best_bag.predict(X_test)

# --- Evaluer nøyaktighet og F1 ---
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

print(f"\nModel performance on test set:")
print(f"Accuracy (riktig andel): {acc:.3f}")
print(f"Macro F1-score (balansert): {f1m:.3f}")

print("\nFull classification report:\n")
print(classification_report(y_test, y_pred, digits=3))

# --- Sammenlikne OOB score og test accuracy ---
print(f"OOB Score (training estimate): {best_oob:.3f}")
print(f"Test Accuracy (held-out data):  {acc:.3f}")


# --- Lage confusion matrix --- # SLETTE??
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix – Bagging ({best_bag.n_estimators} trees)\nAccuracy={acc:.3f}")
plt.tight_layout()
plt.show()
