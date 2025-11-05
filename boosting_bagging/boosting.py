from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

# --- data ---
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

# --- boosting med decision trees ---
gb = GradientBoostingClassifier(
    n_estimators=300,    # antall weak learners (små decision trees)
    learning_rate=0.05,  # hvor mye hvert nytt tre får korrigere
    max_depth=3,         # grunn tre (stump = 1–3)
    random_state=42
)

# --- train model
gb.fit(X_train, y_train)

# --- evaluering ---
y_pred = gb.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

print(f"Accuracy: {acc:.3f}")
print(f"Macro F1: {f1m:.3f}")
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))
