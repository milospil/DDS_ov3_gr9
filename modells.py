# file: run_five_models.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier  # <-- NY!

# ---------- 1) Les data ----------
df = pd.read_csv("datasets/preprocessed_data.csv")

target = "SMK_stat_type_cd"
drop_if_exists = ["drink"]
X = df.drop(columns=[c for c in [target] + drop_if_exists if c in df.columns])
y = df[target].astype(int)

# sikre flyt for numeriske features
X = X.astype(np.float32, errors="ignore")

# ---------- 2) Train/Val/Test split ----------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# (Valgfritt) raskere trening for store datasett
def maybe_subsample(X_, y_, max_rows=150_000):
    if len(X_) <= max_rows:
        return X_, y_
    frac = max_rows / len(X_)
    X_s, _, y_s, _ = train_test_split(X_, y_, train_size=frac, stratify=y_, random_state=42)
    return X_s, y_s

X_train_fast, y_train_fast = maybe_subsample(X_train, y_train, max_rows=150_000)

# ---------- 3) Modeller ----------
# Modeller som trenger skalering: LogReg, NB, NN
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                               n_jobs=-1, random_state=42))
])

nb = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GaussianNB())
])

# Neural Network (MLP) med early stopping
nn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,                 # L2-reg
        learning_rate="adaptive",
        max_iter=100,               # stoppes ofte tidligere av early_stopping
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    ))
])

# Trær trenger ikke skalering
dt = DecisionTreeClassifier(
    max_depth=14, min_samples_leaf=30, class_weight="balanced", random_state=42
)

rf = RandomForestClassifier(
    n_estimators=250, max_depth=12, min_samples_leaf=20, max_features="sqrt",
    bootstrap=True, max_samples=0.3, n_jobs=-1, class_weight="balanced",
    random_state=42, verbose=0
)

models = {
    "LogisticRegression": logreg,
    "NaiveBayes": nb,
    "DecisionTree": dt,
    "RandomForest": rf,
    "NeuralNetwork": nn,   # <-- byttet ut SVM med NN
}

# ---------- 4) Tren og evaluer ----------
summary = []

def evaluate(name, model, Xtr, ytr, Xva, yva, Xte, yte):
    print(f"\n=== {name} ===")

    # Sample weights for class imbalance (brukes kun der det støttes)
    sw = None
    if name in ["NeuralNetwork"]:  # MLPClassifier støtter sample_weight
        sw = compute_sample_weight(class_weight="balanced", y=ytr)

    # fit (med eller uten sample weights)
    if sw is not None:
        model.fit(Xtr, ytr, clf__sample_weight=sw)  # pass videre til 'clf' i pipeline
    else:
        model.fit(Xtr, ytr)

    # VAL
    yv = model.predict(Xva)
    acc_val = accuracy_score(yva, yv)
    f1_val = f1_score(yva, yv, average="macro")
    print(f"VAL  -> Acc: {acc_val:.3f} | Macro-F1: {f1_val:.3f}")

    # TEST
    yt = model.predict(Xte)
    acc = accuracy_score(yte, yt)
    f1m = f1_score(yte, yt, average="macro")
    print(f"TEST -> Acc: {acc:.3f} | Macro-F1: {f1m:.3f}")
    print(classification_report(yte, yt, digits=3))

    summary.append({"Model": name, "Val_Acc": acc_val, "Val_MacroF1": f1_val,
                    "Test_Acc": acc, "Test_MacroF1": f1m})

for name, model in models.items():
    # Bruk undersamplet train for tunge modeller (RF og NN) om ønskelig
    if name in ["RandomForest", "NeuralNetwork"]:
        evaluate(name, model, X_train_fast, y_train_fast, X_val, y_val, X_test, y_test)
    else:
        evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test)

# ---------- 5) Oppsummering ----------
summary_df = pd.DataFrame(summary)
print("\n=== Summary ===")
print(summary_df.sort_values("Test_MacroF1", ascending=False).to_string(index=False))

summary_df.to_csv("model_summary.csv", index=False)
print("\n📁 Lagret: model_summary.csv")
