# file: run_five_models.py
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# ---------- 1) Les data ----------
# Bruk en fil der features er numeriske og target-kolonnen heter SMK_stat_type_cd
df = pd.read_csv("datasets/preprocessed_data.csv")  # tilpass om du bruker en annen fil

target = "SMK_stat_type_cd"
drop_if_exists = ["drink"]  # fjern hvis finnes
X = df.drop(columns=[c for c in [target] + drop_if_exists if c in df.columns])
y = df[target].astype(int)

# (valgfritt) sørg for float for numeriske features
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
# Modeller som trenger skalering: LogReg, NB, SVM
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, multi_class="ovr",
                               class_weight="balanced", n_jobs=-1, random_state=42))
])

nb = Pipeline([
    ("scaler", StandardScaler()),   # hjelper når features har veldig ulik skala
    ("clf", GaussianNB())
])

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000))
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
    "SVM": svm,
}

# ---------- 4) Tren og evaluer ----------
summary = []

def evaluate(name, model, Xtr, ytr, Xva, yva, Xte, yte):
    print(f"\n=== {name} ===")
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
    # Bruk undersamplet train for tunge modeller om ønskelig
    if name in ["RandomForest", "SVM"]:
        evaluate(name, model, X_train_fast, y_train_fast, X_val, y_val, X_test, y_test)
    else:
        evaluate(name, model, X_train, y_train, X_val, y_val, X_test, y_test)

# ---------- 5) Oppsummering ----------
summary_df = pd.DataFrame(summary)
print("\n=== Summary ===")
print(summary_df.sort_values("Test_MacroF1", ascending=False).to_string(index=False))

# (valgfritt) lagre til CSV
summary_df.to_csv("model_summary.csv", index=False)
print("\n📁 Lagret: model_summary.csv")