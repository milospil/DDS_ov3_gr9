# file: run_five_models_presplit.py
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# =========================
# 1) FILER (tilpass stier hvis nødvendig)
# =========================
DATA_DIR = "competition"
X_TRAIN_PATH = f"{DATA_DIR}/x_train.csv"
Y_TRAIN_PATH = f"{DATA_DIR}/y_train.csv"
X_TEST_PATH  = f"{DATA_DIR}/x_test.csv"
Y_TEST_PATH  = f"{DATA_DIR}/y_test.csv"

TARGET = "SMK_stat_type_cd"
ID_COL = "id"
DROP_IF_EXISTS = ["drink"]  # om den finnes

# =========================
# 2) LAST OG ALIGN PÅ id
# =========================
def load_and_align(x_path, y_path):
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)

    # Sjekk at id finnes
    assert ID_COL in X.columns, f"Mangler '{ID_COL}' i {x_path}"
    assert ID_COL in y.columns, f"Mangler '{ID_COL}' i {y_path}"
    assert TARGET in y.columns, f"Mangler target '{TARGET}' i {y_path}"

    # Align på id (inner join for sikker matching)
    X = X.set_index(ID_COL)
    y = y.set_index(ID_COL).sort_index()

    # Behold bare felles id-er og samme rekkefølge
    common_ids = X.index.intersection(y.index)
    X = X.loc[common_ids].copy()
    y = y.loc[common_ids].copy()

    # Fjern ev. uønskede kolonner
    X.drop(columns=[c for c in DROP_IF_EXISTS if c in X.columns], inplace=True, errors="ignore")

    # Sørg for numeriske typer på features
    X = X.astype(np.float32, errors="ignore")

    # y som int
    y = y[TARGET].astype(int)

    return X, y

X_train, y_train = load_and_align(X_TRAIN_PATH, Y_TRAIN_PATH)
X_test,  y_test  = load_and_align(X_TEST_PATH,  Y_TEST_PATH)

# Sikre identiske kolonner/rekkefølge i test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

# =========================
# 3) DEFINER MODELLER
#    (skalering KUN i pipeline for lineære metoder)
# =========================
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ))
])

nb = Pipeline([
    ("scaler", StandardScaler()),   # gjør Gauss NB mer stabil når skalaer varierer
    ("clf", GaussianNB())
])

nn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        random_state=42,
        verbose=False
    ))
])

# Tre-baserte: ingen skalering
dt = DecisionTreeClassifier(
    max_depth=14,
    min_samples_leaf=30,
    class_weight="balanced",
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    min_samples_leaf=20,
    max_features="sqrt",
    bootstrap=True,
    max_samples=0.3,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
    verbose=0
)

models = {
    "LogisticRegression": logreg,
    "NaiveBayes": nb,
    "DecisionTree": dt,
    "RandomForest": rf,
    "NeuralNetwork": nn,
}

# =========================
# 4) TREN & EVALUER (kun pre-splittet train/test)
# =========================
summary = []

def evaluate(name, model, Xtr, ytr, Xte, yte):
    print(f"\n=== {name} ===")

    # sample weights for ubalanse der det støttes
    sw = None
    if name in ["LogisticRegression", "NaiveBayes", "NeuralNetwork"]:
        sw = compute_sample_weight(class_weight="balanced", y=ytr)

    # Fit (legg merke til at sample_weight i Pipeline må rutes til siste steg-navn 'clf')
    if sw is not None:
        model.fit(Xtr, ytr, clf__sample_weight=sw)
    else:
        model.fit(Xtr, ytr)

    # Evaluer på test
    y_pred = model.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    f1m = f1_score(yte, y_pred, average="macro")
    print(f"TEST -> Acc: {acc:.3f} | Macro-F1: {f1m:.3f}")
    print(classification_report(yte, y_pred, digits=3))

    summary.append({"Model": name, "Test_Acc": acc, "Test_MacroF1": f1m})

for name, model in models.items():
    evaluate(name, model, X_train, y_train, X_test, y_test)

# =========================
# 5) OPPSUMMERING
# =========================
summary_df = pd.DataFrame(summary)
print("\n=== Summary (Test) ===")
print(summary_df.sort_values("Test_MacroF1", ascending=False).to_string(index=False))

summary_df.to_csv("model_summary_presplit.csv", index=False)
print("\n📁 Lagret: model_summary_presplit.csv")
