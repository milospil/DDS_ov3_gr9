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

# ==== NEW: timing & plots ====
import time
import matplotlib.pyplot as plt


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

# ==== NEW: complexity helper ====
def _final_estimator(model):
    return model.named_steps["clf"] if isinstance(model, Pipeline) else model

def model_complexity(model):
    est = _final_estimator(model)
    try:
        if isinstance(est, LogisticRegression):
            n_params = est.coef_.size + est.intercept_.size
            return f"LR: classes={getattr(est, 'n_classes_', 'NA')}, params={n_params}"
        if isinstance(est, GaussianNB):
            n_classes = getattr(est, 'classes_', [])
            n_features = getattr(est, 'theta_', np.empty((0,0))).shape[1] if hasattr(est, 'theta_') else 'NA'
            return f"GNB: classes={len(n_classes)}, features={n_features}"
        if isinstance(est, DecisionTreeClassifier):
            return f"DT: depth={est.get_depth()}, nodes={est.tree_.node_count}"
        if isinstance(est, RandomForestClassifier):
            depths = [t.get_depth() for t in est.estimators_]
            nodes  = [t.tree_.node_count for t in est.estimators_]
            return f"RF: trees={len(est.estimators_)}, avg_depth={np.mean(depths):.1f}, avg_nodes={np.mean(nodes):.0f}"
        if isinstance(est, MLPClassifier):
            # antall parametre = sum(|W| + |b|)
            total_params = 0
            for w, b in zip(est.coefs_, est.intercepts_):
                total_params += w.size + b.size
            arch = [w.shape[0] for w in est.coefs_] + [est.coefs_[-1].shape[1]] if est.coefs_ else []
            return f"MLP: layers={arch}, params={total_params}"
    except Exception as e:
        return f"complexity=NA ({e})"
    return "complexity=NA"

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

    # ==== NEW: time training ====
    t0 = time.time()
    if sw is not None:
        model.fit(Xtr, ytr, clf__sample_weight=sw)
    else:
        model.fit(Xtr, ytr)
    train_time = time.time() - t0

    # ==== NEW: time prediction ====
    t1 = time.time()
    y_pred = model.predict(Xte)
    pred_time = time.time() - t1

    acc = accuracy_score(yte, y_pred)
    f1m = f1_score(yte, y_pred, average="macro")
    print(f"TEST -> Acc: {acc:.3f} | Macro-F1: {f1m:.3f} | Train_s: {train_time:.2f} | Predict_s: {pred_time:.2f}")
    print(classification_report(yte, y_pred, digits=3))

    # ==== NEW: complexity string ====
    cx = model_complexity(model)

    summary.append({
        "Model": name,
        "Test_Acc": acc,
        "Test_MacroF1": f1m,
        "Train_Time_s": train_time,
        "Predict_Time_s": pred_time,
        "Complexity": cx
    })

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

# ==== NEW: simple visualizations ====
try:
    # Sort by Macro-F1 for plotting order
    s = summary_df.sort_values("Test_MacroF1", ascending=False)

    # 1) Ytelse (Accuracy & Macro-F1)
    plt.figure(figsize=(8,5))
    plt.bar(s["Model"], s["Test_Acc"])
    plt.title("Test Accuracy by Model")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("plot_test_accuracy.png", dpi=160)
    # plt.show()

    plt.figure(figsize=(8,5))
    plt.bar(s["Model"], s["Test_MacroF1"])
    plt.title("Test Macro-F1 by Model")
    plt.ylabel("Macro-F1")
    plt.tight_layout()
    plt.savefig("plot_test_macrof1.png", dpi=160)
    # plt.show()

    # 2) Tidsbruk (train & predict)
    plt.figure(figsize=(8,5))
    plt.bar(s["Model"], s["Train_Time_s"])
    plt.title("Training Time by Model (s)")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig("plot_train_time.png", dpi=160)

    plt.figure(figsize=(8,5))
    plt.bar(s["Model"], s["Predict_Time_s"])
    plt.title("Prediction Time by Model (s)")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig("plot_predict_time.png", dpi=160)

    print("🖼️ Lagret figurer: plot_test_accuracy.png, plot_test_macrof1.png, plot_train_time.png, plot_predict_time.png")
except Exception as e:
    print(f"[WARN] Plotting skipped due to error: {e}")

# ==== NEW: lag også en ren tabell med kun tall for rapport ====
summary_df[["Model","Test_Acc","Test_MacroF1","Train_Time_s","Predict_Time_s","Complexity"]] \
    .to_csv("model_summary_presplit_with_times.csv", index=False)
print("📁 Lagret: model_summary_presplit_with_times.csv")
