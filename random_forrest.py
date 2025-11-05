import matplotlib
matplotlib.use("Agg")  # unngå GUI-blokkering
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === 1) Les og del data
df = pd.read_csv("datasets/smoking_drinking_scaled.csv")

drop_cols = [c for c in ['SMK_stat_type_cd', 'drink'] if c in df.columns]
X = df.drop(columns=drop_cols).astype(np.float32)   # komprimer til float32
y = df['SMK_stat_type_cd'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === 2) Raskere RandomForest
rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,             # begrens dybde
    min_samples_leaf=20,      # grovere blader
    max_features="sqrt",
    bootstrap=True,
    max_samples=0.3,          # hver tre trenes på 30% av treningsdata
    n_jobs=-1,
    class_weight="balanced",  # bedre ved skjeve klasser
    random_state=42,
    verbose=1                 # vis fremdrift
)

rf.fit(X_train, y_train)

# === 3) Eval
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Lagre figurer til filer (ingen plt.show())
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted"); plt.ylabel("True")
plt.title("RF – Smoking Status (Confusion Matrix)")
plt.tight_layout(); plt.savefig("rf_cm.png", dpi=150); plt.close()

imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 feature importances:\n", imp.head(15).to_string())

plt.figure(figsize=(8,6))
imp.head(20).iloc[::-1].plot(kind="barh")
plt.xlabel("Relative Importance"); plt.title("RF Feature Importance – Smoking")
plt.tight_layout(); plt.savefig("rf_importance.png", dpi=150); plt.close()

print("\n Lagret: rf_cm.png, rf_importance.png")
