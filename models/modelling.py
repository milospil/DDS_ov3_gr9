# 7_modeling.py - COMPLETE FILE FOR TASKS 5-6

"""
Assignment 3 - Tasks 5 & 6: Model Implementation and Comparison
Implements 5 machine learning algorithms for smoking detection
"""

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

# Algorithm imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD PREPROCESSED DATA
# ============================================
print("Loading preprocessed data...")
df = pd.read_csv('datasets/preprocessed_data.csv')

# Separate features and target
X = df.drop('SMK_stat', axis=1)
y = df['SMK_stat']

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}\n")

# ============================================
# 2. TRAIN-TEST SPLIT
# ============================================
print("Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (for Logistic Regression, SVM, Neural Network)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}\n")

# ============================================
# 3. MODEL TRAINING & EVALUATION
# ============================================

# Storage for results
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'ROC-AUC': []
}

predictions = {}

# -------------------- ALGORITHM 1: LOGISTIC REGRESSION --------------------
print("="*60)
print("TRAINING ALGORITHM 1: LOGISTIC REGRESSION")
print("="*60)

log_reg = LogisticRegression(max_iter=1000, random_state=42, 
                              multi_class='multinomial', solver='lbfgs')
log_reg.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_proba_lr = log_reg.predict_proba(X_test_scaled)

# Store results
results['Model'].append('Logistic Regression')
results['Accuracy'].append(accuracy_score(y_test, y_pred_lr))
results['Precision'].append(precision_score(y_test, y_pred_lr, average='weighted'))
results['Recall'].append(recall_score(y_test, y_pred_lr, average='weighted'))
results['F1-Score'].append(f1_score(y_test, y_pred_lr, average='weighted'))
results['ROC-AUC'].append(roc_auc_score(y_test, y_pred_proba_lr, 
                                        multi_class='ovr', average='weighted'))
predictions['Logistic Regression'] = y_pred_lr

print(f"Accuracy: {results['Accuracy'][-1]:.4f}")
print(f"F1-Score: {results['F1-Score'][-1]:.4f}")
print(f"ROC-AUC: {results['ROC-AUC'][-1]:.4f}\n")


#EXTRACTION OF COEFFICIENTS for transfer learning purposes

"""
# Get coefficients (weights) for each class
coefficients = log_reg.coef_  # Shape: (n_classes, n_features)
intercepts = log_reg.intercept_  # Shape: (n_classes,)
"""

# -------------------- ALGORITHM 2: DECISION TREE --------------------
print("="*60)
print("TRAINING ALGORITHM 2: DECISION TREE")
print("="*60)

dt = DecisionTreeClassifier(max_depth=10, min_samples_split=50, 
                             min_samples_leaf=20, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_pred_proba_dt = dt.predict_proba(X_test)

results['Model'].append('Decision Tree')
results['Accuracy'].append(accuracy_score(y_test, y_pred_dt))
results['Precision'].append(precision_score(y_test, y_pred_dt, average='weighted'))
results['Recall'].append(recall_score(y_test, y_pred_dt, average='weighted'))
results['F1-Score'].append(f1_score(y_test, y_pred_dt, average='weighted'))
results['ROC-AUC'].append(roc_auc_score(y_test, y_pred_proba_dt, 
                                        multi_class='ovr', average='weighted'))
predictions['Decision Tree'] = y_pred_dt

print(f"Accuracy: {results['Accuracy'][-1]:.4f}")
print(f"F1-Score: {results['F1-Score'][-1]:.4f}")
print(f"ROC-AUC: {results['ROC-AUC'][-1]:.4f}\n")

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(dt, max_depth=3, feature_names=X.columns, 
          class_names=['Non-smoker', 'Former', 'Current'],
          filled=True, fontsize=10)
plt.title("Decision Tree (First 3 Levels)")
plt.savefig('../outputs/figures/decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------- ALGORITHM 3: RANDOM FOREST --------------------
print("="*60)
print("TRAINING ALGORITHM 3: RANDOM FOREST")
print("="*60)

rf = RandomForestClassifier(n_estimators=200, max_depth=15, 
                             min_samples_split=20, min_samples_leaf=10,
                             max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)

results['Model'].append('Random Forest')
results['Accuracy'].append(accuracy_score(y_test, y_pred_rf))
results['Precision'].append(precision_score(y_test, y_pred_rf, average='weighted'))
results['Recall'].append(recall_score(y_test, y_pred_rf, average='weighted'))
results['F1-Score'].append(f1_score(y_test, y_pred_rf, average='weighted'))
results['ROC-AUC'].append(roc_auc_score(y_test, y_pred_proba_rf, 
                                        multi_class='ovr', average='weighted'))
predictions['Random Forest'] = y_pred_rf

print(f"Accuracy: {results['Accuracy'][-1]:.4f}")
print(f"F1-Score: {results['F1-Score'][-1]:.4f}")
print(f"ROC-AUC: {results['ROC-AUC'][-1]:.4f}\n")

# Feature importance
feature_importance_rf = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 8))
top_features = feature_importance_rf.head(15)
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances - Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../outputs/figures/rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# -------------------- ALGORITHM 4: SVM WITH RBF KERNEL --------------------
print("="*60)
print("TRAINING ALGORITHM 4: SVM WITH RBF KERNEL")
print("="*60)

svm_rbf = SVC(kernel='rbf', C=10, gamma='scale', 
              probability=True, random_state=42)
svm_rbf.fit(X_train_scaled, y_train)

y_pred_svm = svm_rbf.predict(X_test_scaled)
y_pred_proba_svm = svm_rbf.predict_proba(X_test_scaled)

results['Model'].append('SVM (RBF)')
results['Accuracy'].append(accuracy_score(y_test, y_pred_svm))
results['Precision'].append(precision_score(y_test, y_pred_svm, average='weighted'))
results['Recall'].append(recall_score(y_test, y_pred_svm, average='weighted'))
results['F1-Score'].append(f1_score(y_test, y_pred_svm, average='weighted'))
results['ROC-AUC'].append(roc_auc_score(y_test, y_pred_proba_svm, 
                                        multi_class='ovr', average='weighted'))
predictions['SVM (RBF)'] = y_pred_svm

print(f"Accuracy: {results['Accuracy'][-1]:.4f}")
print(f"F1-Score: {results['F1-Score'][-1]:.4f}")
print(f"ROC-AUC: {results['ROC-AUC'][-1]:.4f}\n")

# -------------------- ALGORITHM 5: NEURAL NETWORK --------------------
print("="*60)
print("TRAINING ALGORITHM 5: NEURAL NETWORK (MLP)")
print("="*60)

mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 25), activation='relu',
                    solver='adam', alpha=0.001, batch_size=64,
                    learning_rate='adaptive', max_iter=500,
                    early_stopping=True, validation_fraction=0.1,
                    random_state=42, verbose=False)
mlp.fit(X_train_scaled, y_train)

y_pred_mlp = mlp.predict(X_test_scaled)
y_pred_proba_mlp = mlp.predict_proba(X_test_scaled)

results['Model'].append('Neural Network')
results['Accuracy'].append(accuracy_score(y_test, y_pred_mlp))
results['Precision'].append(precision_score(y_test, y_pred_mlp, average='weighted'))
results['Recall'].append(recall_score(y_test, y_pred_mlp, average='weighted'))
results['F1-Score'].append(f1_score(y_test, y_pred_mlp, average='weighted'))
results['ROC-AUC'].append(roc_auc_score(y_test, y_pred_proba_mlp, 
                                        multi_class='ovr', average='weighted'))
predictions['Neural Network'] = y_pred_mlp

print(f"Accuracy: {results['Accuracy'][-1]:.4f}")
print(f"F1-Score: {results['F1-Score'][-1]:.4f}")
print(f"ROC-AUC: {results['ROC-AUC'][-1]:.4f}\n")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Neural Network Training Loss Curve')
plt.grid(True)
plt.savefig('../outputs/figures/nn_loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 4. COMPARISON & VISUALIZATION
# ============================================
print("="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('../outputs/results/model_results.csv', index=False)

# Visualize comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    bars = ax.bar(results_df['Model'], results_df[metric], color='steelblue')
    
    # Highlight best performer
    best_idx = results_df[metric].idxmax()
    bars[best_idx].set_color('darkgreen')
    
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([results_df[metric].min() - 0.05, 1.0])

# Remove empty subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('../outputs/figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nAll models trained and evaluated successfully!")
print(f" Results saved to: ../outputs/results/model_results.csv")
print(f" Visualizations saved to: ../outputs/figures/")
print("\n" + "="*60)
print(f" BEST MODEL: {results_df.loc[results_df['F1-Score'].idxmax(), 'Model']}")
print(f"   F1-Score: {results_df['F1-Score'].max():.4f}")
print(f"   ROC-AUC: {results_df.loc[results_df['F1-Score'].idxmax(), 'ROC-AUC']:.4f}")
print("="*60)