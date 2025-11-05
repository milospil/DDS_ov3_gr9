# file: run_transfer_learning.py
"""
Part 8: Transfer Learning Implementation

Approach: 
- Pre-train on 60% dataset (bigger related dataset from "online")
- Fine-tune on 40% dataset (our actual dataset from Parts 5-7)
- Compare with baseline trained only on 40%
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

print("="*80)
print("PART 8: TRANSFER LEARNING")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATASETS
# ============================================================================
print("\n[STEP 1] Loading datasets...")

# Source dataset (60% - bigger related dataset from "online")
source_df = pd.read_csv("competition/dataset_60_percent.csv")

# Target dataset (40% - our actual dataset used in Parts 5-7)
target_df = pd.read_csv("competition/dataset_40_percent.csv")

# Assuming target column is 'SMK_stat_type_cd'
target_col = "SMK_stat_type_cd"

# Separate features and target
X_source = source_df.drop(columns=[target_col]).astype(np.float32)
y_source = source_df[target_col].astype(int)

X_target = target_df.drop(columns=[target_col]).astype(np.float32)
y_target = target_df[target_col].astype(int)

print(f"✓ Source dataset (60% - bigger dataset): {X_source.shape[0]:,} samples")
print(f"✓ Target dataset (40% - our dataset): {X_target.shape[0]:,} samples")
print(f"✓ Features: {X_source.shape[1]}")
print(f"\nTransfer Learning Setup:")
print(f"  1. Pre-train on 60% (bigger related dataset)")
print(f"  2. Fine-tune on 40% (our actual dataset from Parts 5-7)")
print(f"  3. Compare with Part 5-7 models trained only on 40%")

# ============================================================================
# STEP 2: EXPLORE SOURCE DATASET (BIGGER DATASET FROM "ONLINE")
# ============================================================================
print("\n" + "="*80)
print("SOURCE DATASET ANALYSIS (60% - Bigger Related Dataset)")
print("="*80)

print("\n--- Summary Statistics ---")
print(X_source.describe())

print("\n--- Target Distribution ---")
print(y_source.value_counts().sort_index())
print(f"\nClass proportions:")
for cls in sorted(y_source.unique()):
    prop = (y_source == cls).sum() / len(y_source)
    print(f"  Class {cls}: {prop*100:.1f}%")

print("\n--- Feature Statistics ---")
print("\nContinuous features (first 5):")
for col in X_source.columns[:5]:
    print(f"  {col}: mean={X_source[col].mean():.2f}, std={X_source[col].std():.2f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Target distribution
ax1 = axes[0, 0]
target_counts = y_source.value_counts().sort_index()
ax1.bar(target_counts.index, target_counts.values, color=['#3498db', '#e74c3c', '#2ecc71'])
ax1.set_xlabel('Smoking Status', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Source Dataset: Target Distribution', fontsize=14, fontweight='bold')
ax1.set_xticks([1, 2, 3])
ax1.set_xticklabels(['Never\nSmoker', 'Ex-\nSmoker', 'Current\nSmoker'])
for i, (idx, val) in enumerate(target_counts.items()):
    ax1.text(idx, val, f'{val:,}\n({val/len(y_source)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=10)

# 2. Distribution of a key feature (e.g., hemoglobin)
ax2 = axes[0, 1]
if 'hemoglobin' in X_source.columns:
    feature_to_plot = 'hemoglobin'
elif 'BMI' in X_source.columns:
    feature_to_plot = 'BMI'
else:
    feature_to_plot = X_source.columns[0]
    
X_source[feature_to_plot].hist(bins=50, ax=ax2, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel(feature_to_plot, fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title(f'Source Dataset: {feature_to_plot} Distribution', fontsize=14, fontweight='bold')
ax2.axvline(X_source[feature_to_plot].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax2.legend()

# 3. Correlation heatmap (top features)
ax3 = axes[1, 0]
top_features = X_source.columns[:8]  # First 8 features
corr_matrix = X_source[top_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax3, cbar_kws={'label': 'Correlation'})
ax3.set_title('Source Dataset: Feature Correlations', fontsize=14, fontweight='bold')

# 4. Source vs Target comparison (sample sizes)
ax4 = axes[1, 1]
dataset_sizes = ['Source\n(60%)', 'Target\n(40%)']
sizes = [len(X_source), len(X_target)]
colors_bar = ['#3498db', '#e74c3c']
bars = ax4.bar(dataset_sizes, sizes, color=colors_bar, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Number of Samples', fontsize=12)
ax4.set_title('Dataset Sizes', fontsize=14, fontweight='bold')
for bar, size in zip(bars, sizes):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{size:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('source_dataset_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: source_dataset_analysis.png")
plt.show()

# ============================================================================
# STEP 3: SPLIT TARGET DATASET
# ============================================================================
print("\n[STEP 3] Splitting target dataset for fine-tuning and evaluation...")

# Split target into train (for fine-tuning) and test (for evaluation)
X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target, y_target, test_size=0.3, stratify=y_target, random_state=42
)

print(f"✓ Target train (for fine-tuning): {len(X_target_train):,}")
print(f"✓ Target test (for evaluation): {len(X_target_test):,}")

# ============================================================================
# STEP 4: PRE-TRAINING ON SOURCE DATASET
# ============================================================================
print("\n" + "="*80)
print("PRE-TRAINING PHASE: Training on Source Dataset (60%)")
print("="*80)

pretrain_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    min_samples_leaf=20,
    max_features='sqrt',
    random_state=42,
    verbose=0
)

print("\nTraining on source dataset...")
pretrain_model.fit(X_source, y_source)

# Evaluate on source
y_source_pred = pretrain_model.predict(X_source)
source_acc = accuracy_score(y_source, y_source_pred)
source_f1 = f1_score(y_source, y_source_pred, average='macro')

print(f"\n✓ Pre-training complete")
print(f"  Source dataset performance:")
print(f"    Accuracy: {source_acc:.4f}")
print(f"    Macro F1: {source_f1:.4f}")

# ============================================================================
# STEP 5: TRANSFER LEARNING - FINE-TUNING ON TARGET
# ============================================================================
print("\n" + "="*80)
print("FINE-TUNING PHASE: Adapting to Target Dataset (40%)")
print("="*80)

# Create fine-tuning model with warm_start
transfer_model = GradientBoostingClassifier(
    n_estimators=300,  # 200 pre-trained + 100 new
    learning_rate=0.01,  # Lower learning rate for fine-tuning
    max_depth=5,
    subsample=0.8,
    min_samples_leaf=20,
    max_features='sqrt',
    warm_start=True,  # Critical: continues from pre-trained model
    random_state=42,
    verbose=0
)

# Copy pre-trained parameters
print("\nInitializing with pre-trained model...")
transfer_model.fit(X_source, y_source)  # First fit with same data as pre-train
print("✓ Loaded pre-trained weights")

# Now fine-tune on target
print("\nFine-tuning on target dataset...")
transfer_model.fit(X_target_train, y_target_train)
print("✓ Fine-tuning complete")

# Evaluate transfer learning model
y_transfer_pred = transfer_model.predict(X_target_test)
transfer_acc = accuracy_score(y_target_test, y_transfer_pred)
transfer_f1_macro = f1_score(y_target_test, y_transfer_pred, average='macro')
transfer_f1_weighted = f1_score(y_target_test, y_transfer_pred, average='weighted')

print(f"\nTransfer Learning Model Performance:")
print(f"  Test Accuracy: {transfer_acc:.4f}")
print(f"  Macro F1: {transfer_f1_macro:.4f}")
print(f"  Weighted F1: {transfer_f1_weighted:.4f}")

print("\nClassification Report:")
print(classification_report(y_target_test, y_transfer_pred, digits=4))

# ============================================================================
# STEP 6: BASELINE - TRAIN ONLY ON TARGET (AS IN PARTS 5-7)
# ============================================================================
print("\n" + "="*80)
print("BASELINE: Training ONLY on Target Dataset (40% - As in Parts 5-7)")
print("="*80)

baseline_model = GradientBoostingClassifier(
    n_estimators=300,  # Same total as transfer model
    learning_rate=0.05,  # Standard learning rate
    max_depth=5,
    subsample=0.8,
    min_samples_leaf=20,
    max_features='sqrt',
    random_state=42,
    verbose=0
)

print("\nTraining baseline model from scratch...")
baseline_model.fit(X_target_train, y_target_train)
print("✓ Training complete")

# Evaluate baseline
y_baseline_pred = baseline_model.predict(X_target_test)
baseline_acc = accuracy_score(y_target_test, y_baseline_pred)
baseline_f1_macro = f1_score(y_target_test, y_baseline_pred, average='macro')
baseline_f1_weighted = f1_score(y_target_test, y_baseline_pred, average='weighted')

print(f"\nBaseline Model Performance:")
print(f"  Test Accuracy: {baseline_acc:.4f}")
print(f"  Macro F1: {baseline_f1_macro:.4f}")
print(f"  Weighted F1: {baseline_f1_weighted:.4f}")

print("\nClassification Report:")
print(classification_report(y_target_test, y_baseline_pred, digits=4))

# ============================================================================
# STEP 7: COMPARISON
# ============================================================================
print("\n" + "="*80)
print("TRANSFER LEARNING vs BASELINE COMPARISON")
print("="*80)

results = pd.DataFrame({
    'Model': ['Transfer Learning', 'Baseline (No Transfer)'],
    'Training': ['60% pre-train + 40% fine-tune', '40% only'],
    'Accuracy': [transfer_acc, baseline_acc],
    'Macro_F1': [transfer_f1_macro, baseline_f1_macro],
    'Weighted_F1': [transfer_f1_weighted, baseline_f1_weighted]
})

print("\n" + results.to_string(index=False))

# Calculate improvement
improvement_acc = transfer_acc - baseline_acc
improvement_f1_macro = transfer_f1_macro - baseline_f1_macro
improvement_f1_weighted = transfer_f1_weighted - baseline_f1_weighted

print(f"\n--- Transfer Learning Improvement ---")
print(f"Accuracy:     +{improvement_acc:.4f} ({improvement_acc/baseline_acc*100:+.2f}%)")
print(f"Macro F1:     +{improvement_f1_macro:.4f} ({improvement_f1_macro/baseline_f1_macro*100:+.2f}%)")
print(f"Weighted F1:  +{improvement_f1_weighted:.4f} ({improvement_f1_weighted/baseline_f1_weighted*100:+.2f}%)")

# Save results
results.to_csv('transfer_learning_results.csv', index=False)
print("\n✓ Results saved to: transfer_learning_results.csv")

# ============================================================================
# STEP 8: VISUALIZATION
# ============================================================================
print("\n[STEP 8] Creating comparison visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy comparison
ax1 = axes[0]
models = ['Transfer\nLearning', 'Baseline']
accuracies = [transfer_acc, baseline_acc]
colors = ['#2ecc71', '#e74c3c']
bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim([min(accuracies) - 0.02, 1.0])
ax1.axhline(y=baseline_acc, color='red', linestyle='--', alpha=0.5, label='Baseline')
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
if improvement_acc > 0:
    ax1.text(0.5, max(accuracies) - 0.01, f'Improvement:\n+{improvement_acc:.4f}',
             ha='center', va='top', fontsize=10, color='green', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: F1-Score comparison
ax2 = axes[1]
x = np.arange(len(models))
width = 0.35
bars2a = ax2.bar(x - width/2, [transfer_f1_macro, baseline_f1_macro], width, 
                 label='Macro F1', alpha=0.8, color='#3498db')
bars2b = ax2.bar(x + width/2, [transfer_f1_weighted, baseline_f1_weighted], width,
                 label='Weighted F1', alpha=0.8, color='#e67e22')
ax2.set_ylabel('F1-Score', fontsize=12)
ax2.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.set_ylim([min(baseline_f1_macro, baseline_f1_weighted) - 0.02, 1.0])

plt.tight_layout()
plt.savefig('transfer_learning_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: transfer_learning_comparison.png")
plt.show()

print("\n" + "="*80)
print("PART 8 COMPLETE!")
print("="*80)
print("\nFiles created:")
print("  • source_dataset_analysis.png")
print("  • transfer_learning_comparison.png")
print("  • transfer_learning_results.csv")