"""
Assignment 3 - Task 4: Feature Selection and Justification
Selects the most important features for smoking detection using multiple methods
"""

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. LOAD DATA AFTER FEATURE EXTRACTION
# ============================================
print("Loading data after feature extraction...")
df = pd.read_csv('../data/feature_extracted_data.csv')

# Separate features and target
X = df.drop('SMK_stat', axis=1)
y = df['SMK_stat']

print(f"Starting with {X.shape[1]} features")
print(f"Dataset shape: {X.shape}")
print(f"\nFeatures: {list(X.columns)}\n")

# ============================================
# 2. METHOD 1: CORRELATION ANALYSIS
# ============================================
print("="*80)
print("METHOD 1: CORRELATION ANALYSIS - REMOVE HIGHLY CORRELATED FEATURES")
print("="*80)

# Calculate correlation matrix
correlation_matrix = X.corr()

# Find pairs of highly correlated features (r > 0.9)
high_corr_pairs = []
features_to_remove = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            feature1 = correlation_matrix.columns[i]
            feature2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            high_corr_pairs.append((feature1, feature2, corr_value))
            
            # Keep the feature with higher correlation to target
            corr_with_target_1 = abs(X[feature1].corr(y))
            corr_with_target_2 = abs(X[feature2].corr(y))
            
            if corr_with_target_1 >= corr_with_target_2:
                features_to_remove.add(feature2)
            else:
                features_to_remove.add(feature1)

print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (|r| > 0.9):")
for feat1, feat2, corr in high_corr_pairs:
    print(f"  • {feat1} <-> {feat2}: r = {corr:.3f}")

print(f"\nFeatures to remove due to multicollinearity: {features_to_remove}")

# Remove highly correlated features
X_no_corr = X.drop(columns=list(features_to_remove))
print(f"\nAfter removing correlated features: {X_no_corr.shape[1]} features remaining")

# Visualize correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            xticklabels=True, yticklabels=True)
plt.title('Feature Correlation Matrix (Before Selection)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('../outputs/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 3. METHOD 2: ANOVA F-TEST (STATISTICAL SIGNIFICANCE)
# ============================================
print("\n" + "="*80)
print("METHOD 2: ANOVA F-TEST - STATISTICAL SIGNIFICANCE")
print("="*80)

# Perform ANOVA F-test for continuous features
f_scores, p_values = f_classif(X_no_corr, y)

# Create dataframe with results
anova_results = pd.DataFrame({
    'feature': X_no_corr.columns,
    'f_score': f_scores,
    'p_value': p_values
}).sort_values('f_score', ascending=False)

print("\nTop 20 features by F-score:")
print(anova_results.head(20).to_string(index=False))

# Keep features with p < 0.05 (statistically significant)
significant_features = anova_results[anova_results['p_value'] < 0.05]['feature'].tolist()
print(f"\n{len(significant_features)} features are statistically significant (p < 0.05)")

# Visualize F-scores
plt.figure(figsize=(12, 8))
top_20 = anova_results.head(20)
plt.barh(top_20['feature'], top_20['f_score'], color='steelblue')
plt.xlabel('F-Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Top 20 Features by ANOVA F-Score', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../outputs/figures/anova_f_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 4. METHOD 3: MUTUAL INFORMATION
# ============================================
print("\n" + "="*80)
print("METHOD 3: MUTUAL INFORMATION - NON-LINEAR RELATIONSHIPS")
print("="*80)

# Calculate mutual information scores
mi_scores = mutual_info_classif(X_no_corr, y, random_state=42)

mi_results = pd.DataFrame({
    'feature': X_no_corr.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop 20 features by Mutual Information:")
print(mi_results.head(20).to_string(index=False))

# Visualize MI scores
plt.figure(figsize=(12, 8))
top_20_mi = mi_results.head(20)
plt.barh(top_20_mi['feature'], top_20_mi['mi_score'], color='darkgreen')
plt.xlabel('Mutual Information Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Top 20 Features by Mutual Information', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../outputs/figures/mutual_information_scores.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 5. METHOD 4: RANDOM FOREST FEATURE IMPORTANCE
# ============================================
print("\n" + "="*80)
print("METHOD 4: RANDOM FOREST FEATURE IMPORTANCE - EMBEDDED METHOD")
print("="*80)

# Train Random Forest to get feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_no_corr, y)

rf_importance = pd.DataFrame({
    'feature': X_no_corr.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features by Random Forest Importance:")
print(rf_importance.head(20).to_string(index=False))

# Visualize RF importance
plt.figure(figsize=(12, 8))
top_20_rf = rf_importance.head(20)
plt.barh(top_20_rf['feature'], top_20_rf['importance'], color='forestgreen')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Top 20 Features by Random Forest Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../outputs/figures/rf_feature_importance_selection.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 6. METHOD 5: RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================
print("\n" + "="*80)
print("METHOD 5: RECURSIVE FEATURE ELIMINATION (RFE) - WRAPPER METHOD")
print("="*80)

# Use Logistic Regression as estimator for RFE
lr = LogisticRegression(max_iter=1000, random_state=42)

# Scale features for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_no_corr)

# Perform RFE to select top 25 features
rfe = RFE(estimator=lr, n_features_to_select=25, step=1)
rfe.fit(X_scaled, y)

# Get selected features
rfe_selected = X_no_corr.columns[rfe.support_].tolist()
rfe_ranking = pd.DataFrame({
    'feature': X_no_corr.columns,
    'ranking': rfe.ranking_,
    'selected': rfe.support_
}).sort_values('ranking')

print(f"\nRFE selected {len(rfe_selected)} features:")
print(rfe_ranking[rfe_ranking['selected']].to_string(index=False))

# ============================================
# 7. DOMAIN KNOWLEDGE VALIDATION
# ============================================
print("\n" + "="*80)
print("METHOD 6: DOMAIN KNOWLEDGE - MEDICAL LITERATURE")
print("="*80)

# Features known from medical literature to be important for smoking detection
medical_literature_features = [
    'hemoglobin',           # Elevated in smokers (polycythemia)
    'gamma_GT',             # Liver enzyme affected by smoking
    'HDL_chole',            # Reduced in smokers
    'total_hdl_ratio',      # Cardiovascular risk indicator
    'ast_alt_ratio',        # Liver damage pattern
    'pulse_pressure',       # Arterial stiffness
    'smoking_risk_score',   # Composite biomarker
    'BMI',                  # Body composition
    'age',                  # Strong demographic predictor
    'triglyceride'          # Lipid metabolism
]

# Filter to features that exist in our dataset
medical_features_available = [f for f in medical_literature_features if f in X_no_corr.columns]

print(f"\nMedically important features (from literature):")
for feat in medical_features_available:
    print(f"  • {feat}")

# ============================================
# 8. COMBINE SELECTION METHODS
# ============================================
print("\n" + "="*80)
print("COMBINING ALL SELECTION METHODS")
print("="*80)

# Get top N features from each method
n_top = 25

top_anova = set(anova_results.head(n_top)['feature'].tolist())
top_mi = set(mi_results.head(n_top)['feature'].tolist())
top_rf = set(rf_importance.head(n_top)['feature'].tolist())
top_rfe = set(rfe_selected)
top_medical = set(medical_features_available)

# Features selected by at least 3 out of 5 methods
all_features = X_no_corr.columns.tolist()
selection_scores = {}

for feature in all_features:
    score = 0
    methods_selected = []
    
    if feature in top_anova:
        score += 1
        methods_selected.append('ANOVA')
    if feature in top_mi:
        score += 1
        methods_selected.append('MI')
    if feature in top_rf:
        score += 1
        methods_selected.append('RF')
    if feature in top_rfe:
        score += 1
        methods_selected.append('RFE')
    if feature in top_medical:
        score += 1
        methods_selected.append('Medical')
    
    selection_scores[feature] = {
        'score': score,
        'methods': methods_selected
    }

# Create dataframe
selection_df = pd.DataFrame([
    {'feature': feat, 'selection_score': data['score'], 
     'methods': ', '.join(data['methods'])}
    for feat, data in selection_scores.items()
]).sort_values('selection_score', ascending=False)

print("\nFeature Selection Summary (sorted by number of methods):")
print(selection_df.to_string(index=False))

# Select features with score >= 3 (selected by at least 3 methods)
final_selected_features = selection_df[selection_df['selection_score'] >= 3]['feature'].tolist()

# Force-include critical medical features even if score < 3
critical_medical = ['hemoglobin', 'gamma_GT', 'HDL_chole', 'total_hdl_ratio']
for feat in critical_medical:
    if feat in X_no_corr.columns and feat not in final_selected_features:
        final_selected_features.append(feat)
        print(f"\n⚠️ Force-included medical feature: {feat}")

print(f"\n{'='*80}")
print(f"FINAL SELECTED FEATURES: {len(final_selected_features)} features")
print(f"{'='*80}")
for i, feat in enumerate(sorted(final_selected_features), 1):
    score = selection_scores[feat]['score']
    methods = ', '.join(selection_scores[feat]['methods'])
    print(f"{i:2d}. {feat:30s} (Score: {score}, Methods: {methods})")

# ============================================
# 9. VISUALIZE FINAL SELECTION
# ============================================

# Compare: All methods agreement
fig, ax = plt.subplots(figsize=(14, 10))

selection_plot_df = selection_df.sort_values('selection_score', ascending=True)
colors = ['red' if score < 3 else 'orange' if score == 3 else 'green' 
          for score in selection_plot_df['selection_score']]

ax.barh(selection_plot_df['feature'], selection_plot_df['selection_score'], color=colors)
ax.axvline(x=3, color='black', linestyle='--', linewidth=2, label='Selection Threshold (≥3)')
ax.set_xlabel('Number of Methods Selecting This Feature', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Feature Selection Consensus Across Methods', fontsize=14, fontweight='bold')
ax.set_xlim([0, 6])
ax.legend()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/figures/feature_selection_consensus.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================
# 10. SAVE RESULTS
# ============================================

# Save all selection results
selection_df.to_csv('../outputs/results/feature_selection_scores.csv', index=False)
anova_results.to_csv('../outputs/results/anova_scores.csv', index=False)
mi_results.to_csv('../outputs/results/mutual_information_scores.csv', index=False)
rf_importance.to_csv('../outputs/results/rf_importance_scores.csv', index=False)

# Save final selected dataset
X_selected = X[final_selected_features]
y_selected = y

final_data = X_selected.copy()
final_data['SMK_stat'] = y_selected

final_data.to_csv('../data/preprocessed_data.csv', index=False)

print(f"\n✅ Feature selection complete!")
print(f"📊 Selection scores saved to: ../outputs/results/feature_selection_scores.csv")
print(f"📊 Final dataset saved to: ../data/preprocessed_data.csv")
print(f"📈 Visualizations saved to: ../outputs/figures/")
print(f"\n{'='*80}")
print(f"SUMMARY: {X.shape[1]} → {len(final_selected_features)} features")
print(f"Reduction: {(1 - len(final_selected_features)/X.shape[1])*100:.1f}%")
print(f"{'='*80}")