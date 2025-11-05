# 5_feature_selection.py - TASK 4: FEATURE SELECTION

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
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directories if they don't exist
os.makedirs('outputs/figures', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

# ============================================
# 1. LOAD DATA AFTER FEATURE EXTRACTION
# ============================================
print("Loading data after feature extraction...")
df = pd.read_csv('datasets/feature_extracted_data.csv')

# Separate features and target
X = df.drop(['SMK_stat_type_cd', 'drink'], axis=1)  # Drop target and drink
y = df['SMK_stat_type_cd']

print(f"Starting with {X.shape[1]} features")
print(f"Dataset shape: {X.shape}")
print(f"\nTarget distribution:")
print(y.value_counts())

# ============================================
# 2. ENCODE CATEGORICAL FEATURES
# ============================================
print("\n" + "="*80)
print("ENCODING CATEGORICAL FEATURES")
print("="*80)

# Check for categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# IMPORTANT: age_group is already numeric (0, 1, 2, 3) - don't encode it!
if 'age_group' in categorical_cols:
    categorical_cols.remove('age_group')
    print("ℹ️  'age_group' is already numeric - skipping encoding")

if categorical_cols:
    print(f"Found {len(categorical_cols)} categorical column(s) to encode: {categorical_cols}")
    
    for col in categorical_cols:
        print(f"\nOne-hot encoding '{col}'...")
        col_dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
        print(f"  Created {len(col_dummies.columns)} dummy variables:")
        for dummy_col in col_dummies.columns:
            print(f"    • {dummy_col}")
        
        # Drop original and add dummies
        X = X.drop(col, axis=1)
        X = pd.concat([X, col_dummies], axis=1)
else:
    print("No categorical columns found - all features are already numeric")

print(f"\n✓ After encoding: {X.shape[1]} features")

# Verify all numeric
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"\nWARNING: Non-numeric columns remain: {non_numeric}")
else:
    print("✓ All features verified as numeric")
    
# Verify age_group is present
if 'age_group' in X.columns:
    print(f"✓ age_group present with values: {sorted(X['age_group'].unique())}\n")
else:
    print("⚠️  WARNING: age_group not found in features!\n")


if 'age' in X.columns:
    print("\nRationale:")
    print("  • age_group is a categorical binned version of age")
    print("  • age_group captures non-linear age effects better for smoking prediction")
    print("  • Keeping both would introduce multicollinearity")
    print("  • Domain knowledge suggests age categories are more meaningful than continuous age")
    
    X = X.drop('age', axis=1)
    print(f"\n✓ Dropped 'age', keeping 'age_group'")
    print(f"✓ Features remaining: {X.shape[1]}")
else:
    print("\n'age' column not found - skipping")

    
# ============================================
# 3. METHOD 1: CORRELATION ANALYSIS
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

if high_corr_pairs:
    print(f"\nFound {len(high_corr_pairs)} highly correlated pairs (|r| > 0.9):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  • {feat1} <-> {feat2}: r = {corr:.3f}")
    
    print(f"\nFeatures to remove due to multicollinearity ({len(features_to_remove)}):")
    for feat in sorted(features_to_remove):
        print(f"  • {feat}")
    
    # Remove highly correlated features
    X_no_corr = X.drop(columns=list(features_to_remove))
    print(f"\n✓ After removing correlated features: {X_no_corr.shape[1]} features remaining")
else:
    print("\nNo highly correlated pairs found (|r| > 0.9)")
    X_no_corr = X.copy()

# Visualize correlation matrix
plt.figure(figsize=(18, 16))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot=False, fmt='.2f')
plt.title('Feature Correlation Matrix (Before Selection)', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('outputs/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved correlation matrix heatmap\n")

# ============================================
# 4. METHOD 2: ANOVA F-TEST
# ============================================
print("="*80)
print("METHOD 2: ANOVA F-TEST - STATISTICAL SIGNIFICANCE")
print("="*80)

# Perform ANOVA F-test
f_scores, p_values = f_classif(X_no_corr, y)

# Create dataframe with results
anova_results = pd.DataFrame({
    'feature': X_no_corr.columns,
    'f_score': f_scores,
    'p_value': p_values
}).sort_values('f_score', ascending=False)

print("\nTop 20 features by F-score:")
print(anova_results.head(20).to_string(index=False))

# Keep features with p < 0.05
significant_features = anova_results[anova_results['p_value'] < 0.05]['feature'].tolist()
print(f"\n✓ {len(significant_features)} features are statistically significant (p < 0.05)")

# Features with p >= 0.05 (not significant)
non_significant = anova_results[anova_results['p_value'] >= 0.05]['feature'].tolist()
if non_significant:
    print(f"\nFeatures NOT significant (p >= 0.05): {len(non_significant)}")
    for feat in non_significant:
        p_val = anova_results[anova_results['feature'] == feat]['p_value'].values[0]
        print(f"  • {feat} (p = {p_val:.4f})")

# Visualize F-scores
plt.figure(figsize=(12, 10))
top_20 = anova_results.head(20)
colors = ['green' if p < 0.05 else 'red' for p in top_20['p_value']]
plt.barh(range(len(top_20)), top_20['f_score'], color=colors)
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('F-Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Top 20 Features by ANOVA F-Score', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.legend(['p < 0.05 (Significant)', 'p >= 0.05 (Not Significant)'], loc='lower right')
plt.tight_layout()
plt.savefig('outputs/figures/anova_f_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved ANOVA F-scores plot\n")

# ============================================
# 5. METHOD 3: MUTUAL INFORMATION
# ============================================
print("="*80)
print("METHOD 3: MUTUAL INFORMATION - NON-LINEAR RELATIONSHIPS")
print("="*80)

# Calculate mutual information
mi_scores = mutual_info_classif(X_no_corr, y, random_state=42)

mi_results = pd.DataFrame({
    'feature': X_no_corr.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop 20 features by Mutual Information:")
print(mi_results.head(20).to_string(index=False))

# Visualize MI scores
plt.figure(figsize=(12, 10))
top_20_mi = mi_results.head(20)
plt.barh(range(len(top_20_mi)), top_20_mi['mi_score'], color='darkgreen')
plt.yticks(range(len(top_20_mi)), top_20_mi['feature'])
plt.xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Top 20 Features by Mutual Information', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/figures/mutual_information_scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved Mutual Information plot\n")

# ============================================
# 6. METHOD 4: RANDOM FOREST FEATURE IMPORTANCE
# ============================================
print("="*80)
print("METHOD 4: RANDOM FOREST FEATURE IMPORTANCE - EMBEDDED METHOD")
print("="*80)

# Train Random Forest
print("Training Random Forest (this may take a moment)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
rf.fit(X_no_corr, y)

rf_importance = pd.DataFrame({
    'feature': X_no_corr.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features by Random Forest Importance:")
print(rf_importance.head(20).to_string(index=False))

# Visualize RF importance
plt.figure(figsize=(12, 10))
top_20_rf = rf_importance.head(20)
plt.barh(range(len(top_20_rf)), top_20_rf['importance'], color='forestgreen')
plt.yticks(range(len(top_20_rf)), top_20_rf['feature'])
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Top 20 Features by Random Forest Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/figures/rf_feature_importance_selection.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved Random Forest importance plot\n")

# ============================================
# 7. METHOD 5: RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================
print("="*80)
print("METHOD 5: RECURSIVE FEATURE ELIMINATION (RFE) - WRAPPER METHOD")
print("="*80)

# Scale features for RFE with Logistic Regression
print("Scaling features for RFE...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_no_corr)

# Perform RFE
print("Performing RFE (this may take a moment)...")
lr = LogisticRegression(max_iter=1000, random_state=42)
rfe = RFE(estimator=lr, n_features_to_select=25, step=1)
rfe.fit(X_scaled, y)

# Get selected features
rfe_selected = X_no_corr.columns[rfe.support_].tolist()
rfe_ranking = pd.DataFrame({
    'feature': X_no_corr.columns,
    'ranking': rfe.ranking_,
    'selected': rfe.support_
}).sort_values('ranking')

print(f"\n✓ RFE selected {len(rfe_selected)} features:")
print(rfe_ranking[rfe_ranking['selected']]['feature'].tolist())

print("\nFeatures NOT selected by RFE:")
not_selected = rfe_ranking[~rfe_ranking['selected']]
for _, row in not_selected.head(10).iterrows():
    print(f"  • {row['feature']} (rank: {row['ranking']})")

# ============================================
# 8. DOMAIN KNOWLEDGE VALIDATION
# ============================================
print("\n" + "="*80)
print("METHOD 6: DOMAIN KNOWLEDGE - MEDICAL LITERATURE")
print("="*80)

# Features known from medical literature
medical_literature_features = [
    'hemoglobin', 'gamma_GTP', 'HDL_chole', 'total_hdl_ratio',
    'ast_alt_ratio', 'pulse_pressure', 'smoking_risk_score',
    'BMI', 'age', 'triglyceride', 'SGOT_AST', 'SGOT_ALT'
]

medical_features_available = [f for f in medical_literature_features if f in X_no_corr.columns]

print(f"\nMedically important features (from literature): {len(medical_features_available)}")
for feat in medical_features_available:
    print(f"  • {feat}")

# ============================================
# 9. COMBINE SELECTION METHODS
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

# Calculate consensus scores
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

print("\nFeature Selection Consensus (Top 30):")
print(selection_df.head(30).to_string(index=False))

# ============================================
# 10. SELECT FINAL FEATURES
# ============================================
print("\n" + "="*80)
print("FINAL FEATURE SELECTION")
print("="*80)

# Select features with score >= 3 (selected by at least 3 methods)
final_selected_features = selection_df[selection_df['selection_score'] >= 3]['feature'].tolist()

# Force-include critical medical features even if score < 3
critical_medical = ['hemoglobin', 'gamma_GTP', 'HDL_chole', 'total_hdl_ratio']
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
    print(f"{i:2d}. {feat:30s} (Score: {score}/5, Methods: {methods})")
# ============================================
# 10.5. DOMAIN KNOWLEDGE OVERRIDE - REMOVE CONFOUNDED FEATURES
# ============================================
print("\n" + "="*80)
print("DOMAIN KNOWLEDGE OVERRIDE - CONFOUNDING ANALYSIS")
print("="*80)

# Analyze sex-height-weight relationship
print("\nAnalyzing potential confounding by sex:")
sex_fscore = anova_results[anova_results['feature']=='sex']['f_score'].values[0]
height_fscore = anova_results[anova_results['feature']=='height']['f_score'].values[0]
weight_fscore = anova_results[anova_results['feature']=='weight']['f_score'].values[0]
bmi_fscore = anova_results[anova_results['feature']=='BMI']['f_score'].values[0]

print(f"  sex F-score:    {sex_fscore:,.0f} (1st rank)")
print(f"  height F-score: {height_fscore:,.0f} (2nd rank)")
print(f"  weight F-score: {weight_fscore:,.0f} (4th rank)")
print(f"  BMI F-score:    {bmi_fscore:,.0f} (13th rank)")

print("\nRationale for removal:")
print("  1. Males are significantly taller and heavier than females (biological sex difference)")
print("  2. Males have substantially higher smoking rates (behavioral sex difference)")
print("  3. Height and weight serve as proxy variables for sex rather than independent predictors")
print("  4. Sex dominates all methods (2-3x stronger than height/weight)")
print("  5. BMI captures the meaningful body composition relationship normalized for height")

# Remove confounded features
confounded_features = ['height', 'weight']

print("\nRemoving confounded features:")
for feat in confounded_features:
    if feat in final_selected_features:
        score = selection_scores[feat]['score']
        methods = ', '.join(selection_scores[feat]['methods'])
        final_selected_features.remove(feat)
        print(f"  ✓ Removed '{feat}' (Score: {score}/5, Methods: {methods})")
        print(f"    Information captured by: sex + BMI")

print(f"\n{'='*80}")
print(f"UPDATED FINAL FEATURE COUNT: {len(final_selected_features)} features")
print(f"{'='*80}")

# Re-print updated final selected features
for i, feat in enumerate(sorted(final_selected_features), 1):
    score = selection_scores[feat]['score']
    methods = ', '.join(selection_scores[feat]['methods'])
    print(f"{i:2d}. {feat:30s} (Score: {score}/5, Methods: {methods})")
# ============================================
# 11. VISUALIZE CONSENSUS
# ============================================
print("\nCreating consensus visualization...")

fig, ax = plt.subplots(figsize=(14, 12))

selection_plot_df = selection_df.sort_values('selection_score', ascending=True)
colors = ['red' if score < 3 else 'orange' if score == 3 else 'yellowgreen' if score == 4 else 'darkgreen' 
          for score in selection_plot_df['selection_score']]

bars = ax.barh(range(len(selection_plot_df)), selection_plot_df['selection_score'], color=colors)
ax.set_yticks(range(len(selection_plot_df)))
ax.set_yticklabels(selection_plot_df['feature'], fontsize=8)
ax.axvline(x=3, color='black', linestyle='--', linewidth=2, label='Selection Threshold (≥3)')
ax.set_xlabel('Number of Methods Selecting This Feature', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('Feature Selection Consensus Across Methods', fontsize=14, fontweight='bold')
ax.set_xlim([0, 6])
ax.legend(['Threshold (≥3 methods)'], loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/feature_selection_consensus.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved consensus visualization\n")

# ============================================
# 12. SAVE RESULTS
# ============================================
print("="*80)
print("SAVING RESULTS")
print("="*80)

# Save all selection results
selection_df.to_csv('outputs/results/feature_selection_scores.csv', index=False)
print("✓ Saved feature_selection_scores.csv")

anova_results.to_csv('outputs/results/anova_scores.csv', index=False)
print("✓ Saved anova_scores.csv")

mi_results.to_csv('outputs/results/mutual_information_scores.csv', index=False)
print("✓ Saved mutual_information_scores.csv")

rf_importance.to_csv('outputs/results/rf_importance_scores.csv', index=False)
print("✓ Saved rf_importance_scores.csv")

rfe_ranking.to_csv('outputs/results/rfe_ranking.csv', index=False)
print("✓ Saved rfe_ranking.csv")

# Save final selected dataset
X_selected = X[final_selected_features]
final_data = X_selected.copy()
final_data['SMK_stat_type_cd'] = y

final_data.to_csv('datasets/preprocessed_data.csv', index=False)
print("✓ Saved preprocessed_data.csv (with selected features only)")

# ============================================
# 13. SUMMARY
# ============================================
print("\n" + "="*80)
print("FEATURE SELECTION COMPLETE!")
print("="*80)
print(f"Original features: {X.shape[1]}")
print(f"After correlation removal: {X_no_corr.shape[1]}")
print(f"Final selected features: {len(final_selected_features)}")
print(f"Reduction: {(1 - len(final_selected_features)/X.shape[1])*100:.1f}%")
print(f"\nFiles saved:")
print(f"  • preprocessed_data.csv (ready for modeling)")
print(f"  • outputs/results/*.csv (selection details)")
print(f"  • outputs/figures/*.png (visualizations)")
print("="*80)

