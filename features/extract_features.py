# extract_features.py - CORRECTED VERSION

import pandas as pd
import numpy as np

# ============================================
# 1. LOAD DATA
# ============================================
df = pd.read_csv('smoking_drinking_numeric.csv')  # Adjust your filename

print(f"Original feature count: {len(df.columns)}")
print(f"Dataset shape: {df.shape}\n")

# ============================================
# 2. FEATURE EXTRACTION (12 NEW FEATURES)
# ============================================

print("Extracting features...\n")

# 1. BMI - Body Mass Index
df['BMI'] = df['weight'] / ((df['height']/100) ** 2)
print("✓ Created BMI")

# 2. Total to HDL Cholesterol Ratio
df['total_hdl_ratio'] = df['tot_chole'] / df['HDL_chole']
print("✓ Created total_hdl_ratio")

# 3. Pulse Pressure
df['pulse_pressure'] = df['SBP'] - df['DBP']
print("✓ Created pulse_pressure")

# 4. AST to ALT Ratio
df['ast_alt_ratio'] = df['SGOT_AST'] / df['SGOT_ALT']
print("✓ Created ast_alt_ratio")

# 5. Atherogenic Index
df['atherogenic_index'] = (df['tot_chole'] - df['HDL_chole']) / df['HDL_chole']
print("✓ Created atherogenic_index")

# 6. Waist-to-Height Ratio
df['waist_height_ratio'] = df['waistline'] / df['height']
print("✓ Created waist_height_ratio")

# 7. Mean Arterial Pressure
df['MAP'] = df['DBP'] + (df['pulse_pressure'] / 3)
print("✓ Created MAP")

# 8. LDL to HDL Ratio
df['ldl_hdl_ratio'] = df['LDL_chole'] / df['HDL_chole']
print("✓ Created ldl_hdl_ratio")

# 9. Smoking Risk Score (FIXED - using gamma_GTP)
def smoking_risk_score(row):
    score = 0
    if row['hemoglobin'] > 16: score += 2
    if row['gamma_GTP'] > 60: score += 2  # ✅ FIXED: gamma_GTP not gamma_GT
    if row['HDL_chole'] < 40: score += 1
    if row['triglyceride'] > 150: score += 1
    return score

df['smoking_risk_score'] = df.apply(smoking_risk_score, axis=1)
print("✓ Created smoking_risk_score")

# 10. Metabolic Syndrome Count
df['metabolic_syndrome_count'] = (
    (df['waistline'] > 90).astype(int) +
    (df['triglyceride'] > 150).astype(int) +
    (df['HDL_chole'] < 40).astype(int) +
    (df['SBP'] >= 130).astype(int) +
    (df['BLDS'] > 100).astype(int)
)
print("✓ Created metabolic_syndrome_count")

# 11. Age Groups
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 30, 40, 50, 60, 100],
                         labels=['young', '30s', '40s', '50s', 'elderly'])
print("✓ Created age_group")

# 12. Elevated Hemoglobin Indicator
# Assuming sex: 1=male, 0=female (adjust if different)
df['elevated_hemoglobin'] = (
    ((df['sex'] == 1) & (df['hemoglobin'] > 17)) |
    ((df['sex'] == 0) & (df['hemoglobin'] > 15))
).astype(int)
print("✓ Created elevated_hemoglobin")

# ============================================
# 3. SUMMARY
# ============================================
print(f"\n{'='*60}")
print(f"FEATURE EXTRACTION COMPLETE")
print(f"{'='*60}")
print(f"Original features: 24")
print(f"New features created: 12")
print(f"Total features: {len(df.columns)}")
print(f"\nNew features added:")
new_features = [
    'BMI', 'total_hdl_ratio', 'pulse_pressure', 'ast_alt_ratio',
    'atherogenic_index', 'waist_height_ratio', 'MAP', 'ldl_hdl_ratio',
    'smoking_risk_score', 'metabolic_syndrome_count', 'age_group',
    'elevated_hemoglobin'
]
for i, feat in enumerate(new_features, 1):
    print(f"  {i:2d}. {feat}")

# ============================================
# 4. SAVE
# ============================================
df.to_csv('feature_extracted_data.csv', index=False)
print(f"\n✅ Data saved to: feature_extracted_data.csv")
print(f"{'='*60}\n")

# Show sample of new features
print("Sample of extracted features:")
print(df[['BMI', 'total_hdl_ratio', 'smoking_risk_score', 'age_group']].head())