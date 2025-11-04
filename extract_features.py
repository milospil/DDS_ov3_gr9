import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('smoking_drinking_dataset.csv')

print(f"Original feature count: {len(df.columns)}")

# ============================================
# FEATURE EXTRACTION (12 NEW FEATURES)
# ============================================

# 1. BMI - Body Mass Index
df['BMI'] = df['weight'] / ((df['height']/100) ** 2)

# 2. Total to HDL Cholesterol Ratio
df['total_hdl_ratio'] = df['tot_chole'] / df['HDL_chole']

# 3. Pulse Pressure
df['pulse_pressure'] = df['SBP'] - df['DBP']

# 4. AST to ALT Ratio
df['ast_alt_ratio'] = df['SGOT_AST'] / df['SGOT_ALT']

# 5. Atherogenic Index
df['atherogenic_index'] = (df['tot_chole'] - df['HDL_chole']) / df['HDL_chole']

# 6. Waist-to-Height Ratio
df['waist_height_ratio'] = df['waistline'] / df['height']

# 7. Mean Arterial Pressure
df['MAP'] = df['DBP'] + (df['pulse_pressure'] / 3)

# 8. LDL to HDL Ratio
df['ldl_hdl_ratio'] = df['LDL_chole'] / df['HDL_chole']

# 9. Smoking Risk Score
def smoking_risk_score(row):
    score = 0
    if row['hemoglobin'] > 16: score += 2
    if row['gamma_GT'] > 60: score += 2
    if row['HDL_chole'] < 40: score += 1
    if row['triglyceride'] > 150: score += 1
    return score

df['smoking_risk_score'] = df.apply(smoking_risk_score, axis=1)

# 10. Metabolic Syndrome Count
df['metabolic_syndrome_count'] = (
    (df['waistline'] > 90).astype(int) +
    (df['triglyceride'] > 150).astype(int) +
    (df['HDL_chole'] < 40).astype(int) +
    (df['SBP'] >= 130).astype(int) +
    (df['BLDS'] > 100).astype(int)
)

# 11. Age Groups
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 30, 40, 50, 60, 100],
                         labels=['young', '30s', '40s', '50s', 'elderly'])

# 12. Elevated Hemoglobin Indicator
df['elevated_hemoglobin'] = (
    ((df['sex'] == 1) & (df['hemoglobin'] > 17)) |
    ((df['sex'] == 0) & (df['hemoglobin'] > 15))
).astype(int)

print(f"After extraction: {len(df.columns)} features")
print(f"New features created: 12")

# Show summary of new features
new_features = ['BMI', 'total_hdl_ratio', 'pulse_pressure', 'ast_alt_ratio',
                'atherogenic_index', 'waist_height_ratio', 'MAP', 'ldl_hdl_ratio',
                'smoking_risk_score', 'metabolic_syndrome_count', 'age_group',
                'elevated_hemoglobin']

print("\nNew features summary:")
for feature in new_features:
    if df[feature].dtype in ['int64', 'float64']:
        print(f"{feature}: mean={df[feature].mean():.2f}, std={df[feature].std():.2f}")
    else:
        print(f"{feature}: {df[feature].value_counts().to_dict()}")