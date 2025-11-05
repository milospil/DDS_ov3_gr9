# extract_features.py - CORRECTED VERSION

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================
# 1. LOAD DATA
# ============================================
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("datasets/smoking_drinking_filtered.csv")

print(f"Original dataset shape: {df.shape}")

# -------------------------
# First split: 60 / 40
# -------------------------
fe_df, holdout_60_df = train_test_split(
    df,
    test_size=0.60,  # 60% held untouched for later training
    random_state=42,
    shuffle=True,
)

print(f"Feature-engineering pool (40%): {fe_df.shape}")
print(f"Held-out final data (60%):     {holdout_60_df.shape}")

# -------------------------
# Second split on the 40%: 80 / 20
# -------------------------
fe_train_df, fe_test_df = train_test_split(
    fe_df,
    test_size=0.20,  # 20% of 40% = 8% of original data
    random_state=42,
    shuffle=True,
)

print(f"FE Train set (80% of 40%): {fe_train_df.shape}")
print(f"FE Test set (20% of 40%):  {fe_test_df.shape}")
print("\nUsing ONLY fe_train_df for feature extraction.\n")

# This df is the one you now engineer features on
df = fe_train_df.copy()
# ============================================
# 2. FEATURE EXTRACTION (12 NEW FEATURES)
# ============================================

print("Extracting features...\n")


# 1. BMI - Body Mass Index
df["BMI"] = (df["weight"] / ((df["height"] / 100) ** 2)).round(3)
print("✓ Created BMI")


# 2. Total to HDL Cholesterol Ratio
df["total_hdl_ratio"] = (df["tot_chole"] / df["HDL_chole"]).round(3)
print("✓ Created total_hdl_ratio")

# 3. Pulse Pressure
df["pulse_pressure"] = (df["SBP"] - df["DBP"]).round(3)
print("✓ Created pulse_pressure")


# 4. AST to ALT Ratio
df["ast_alt_ratio"] = (df["SGOT_AST"] / df["SGOT_ALT"]).round(3)
print("✓ Created ast_alt_ratio")


# 5. Atherogenic Index
df["atherogenic_index"] = ((df["tot_chole"] - df["HDL_chole"]) / df["HDL_chole"]).round(
    3
)
print("✓ Created atherogenic_index")

# 6. Waist-to-Height Ratio
df["waist_height_ratio"] = (df["waistline"] / df["height"]).round(3)
print("✓ Created waist_height_ratio")

# 7. Mean Arterial Pressure
df["MAP"] = (df["DBP"] + (df["pulse_pressure"] / 3)).round(3)
print("✓ Created MAP")


# 8. LDL to HDL Ratio
df["ldl_hdl_ratio"] = (df["LDL_chole"] / df["HDL_chole"]).round(3)
print("✓ Created ldl_hdl_ratio")


# 9. Smoking Risk Score (FIXED - using gamma_GTP)
def smoking_risk_score(row):
    score = 0
    if row["hemoglobin"] > 16:
        score += 2
    if row["gamma_GTP"] > 60:
        score += 2  # FIXED: gamma_GTP not gamma_GT
    if row["HDL_chole"] < 40:
        score += 1
    if row["triglyceride"] > 150:
        score += 1
    return score


df["smoking_risk_score"] = df.apply(smoking_risk_score, axis=1)
print("✓ Created smoking_risk_score")

# 10. Metabolic Syndrome Count
df["metabolic_syndrome_count"] = (
    (df["waistline"] > 90).astype(int)
    + (df["triglyceride"] > 150).astype(int)
    + (df["HDL_chole"] < 40).astype(int)
    + (df["SBP"] >= 130).astype(int)
    + (df["BLDS"] > 100).astype(int)
)
print("Created metabolic_syndrome_count")

# 11. Age Groups
df["age_group"] = pd.cut(
    df["age"], bins=[0, 30, 50, 70, 110], labels=[0, 1, 2, 3]
).astype(int)
print("Created age_group")

# 12. Elevated Hemoglobin Indicator
# Assuming sex: 1=male, 0=female (adjust if different)
df["elevated_hemoglobin"] = (
    ((df["sex"] == 1) & (df["hemoglobin"] > 17))
    | ((df["sex"] == 0) & (df["hemoglobin"] > 15))
).astype(int)
print("Created elevated_hemoglobin")

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
    "BMI",
    "total_hdl_ratio",
    "pulse_pressure",
    "ast_alt_ratio",
    "atherogenic_index",
    "waist_height_ratio",
    "MAP",
    "ldl_hdl_ratio",
    "smoking_risk_score",
    "metabolic_syndrome_count",
    "age_group",
    "elevated_hemoglobin",
]
for i, feat in enumerate(new_features, 1):
    print(f"  {i:2d}. {feat}")

# ============================================
# 4. SAVE
# ============================================
df.to_csv("datasets/feature_extracted_data.csv", index=False)
print(f"\nData saved to: datasets/feature_extracted_data.csv")
print(f"{'='*60}\n")


# Show sample of new features
print("Sample of extracted features:")
print(df[["BMI", "total_hdl_ratio", "smoking_risk_score", "age_group"]].head())

# At the end of extract_features.py
print("\nVerifying age_group column:")
print(f"  Data type: {df['age_group'].dtype}")
print(f"  Unique values: {sorted(df['age_group'].unique())}")
print(f"  Value counts:\n{df['age_group'].value_counts().sort_index()}")
