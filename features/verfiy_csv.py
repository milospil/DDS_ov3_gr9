import pandas as pd

# Load the saved file to verify
df = pd.read_csv('datasets/feature_extracted_data.csv')

print("="*60)
print("VERIFYING SAVED CSV FILE")
print("="*60)
print(f"\nTotal columns: {len(df.columns)}")
print(f"Dataset shape: {df.shape}")

print("\nAll columns in CSV:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\nChecking age_group column:")
if 'age_group' in df.columns:
    print("  ✓ age_group EXISTS in CSV")
    print(f"  Data type: {df['age_group'].dtype}")
    print(f"  Value counts:\n{df['age_group'].value_counts().sort_index()}")
    print(f"  Sample values: {df['age_group'].head(10).tolist()}")
else:
    print("  ✗ age_group NOT FOUND in CSV")

print("\nNew features present:")
new_features = [
    'BMI', 'total_hdl_ratio', 'pulse_pressure', 'ast_alt_ratio',
    'atherogenic_index', 'waist_height_ratio', 'MAP', 'ldl_hdl_ratio',
    'smoking_risk_score', 'metabolic_syndrome_count', 'age_group',
    'elevated_hemoglobin'
]
for feat in new_features:
    status = "✓" if feat in df.columns else "✗"
    print(f"  {status} {feat}")