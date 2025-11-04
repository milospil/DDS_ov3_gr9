import pandas as pd
from sklearn.preprocessing import StandardScaler

# === 1. Les inn renset CSV ===
df = pd.read_csv("smoking_drinking_numeric.csv")

# === 2. Velg hvilke kolonner som skal normaliseres ===
features_to_scale = [
    'age', 'SBP', 'DBP', 'BLDS', 'tot_chole', 'HDL_chole', 'LDL_chole',
    'triglyceride', 'hemoglobin', 'SGOT_AST', 'SGOT_ALT', 'gamma_GTP'
]

# Målvariabler (ikke normaliser disse)
target_cols = ['SMK_stat_type_cd', 'drink']

# === 3. Lag scaler og tilpass på data ===
scaler = StandardScaler()

# Fit-transformer på feature-kolonnene
scaled_features = scaler.fit_transform(df[features_to_scale])

# === 4. Lag ny DataFrame med skalerte verdier ===
df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale)

# === 5. Legg til målvariabler igjen ===
df_scaled[target_cols] = df[target_cols].values

# === 6. Lagre som ny fil ===
df_scaled.to_csv("smoking_drinking_scaled.csv", index=False)

print("✅ Ferdig! Filen 'smoking_drinking_scaled.csv' er normalisert og klar for modelltrening.")