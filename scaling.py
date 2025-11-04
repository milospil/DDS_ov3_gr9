import pandas as pd
from sklearn.preprocessing import StandardScaler

# === 1) Les inn renset CSV ===
df = pd.read_csv("smoking_drinking_filtered.csv")

# === 2) Definer hva som skal/ikke skal skaleres ===
# - Ikke skaler: binære/ordinære kategorier og target
do_not_scale = {
    'sex',                   # binær (0/1)
    'urine_protein',         # ordinal kode (1–6)
    'SMK_stat_type_cd',      # target (1=never, 2=quit, 3=current)
    'drink'                  # (kan fortsatt finnes i filen din)
}

# Kontinuerlige features i datasettet ditt nå:
continuous_candidates = [
    'age', 'waistline',
    'SBP', 'DBP',
    'BLDS',
    'tot_chole', 'HDL_chole', 'LDL_chole', 'triglyceride',
    'hemoglobin',
    'serum_creatinine',
    'SGOT_AST', 'SGOT_ALT', 'gamma_GTP',
    'BMI', 'BP_ratio', 'HDL_LDL_ratio'
]

# Ta bare de som faktisk finnes i filen:
features_to_scale = [c for c in continuous_candidates if c in df.columns and c not in do_not_scale]

# === 3) Standardiser kontinuerlige features (fit på hele datasettet her for enkelhets skyld) ===
scaler = StandardScaler()
scaled = scaler.fit_transform(df[features_to_scale])

# === 4) Bygg utdata: erstatt de skalerte kolonnene, behold resten urørt ===
df_out = df.copy()
df_out.loc[:, features_to_scale] = scaled

# === 5) Avrund skalerte kolonner til 3 desimaler for lesbarhet ===
df_out[features_to_scale] = df_out[features_to_scale].round(3)

# (Valgfritt) Sørg for heltall i tydelig kategoriske kolonner
for col in ['sex', 'urine_protein', 'SMK_stat_type_cd', 'drink']:
    if col in df_out.columns:
        # Behold ints der det gir mening (unngå float-serialisering som 1.0)
        try:
            df_out[col] = df_out[col].astype('Int64')
        except Exception:
            pass  # hopp over hvis kolonnen ikke lar seg caste (f.eks. mangler verdier)

# === 6) Lagre som ny fil ===
df_out.to_csv("smoking_drinking_scaled.csv", index=False)

print("✅ Skalert kolonner:", features_to_scale)
print("📁 Lagret: 'smoking_drinking_scaled.csv' (skalerte verdier avrundet til 3 desimaler)")
