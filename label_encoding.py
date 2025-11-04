import pandas as pd

# === 1. Les inn CSV-filen ===
df = pd.read_csv("sd_dataset.csv")

# === 2. Sjekk unike verdier før encoding (for kontroll) ===
print("Unique values in 'sex':", df['sex'].unique())
print("Unique values in 'DRK_YN':", df['DRK_YN'].unique())

# === 3. Label-encoding ===
# Konverter 'sex' til binær verdi: Male=1, Female=0
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# Konverter 'DRK_YN' til binær verdi: Y=1 (drinker), N=0 (non-drinker)
df['DRK_YN'] = df['DRK_YN'].map({'Y': 1, 'N': 0})

# === 4. Gi DRK_YN nytt navn for klarhet ===
df = df.rename(columns={'DRK_YN': 'drink'})

# === 5. Fjern irrelevante kolonner ===
cols_to_drop = ['sight_left', 'sight_right', 'hear_left', 'hear_right']
df = df.drop(columns=cols_to_drop, errors='ignore')

print(f"Fjernet kolonner: {cols_to_drop}")

# === 6. Sjekk at ingen tekstkolonner gjenstår ===
print("Data types after encoding:\n", df.dtypes)

df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['BP_ratio'] = df['SBP'] / df['DBP']
df['HDL_LDL_ratio'] = df['HDL_chole'] / df['LDL_chole']

df[['BMI', 'BP_ratio', 'HDL_LDL_ratio']] = df[['BMI', 'BP_ratio', 'HDL_LDL_ratio']].round(1)

# === 3. Fjern kolonner som ikke lenger trengs ===
cols_to_drop = ['height', 'weight']  # fjern bare de to som er "innebygd" i BMI
df = df.drop(columns=cols_to_drop, errors='ignore')


# === 7. Lagre ny fil uten tekstkolonner ===
df.to_csv("smoking_drinking_numeric.csv", index=False)

print("New CSV file")