import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Les inn data ===
df = pd.read_csv("datasets/smoking_drinking_scaled.csv")

# === 2. Definer target ===
target = 'SMK_stat_type_cd'

# === 3. Beregn korrelasjon mot alle numeriske features ===
corr_series = df.corr(numeric_only=True)[target].sort_values(ascending=False)

# === 4. Konverter til DataFrame for enklere utskrift og lagring ===
corr_df = corr_series.reset_index()
corr_df.columns = ['Feature', 'Correlation_with_SmokingStatus']

# === 5. Print topp 10 features (kan justeres) ===
print("📊 Top 10 features most correlated with smoking status (1=Never, 2=Former, 3=Current):\n")
print(corr_df.head(10).to_string(index=False))

# === 6. (Valgfritt) Lagre hele korrelasjonslisten til CSV for rapportbruk ===
corr_df.to_csv("datasets/correlation_with_smoking_status.csv", index=False)
print("\nFull correlation list saved as 'datasets/correlation_with_smoking_status.csv'")

# === 7. Plot korrelasjoner ===
plt.figure(figsize=(8,6))
sns.barplot(
    x='Correlation_with_SmokingStatus',
    y='Feature',
    hue='Feature',           # ← NY linje: trengs for å bruke fargepalett
    data=corr_df,
    palette="coolwarm",
    legend=False             # ← skjuler unødvendig fargeforklaring
)
plt.title("Correlation of Features with Smoking Status (1=Never, 2=Former, 3=Current)")
plt.xlabel("Pearson Correlation")
plt.tight_layout()
plt.show()