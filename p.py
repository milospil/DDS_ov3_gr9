import pandas as pd
import pyreadstat

# Read the .sav file
df, meta = pyreadstat.read_sav("pgph.0003946.s001.csv")

# Check the first rows
print(df.head())

# Save to CSV
df.to_csv("new_file.csv", index=False)

print("Conversion complete: 'new_file.csv'")