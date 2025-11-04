import pandas as pd
import numpy as np

sd_df = pd.read_csv("sd_dataset.csv")


print("\n")
print("\n Smoking Drinking dataset Overview:")
print(sd_df.head())
print(sd_df.describe())
print(sd_df.info())
print(sd_df.dtypes)
print("\nMissing Values:")
print(sd_df.isnull().sum())



print("\n")
print("\n")
print("\nOutliers:")
print("\n")
# Outlier detection (example for numerical columns)
numerical_cols = sd_df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    Q1 = sd_df[col].quantile(0.25)
    Q3 = sd_df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = sd_df[(sd_df[col] < Q1 - 1.5*IQR) | (sd_df[col] > Q3 + 1.5*IQR)]
    if len(outliers) >= 0:
        print(f"Outliers in {col}: {len(outliers)}")

summary_df = pd.DataFrame({
    'Column': sd_df.columns,
    'Data_Type': sd_df.dtypes,
    'Missing_Values': sd_df.isnull().sum(),
    'Missing_Percentage': (sd_df.isnull().sum() / len(sd_df)) * 100,
    'Unique_Values': [sd_df[col].nunique() for col in sd_df.columns]
})
print("\nData Quality Summary:")
print(summary_df)