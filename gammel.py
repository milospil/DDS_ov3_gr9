import pandas as pd
import numpy as np

smoking_df = pd.read_csv("sd_dataset.csv")



print(smoking_df.head())
print(smoking_df.describe())
print(smoking_df.info())
print(smoking_df.dtypes)
print("\nMissing Values:")
print(smoking_df.isnull().sum())