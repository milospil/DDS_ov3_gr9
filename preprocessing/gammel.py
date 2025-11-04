import pandas as pd
import numpy as np

sd_df = pd.read_csv("datasets/smoking_drinking_numeric.csv")


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

# Define clinically meaningful ranges
medical_ranges = {
    'SBP': (70, 200),            # Systolic blood pressure
    'DBP': (40, 130),            # Diastolic blood pressure
    'BLDS': (50, 300),           # Blood sugar
    'tot_chole': (100, 300),     # Total cholesterol
    'HDL_chole': (20, 150),      # HDL cholesterol
    'LDL_chole': (30, 300),      # LDL cholesterol
    'triglyceride': (30, 1000),   # Triglycerides
    'hemoglobin': (6, 18),      # Hemoglobin

    'serum_creatinine': (0.4, 30),  # Kidney function
    'SGOT_AST': (5, 200),         # Liver enzyme
    'SGOT_ALT': (5, 200),         # Liver enzyme
    'gamma_GTP': (5, 500)        # Liver enzyme
}



# Detect outliers based on medical thresholds
for col, (low, high) in medical_ranges.items():
    outliers = sd_df[(sd_df[col] < low) | (sd_df[col] > high)]
    print(f"{col}: {len(outliers)} outliers (outside [{low}, {high}])")


for col, (low, high) in medical_ranges.items():
    sd_df = sd_df[(sd_df[col] >= low) & (sd_df[col] <= high)]

# Detect outliers based on medical thresholds
print("\nMedical range outliers:")
for col, (low, high) in medical_ranges.items():
    outliers = sd_df[(sd_df[col] < low) | (sd_df[col] > high)]
    print(f"{col}: {len(outliers)} outliers (outside [{low}, {high}])")

# Filter dataset to only include realistic medical values
for col, (low, high) in medical_ranges.items():
    sd_df = sd_df[(sd_df[col] >= low) & (sd_df[col] <= high)]


# Save the filtered dataset to a new CSV file
sd_df.to_csv("datasets/smoking_drinking_filtered.csv", index=False)
print("Filtered dataset saved to 'smoking_drinking_filtered.csv'.")