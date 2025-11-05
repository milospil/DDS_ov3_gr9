import pandas as pd
from sklearn.model_selection import train_test_split

# Load your complete dataset
df = pd.read_csv('datasets/preprocessed_data.csv')

# Add ID column if it doesn't exist
if 'id' not in df.columns:
    df.insert(0, 'id', range(len(df)))

# Separate features (X) and target variable (y)
X = df.drop('SMK_stat_type_cd', axis=1)
y = df['SMK_stat_type_cd']

# Split into 60% and 40% - STRATIFIED
X_60, X_40, y_60, y_40 = train_test_split(
    X, y, 
    test_size=0.4,  # 40% portion
    random_state=42,
    stratify=y  # Keep class distribution balanced
)

print(f"60% portion: {X_60.shape}")
print(f"40% portion: {X_40.shape}")

# Prepare 60% portion with IDs and target
df_60 = X_60.copy()
df_60['SMK_stat_type_cd'] = y_60

# Prepare 40% portion with IDs and target
df_40 = X_40.copy()
df_40['SMK_stat_type_cd'] = y_40

# Save to CSV files
df_60.to_csv('competition/oppg8/dataset_60_percent.csv', index=False)
df_40.to_csv('competition/dataset_40_percent.csv', index=False)

print("\n✓ Files saved to competition/ folder")
print(f"  - dataset_60_percent.csv: {df_60.shape}")
print(f"  - dataset_40_percent.csv: {df_40.shape}")
print("\nClass distribution in 60%:")
print(y_60.value_counts().sort_index())
print("\nClass distribution in 40%:")
print(y_40.value_counts().sort_index())

# Now split the 40% portion into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_40, y_40,
    test_size=0.2,  # 20% of 40% = 8% of total
    random_state=42,
    stratify=y_40
)

print(f"\nFrom 40% portion:")
print(f"  - Train set: {X_train.shape} (80% of 40%)")
print(f"  - Test set: {X_test.shape} (20% of 40%)")

# Prepare y_train and y_test with IDs
y_train_df = pd.DataFrame({
    'id': X_train['id'].values,
    'SMK_stat_type_cd': y_train.values
})

y_test_df = pd.DataFrame({
    'id': X_test['id'].values,
    'SMK_stat_type_cd': y_test.values
})

# Save train/test splits
X_train.to_csv('competition/x_train.csv', index=False)
X_test.to_csv('competition/x_test.csv', index=False)
y_train_df.to_csv('competition/y_train.csv', index=False)
y_test_df.to_csv('competition/y_test.csv', index=False)

print("\n✓ Train/Test files saved:")
print(f"  - x_train.csv: {X_train.shape}")
print(f"  - x_test.csv: {X_test.shape}")
print(f"  - y_train.csv: {y_train_df.shape}")
print(f"  - y_test.csv: {y_test_df.shape}")