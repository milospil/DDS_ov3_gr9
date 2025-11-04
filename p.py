import pandas as pd
from sklearn.model_selection import train_test_split

# Load your complete dataset
df = pd.read_csv('datasets/preprocessed_data.csv')

# Add ID column if it doesn't exist
if 'id' not in df.columns:
    df.insert(0, 'id', range(len(df)))

# Separate features (X) and target variable (y)
X = df.drop('SMK_stat_type_cd', axis=1)
y = df[["id", 'SMK_stat_type_cd']]
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save to CSV files
X_train.to_csv('competition/x_train.csv', index=False)
X_test.to_csv('competition/x_test.csv', index=False)
y_train.to_csv('competition/y_train.csv', index=False)
y_test.to_csv('competition/y_test.csv', index=False)