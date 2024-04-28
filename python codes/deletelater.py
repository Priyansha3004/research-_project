# check_columns.py

import pandas as pd

# Load your CSV file into a DataFrame
df = pd.read_csv('ai.csv')

# Print column names
print(df.columns)