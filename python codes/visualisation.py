import pandas as pd
import matplotlib.pyplot as plt

# Read the first file
file1 = "ai.csv"  # Replace with your file path
df1 = pd.read_csv(file1)

# Read the second file
file2 = "human.csv"  # Replace with your file path
df2 = pd.read_csv(file2)

# Plotting for the first file
plt.figure(figsize=(10, 6))
df1['type'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Types in AI Generated Content')
plt.xlabel('Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plotting for the second file
plt.figure(figsize=(10, 6))
df2['type'].value_counts().plot(kind='bar', color='salmon')
plt.title('Distribution of Types in Human Generated Content')
plt.xlabel('Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
