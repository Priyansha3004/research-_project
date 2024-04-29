import pandas as pd

# Load your data into a DataFrame
filename = pd.read_csv("ai.csv")

# Get the dimensions of the DataFrame
shape = filename.shape

# Print the dimensions
print("Number of rows:", shape[0])
print("Number of columns:", shape[1])
