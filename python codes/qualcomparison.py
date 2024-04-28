import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into DataFrame
df = pd.read_csv("aigenerated.csv")

# Filter the DataFrame based on the Source column
chatgpt_df = df[df['Source'] == 'ChatGPT']
bard_df = df[df['Source'] == 'Bard']

# Initialize lists to store metrics
types = []  # List to store types
word_counts_chatgpt = []  # List to store word counts for ChatGPT
word_counts_bard = []  # List to store word counts for Bard

# Compare word count for each type
for type_ in df['Type'].unique():
    chatgpt_content = chatgpt_df[chatgpt_df['Type'] == type_]['Content'].values
    bard_content = bard_df[bard_df['Type'] == type_]['Content'].values
    
    # Check if there are rows matching the current type in both dataframes
    if len(chatgpt_content) > 0 and len(bard_content) > 0:
        chatgpt_content = chatgpt_content[0]
        bard_content = bard_content[0]
        
        # Calculate word counts
        types.append(type_)
        word_counts_chatgpt.append(len(chatgpt_content.split()))
        word_counts_bard.append(len(bard_content.split()))
    else:
        print(f"No matching content found for type '{type_}' in both ChatGPT and Bard data.")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(types, word_counts_chatgpt, label='ChatGPT Word Count')
plt.plot(types, word_counts_bard, label='Bard Word Count')
plt.xlabel('Type')
plt.ylabel('Word Count')
plt.title('Comparison of Word Count between ChatGPT and Bard')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
