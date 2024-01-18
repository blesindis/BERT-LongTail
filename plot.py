import pandas as pd
import matplotlib.pyplot as plt

# Re-loading the data due to execution state reset
file_paths_new = [
    "0112-mot-vanilla-wiki103(256).csv",
    "0117-moet-switch-wiki103(256)-bs24-(save)-sentence(3epoch).csv",
    "0117-moet-switch-wiki103(256)-bs24-(save)-token.csv",
    "0117-mot-vanilla-wiki103(256)-bs24-(save)-token.csv",
    "0115-moe4-wiki103(256)-bs24.csv"
]

# Loading and filtering the data
dataframes_new = []
for file_path in file_paths_new:
    df = pd.read_csv(file_path)
    df_filtered = df[df['Step'] <= 42800]
    # df_filtered = df
    dataframes_new.append(df_filtered)

# Colors for each line with more distinct colors for the last two
colors_updated = ['red', 'green', 'orange', 'magenta', 'blue']  # Substituted blue with cyan and purple with magenta

# Labels as specified
labels_updated = ['MoT-Input (Sentence)', 'MoT-Gating (Sentence)', 'MoT-Gating (Token)', 'MoT-Input (Token)', "MoE"]

# Plotting the data with updated labels and colors
plt.figure(figsize=(12, 8))

for i, df in enumerate(dataframes_new):
    plt.plot(df['Step'], df['Value'], label=labels_updated[i], color=colors_updated[i])

# Setting titles and labels
plt.title('Training Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')

# Adding legend
plt.legend()

# Show the plot
plt.savefig("Loss.png")
