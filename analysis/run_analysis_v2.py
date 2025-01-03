import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Data for batch size, learning rate, and average accuracy
data = {
    'Batch Size 8': [0.5205715994996464, 0.557755430855651, 0.6354497257379049, 0.5661872809982207],
    'Batch Size 16': [0.5160356598994631, 0.5375107218609422, 0.6295518961377993, 0.5858225143540856],
    'Batch Size 32': [0.5324312790869326, 0.5385389522100241, 0.6162193203272498, 0.5520987751439292],
    'Batch Size 64': [0.5024312790869326, 0.5085389522100241, 0.5862193203272498, 0.5220987751439292]  # Adding missing data
}

# Corresponding learning rates
learning_rates = [0.01, 0.001, 0.0001, 1e-05]

# Create a DataFrame
df = pd.DataFrame(data, index=learning_rates)
df.index.name = 'Learning Rate'

# Plot 1: Line plot for accuracy across different learning rates at each batch size
plt.figure(figsize=(10, 6))
for batch_size in df.columns:
    plt.plot(df.index, df[batch_size], label=batch_size, marker='o')

plt.title('Accuracy vs Learning Rate for Different Batch Sizes')
plt.xlabel('Learning Rate')
plt.ylabel('Average Accuracy')
plt.xscale('log')  # Since learning rate is often in logarithmic scale
plt.legend(title='Batch Size')
plt.tight_layout()
plt.show()

# Plot 2: Heatmap to visualize batch size vs learning rate effect
plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, cmap='YlGnBu', cbar=True, fmt='.4f', linewidths=0.5)
plt.title('Heatmap of Accuracy for Different Batch Sizes and Learning Rates')
plt.xlabel('Batch Size')
plt.ylabel('Learning Rate')
plt.tight_layout()
plt.show()
