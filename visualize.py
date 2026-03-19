# visualize.py

import torch
import os.path as osp
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

print("✅ Loading data for visualization...")

# --- Configuration ---/
# C:\Users\Geeten\zzzzproject\TwiBot-22\src\BotRGCN\cresci_15\processed_data
# C:\Users\Geeten\zzzzproject\TwiBot-22\src\BotRGCN\twibot_22
path = '../BotRGCN/twibot_22/processed_data' 
sample_size = 20000 

# --- Load Data ---
labels = torch.load(osp.join(path, 'label.pt'))
num_properties = torch.load(osp.join(path, 'num_properties_tensor.pt'))

print(f"Data loaded. Taking a random sample of {sample_size} nodes.")

# --- Prepare a Sample ---
indices = torch.randperm(len(labels))[:sample_size]

sample_labels = labels[indices].numpy()
sample_properties = num_properties[indices].numpy()

label_map = {0: 'Human', 1: 'Bot'}
sample_labels_named = [label_map[label] for label in sample_labels]

print("✅ Sample created. Running t-SNE... (this may take a few minutes)")

# --- Run t-SNE ---
tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000) # Using max_iter for newer versions
tsne_results = tsne.fit_transform(sample_properties)

print("✅ t-SNE complete. Generating plot...")

# --- Plot the Results ---
df = pd.DataFrame()
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]
df['label'] = sample_labels_named

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    # --- CHANGE: Removed the incorrect sns.color_palette() wrapper ---
    palette={"Human": "dodgerblue", "Bot": "red"},
    data=df,
    legend="full",
    alpha=0.6
)

plt.title('t-SNE Visualization of Node Features (Humans vs. Bots)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True)

output_filename = 'tsne_visualization.png'
plt.savefig(output_filename)
print(f"✅ Plot saved as {output_filename}")

# Optionally show the plot in a window
# plt.show()