import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gaemodel import GATEncoder
from torch_geometric.nn import GAE
from dataset import get_train_data
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Data and Model
print("Loading data...")
data = get_train_data('cresci-2015')
encoder = GATEncoder(hidden_dim=128, out_channels=64, 
                     num_prop_size=data.num_property_embedding.shape[-1],
                     cat_prop_size=data.cat_properties_tensor.shape[-1]).to(device)
model = GAE(encoder).to(device)

# Load checkpoint
checkpoint_path = 'checkpoints/gae_cresci15_auc_0.98.pt'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    print(f"Error: {checkpoint_path} not found!")

model.eval()

@torch.no_grad()
def get_results_df():
    # Generate node embeddings Z
    z = model.encode(
        data.des_embedding.to(device), 
        data.tweet_embedding.to(device),
        data.num_property_embedding.to(device),
        data.cat_properties_tensor.to(device), 
        data.edge_index.to(device)
    )
    
    # Reconstruct Adjacency Matrix
    adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
    
    from torch_geometric.utils import to_dense_adj
    adj_real = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(device)
    
    # Calculate Mean Squared Error (MSE) per node
    error = torch.pow(adj_real - adj_recon, 2)
    scores = torch.mean(error, dim=1).cpu().numpy()
    labels = data.y.numpy()

    return pd.DataFrame({
        'node_id': range(len(scores)),
        'anomaly_score': scores,
        'is_bot': labels
    })

# --- Execution ---
results = get_results_df()

# 2. Statistical Analysis (IQR Method)
Q1 = results['anomaly_score'].quantile(0.25)
Q3 = results['anomaly_score'].quantile(0.75)
IQR = Q3 - Q1
iqr_threshold = Q3 + (1.5 * IQR)

results['is_spammer'] = results['anomaly_score'] > iqr_threshold
spammer_list = results[results['is_spammer'] == True].copy()

# 3. Enhanced Visualization with IQR Boundary
plt.figure(figsize=(12, 7))

# Plot Human and Bot distributions
sns.histplot(results[results['is_bot'] == 0]['anomaly_score'], 
             color='skyblue', label='Humans', kde=True, stat="density", bins=50, alpha=0.6)
sns.histplot(results[results['is_bot'] == 1]['anomaly_score'], 
             color='salmon', label='Bots', kde=True, stat="density", bins=50, alpha=0.6)

# Draw the IQR Anomaly Boundary
plt.axvline(iqr_threshold, color='red', linestyle='--', linewidth=2, label=f'Spammer Threshold ({iqr_threshold:.4f})')

# Fill the Spammer Zone
plt.fill_betweenx([0, plt.gca().get_ylim()[1]], iqr_threshold, results['anomaly_score'].max(), 
                  color='red', alpha=0.1, label='Detected Spammer Zone')

plt.title('Cresci-15: Unsupervised Anomaly Detection (IQR Thresholding)')
plt.xlabel('Reconstruction Error (Anomaly Score)')
plt.ylabel('Density')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig('spammer_detection_plot.png')
print("Visual saved as 'spammer_detection_plot.png'")
plt.show()

# 4. Final Logs
print(f"\n--- Statistical Summary ---")
print(f"IQR Threshold: {iqr_threshold:.6f}")
print(f"Total Users: {len(results)}")
print(f"Detected Spammers (Outliers): {len(spammer_list)}")

if 'is_bot' in results.columns:
    bot_match = spammer_list['is_bot'].sum()
    print(f"Spammers that are confirmed Bots: {bot_match} ({ (bot_match/len(spammer_list)*100):.2f}%)")
    print("\n--- Averages ---")
    print(results.groupby('is_bot')['anomaly_score'].mean())

# Save to CSV
spammer_list[['node_id', 'anomaly_score', 'is_bot']].to_csv('detected_spammers.csv', index=False)
print("\nList saved to 'detected_spammers.csv'")



# import torch
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from gaemodel import GATEncoder
# from torch_geometric.nn import GAE
# from dataset import get_train_data

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 1. Load Data and Model
# data = get_train_data('cresci-2015')
# encoder = GATEncoder(hidden_dim=128, out_channels=64, 
#                      num_prop_size=data.num_property_embedding.shape[-1],
#                      cat_prop_size=data.cat_properties_tensor.shape[-1]).to(device)
# model = GAE(encoder).to(device)

# # Ensure the checkpoint name matches your saved file
# model.load_state_dict(torch.load('checkpoints/gae_cresci15_auc_0.98.pt'))
# model.eval()

# @torch.no_grad()
# def get_results_df():
#     # Generate embeddings
#     z = model.encode(
#         data.des_embedding.to(device), 
#         data.tweet_embedding.to(device),
#         data.num_property_embedding.to(device),
#         data.cat_properties_tensor.to(device), 
#         data.edge_index.to(device)
#     )
    
#     # Reconstruction
#     adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
    
#     from torch_geometric.utils import to_dense_adj
#     adj_real = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(device)
    
#     # Calculate Mean Squared Error per node
#     error = torch.pow(adj_real - adj_recon, 2)
#     scores = torch.mean(error, dim=1).cpu().numpy()
#     labels = data.y.numpy()

#     # Create the results DataFrame inside the script
#     df = pd.DataFrame({
#         'node_id': range(len(scores)),
#         'anomaly_score': scores,
#         'is_bot': labels
#     })
#     return df

# # --- Execution ---
# results = get_results_df() # This defines 'results'
# import pandas as pd

# # 1. Calculate the Quartiles and IQR
# Q1 = results['anomaly_score'].quantile(0.25)
# Q3 = results['anomaly_score'].quantile(0.75)
# IQR = Q3 - Q1

# # 2. Define the 'Upper Fence' Threshold
# # Any score higher than this is statistically an outlier (Spammer)
# iqr_threshold = Q3 + (1.5 * IQR)

# # 3. Flag and Extract the Spammers
# results['is_spammer'] = results['anomaly_score'] > iqr_threshold
# spammer_list = results[results['is_spammer'] == True].copy()

# # 4. Results Summary
# print(f"--- Statistical Analysis ---")
# print(f"IQR Threshold: {iqr_threshold:.6f}")
# print(f"Total Users: {len(results)}")
# print(f"Detected Spammers: {len(spammer_list)}")

# # 5. Cross-reference with your Dashboard labels
# # This shows how many detected 'spammers' were originally labeled as bots
# if 'is_bot' in spammer_list.columns:
#     bot_match = spammer_list['is_bot'].sum()
#     print(f"Spammers that are confirmed Bots: {bot_match} ({ (bot_match/len(spammer_list)*100):.2f}%)")

# # 6. Save the results for your Dashboard
# spammer_list[['node_id', 'anomaly_score']].to_csv('detected_spammers.csv', index=False)
# print("\nSpammer list saved to 'detected_spammers.csv'")
# # 2. Plotting the Distribution
# plt.figure(figsize=(10, 6))
# sns.histplot(results[results['is_bot'] == 0]['anomaly_score'], 
#              color='blue', label='Humans', kde=True, stat="density", bins=50)
# sns.histplot(results[results['is_bot'] == 1]['anomaly_score'], 
#              color='red', label='Bots', kde=True, stat="density", bins=50)

# plt.title('Cresci-15: Distribution of Anomaly Scores (GAE Reconstruction Error)')
# plt.xlabel('Reconstruction Error')
# plt.ylabel('Density')
# plt.legend()
# plt.savefig('anomaly_distribution.png')
# plt.show()

# print("\n--- Averages ---")
# print(results.groupby('is_bot')['anomaly_score'].mean())


