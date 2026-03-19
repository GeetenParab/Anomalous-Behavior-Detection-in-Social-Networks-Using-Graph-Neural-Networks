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

# 1. Load Data and the Human-Baseline Model
data = get_train_data('cresci-2015')
encoder = GATEncoder(
    hidden_dim=128, 
    out_channels=64, 
    num_prop_size=data.num_property_embedding.shape[-1],
    cat_prop_size=data.cat_properties_tensor.shape[-1]
).to(device)

model = GAE(encoder).to(device)

# LOAD THE NEW HUMAN-ONLY CHECKPOINT
checkpoint_path = 'checkpoints/gae_human_baseline.pt'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Using Human-Baseline Model: {checkpoint_path}")
model.eval()

@torch.no_grad()
def get_results_df():
    z = model.encode(
        data.des_embedding.to(device), 
        data.tweet_embedding.to(device),
        data.num_property_embedding.to(device),
        data.cat_properties_tensor.to(device), 
        data.edge_index.to(device)
    )
    
    # Calculate reconstruction error against full graph structure
    adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
    from torch_geometric.utils import to_dense_adj
    adj_real = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(device)
    
    error = torch.pow(adj_real - adj_recon, 2)
    scores = torch.mean(error, dim=1).cpu().numpy()
    
    return pd.DataFrame({
        'node_id': range(len(scores)),
        'anomaly_score': scores,
        'is_bot': data.y.numpy()
    })

# --- Run Analysis ---
results = get_results_df()

# 2. Use 95th Percentile for a stricter Spammer Zone
# IQR was capturing too much; Percentile is better for targeted dashboards
threshold = results['anomaly_score'].quantile(0.95) 

results['is_spammer'] = results['anomaly_score'] > threshold
spammers = results[results['is_spammer']].copy()

# 3. Final Visual
plt.figure(figsize=(12, 7))
sns.histplot(results[results['is_bot'] == 0]['anomaly_score'], 
             color='blue', label='Humans (Normal)', kde=True, stat="density", bins=50)
sns.histplot(results[results['is_bot'] == 1]['anomaly_score'], 
             color='red', label='Bots/Spammers (Anomalies)', kde=True, stat="density", bins=50)

plt.axvline(threshold, color='black', linestyle='--', label=f'Spammer Threshold ({threshold:.4f})')
plt.fill_betweenx([0, plt.gca().get_ylim()[1]], threshold, results['anomaly_score'].max(), 
                  color='red', alpha=0.2, label='High-Confidence Spammer Zone')

plt.title('Cresci-15: Spammer Detection via Human-Normality Baseline')
plt.xlabel('Reconstruction Error (How "Non-Human" is the account?)')
plt.ylabel('Density')
plt.legend()
plt.show()

# 4. Export for Dashboard
spammers[['node_id', 'anomaly_score']].to_csv('spammer_dashboard_data.csv', index=False)
print(f"Detected {len(spammers)} high-confidence anomalies.")
print(f"Success Rate (Anomalies that are Bots): {(spammers['is_bot'].sum()/len(spammers)*100):.2f}%")


import pandas as pd

# 1. Define the High-Confidence Threshold based on your visual (image_d568c5)
# Accounts with error > 0.30 are statistically significant non-human anomalies
spammer_threshold = 0.30

# 2. Filter the results
# We flag humans with high error as 'Spammers' and bots as 'Structural Anomalies'
spammers_df = results[results['anomaly_score'] > spammer_threshold].copy()

# 3. Sort by highest score to show the most extreme cases first
spammers_df = spammers_df.sort_values(by='anomaly_score', ascending=False)

# 4. Results Summary for console
print(f"--- Spammer Integration Summary ---")
print(f"Threshold applied: {spammer_threshold}")
print(f"Accounts flagged for Spammer Table: {len(spammers_df)}")

# 5. Export to CSV (This file will populate your dashboard table)
# Includes is_bot to help you verify if they were labeled bots or unique humans
spammers_df[['node_id', 'anomaly_score', 'is_bot']].to_csv('spammer_accounts.csv', index=False)
print("Data saved to 'spammer_accounts.csv'. Ready for dashboard integration.")