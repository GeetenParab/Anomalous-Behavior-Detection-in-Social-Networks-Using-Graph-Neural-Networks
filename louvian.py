import torch
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from gaemodel import GATEncoder
from torch_geometric.nn import GAE
from data1 import get_train_data

# 1. Setup Device and Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Loading data using {device}...")

# Load core data and user JSON for names/usernames
data = get_train_data('cresci-2015')
user_df = pd.read_json("../BotRGCN/datasets/cresci-2015/node.json")
user_id_map = {i: uid for i, uid in enumerate(user_df['id'])}
user_df = user_df.set_index('id')

# 2. Load GAE Model
encoder = GATEncoder(
    hidden_dim=128, 
    out_channels=64,
    num_prop_size=data.num_property_embedding.shape[-1],
    cat_prop_size=data.cat_property_embedding.shape[-1]
).to(device)
model = GAE(encoder).to(device)
model.load_state_dict(torch.load('checkpoints/gae_human_baseline.pt', map_location=device))
model.eval()

# 3. Generate Anomaly Scores
print("Generating Anomaly Scores...")
with torch.no_grad():
    z = model.encode(
        data.des_embedding.to(device), data.tweet_embedding.to(device),
        data.num_property_embedding.to(device), data.cat_property_embedding.to(device),
        data.edge_index.to(device)
    )
    adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
    from torch_geometric.utils import to_dense_adj
    adj_real = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(device)
    error = torch.pow(adj_real - adj_recon, 2)
    scores = torch.mean(error, dim=1).cpu().numpy()

# 4. Louvain Community Detection
print("Detecting Communities...")
G = nx.Graph()
edge_index = data.edge_index.cpu().numpy()
G.add_edges_from(list(zip(edge_index[0], edge_index[1])))
partition = community_louvain.best_partition(G)

# 5. Build Comprehensive Results Table
results_df = pd.DataFrame({
    'node_index': list(partition.keys()),
    'community_id': list(partition.values()),
    'anomaly_score': [scores[i] for i in partition.keys()]
})

# Add real User IDs and Usernames
results_df['user_id'] = results_df['node_index'].map(user_id_map)
results_df['username'] = results_df['user_id'].map(lambda x: user_df.loc[x, 'username'] if x in user_df.index else "Unknown")

# 6. Community Stats and Naming
community_stats = results_df.groupby('community_id').agg({
    'node_index': 'count',
    'anomaly_score': ['mean', 'std']
})
community_stats.columns = ['member_count', 'mean_error', 'std_error']

# Identify Anomalies
global_mean = community_stats['mean_error'].mean()
global_std = community_stats['mean_error'].std()
threshold = global_mean + (2 * global_std)

def name_community(row):
    if row['mean_error'] > threshold:
        return "Coordinated Botnet" if row['member_count'] > 10 else "Suspicious Cluster"
    return "Authentic Community"

community_stats['community_type'] = community_stats.apply(name_community, axis=1)

# 7. Visualization
print("Visualizing Top Anomalous Community...")
# Find the community with the highest mean error
top_anom_id = community_stats[community_stats['member_count'] > 5]['mean_error'].idxmax()
sub_nodes = [nodes for nodes, comm in partition.items() if comm == top_anom_id]
subgraph = G.subgraph(sub_nodes)

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(subgraph)
nx.draw(subgraph, pos, node_size=50, node_color='red', with_labels=False, alpha=0.6)
plt.title(f"Visualizing Botnet Community #{top_anom_id}\n(Avg Error: {community_stats.loc[top_anom_id, 'mean_error']:.4f})")
plt.show()

# 8. Output Report
print("\n--- DETECTED ANOMALOUS GROUPS ---")
anom_list = community_stats[community_stats['community_type'] != "Authentic Community"].sort_values(by='mean_error', ascending=False)
print(anom_list)

print("\n--- MEMBERS OF TOP ANOMALOUS GROUP ---")
members = results_df[results_df['community_id'] == top_anom_id]['username'].tolist()
print(f"Community #{top_anom_id} Members: {members[:20]}... (Total: {len(members)})")