import torch
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from math import pi
from gaemodel import GATEncoder
from torch_geometric.nn import GAE
from data1 import get_train_data

# 1. Setup Device and Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running validation on {device}...")

data = get_train_data('cresci-2015')
user_df = pd.read_json("../BotRGCN/datasets/cresci-2015/node.json")

# Extract nested metrics from image_86a592 data structure
def extract_metrics(df):
    if 'public_metrics' in df.columns:
        df['followers_count'] = df['public_metrics'].apply(lambda x: x.get('followers_count', 0) if isinstance(x, dict) else 0)
        df['friends_count'] = df['public_metrics'].apply(lambda x: x.get('following_count', 0) if isinstance(x, dict) else 0)
        df['statuses_count'] = df['public_metrics'].apply(lambda x: x.get('tweet_count', 0) if isinstance(x, dict) else 0)
        df['favourites_count'] = df['public_metrics'].apply(lambda x: x.get('like_count', 0) if isinstance(x, dict) else 0)
    return df

user_df = extract_metrics(user_df)
user_id_map = {i: uid for i, uid in enumerate(user_df['id'])}

# 2. Generate Anomaly Scores
encoder = GATEncoder(hidden_dim=128, out_channels=64, num_prop_size=data.num_property_embedding.shape[-1], 
                     cat_prop_size=data.cat_property_embedding.shape[-1]).to(device)
model = GAE(encoder).to(device)
model.load_state_dict(torch.load('checkpoints/gae_human_baseline.pt', map_location=device))
model.eval()

with torch.no_grad():
    z = model.encode(data.des_embedding.to(device), data.tweet_embedding.to(device),
                     data.num_property_embedding.to(device), data.cat_property_embedding.to(device),
                     data.edge_index.to(device))
    adj_recon = torch.sigmoid(torch.matmul(z, z.t()))
    from torch_geometric.utils import to_dense_adj
    adj_real = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].to(device)
    error = torch.pow(adj_real - adj_recon, 2)
    scores = torch.mean(error, dim=1).cpu().numpy()

# 3. Form Communities
G = nx.Graph()
edge_index = data.edge_index.cpu().numpy()
G.add_edges_from(list(zip(edge_index[0], edge_index[1])))
partition = community_louvain.best_partition(G, random_state=42)

results_df = pd.DataFrame({'node_index': list(partition.keys()), 'community_id': list(partition.values()), 'anomaly_score': [scores[i] for i in partition.keys()]})
results_df['user_id'] = results_df['node_index'].map(user_id_map)

# 4. Target Community Analysis (Community #2 as seen in your logs)
TARGET_COMMUNITY = 3
comm_members = results_df[results_df['community_id'] == TARGET_COMMUNITY]
comm_user_details = user_df[user_df['id'].isin(comm_members['user_id'])]
human_baseline = user_df[user_df['id'].isin(results_df[results_df['anomaly_score'] < 0.25]['user_id'])]

# 5. Visualizations
def plot_validation(comm_df, baseline_df, comm_id):
    fig = plt.figure(figsize=(16, 8))
    labels = ['Followers', 'Following', 'Tweets', 'Favorites']
    
    def get_stats(df):
        return [df['followers_count'].mean(), df['friends_count'].mean(), df['statuses_count'].mean(), df['favourites_count'].mean()]

    comm_vals = get_stats(comm_df)
    base_vals = get_stats(baseline_df)
    norm_vals = [cv / (bv + 1) for cv, bv in zip(comm_vals, base_vals)]
    norm_vals += norm_vals[:1]
    
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    ax1 = plt.subplot(121, polar=True)
    plt.xticks(angles[:-1], labels)
    ax1.plot(angles, [1.0]*5, color='blue', linestyle='dashed', label='Human Baseline')
    ax1.plot(angles, norm_vals, color='red', linewidth=2, label=f'Comm #{comm_id}')
    ax1.fill(angles, norm_vals, 'red', alpha=0.3)
    plt.title(f"Behavioral Footprint (Comm #{comm_id})", size=15, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # --- WORDCLOUD FIX ---
    ax2 = plt.subplot(122)
    text = " ".join(comm_df['description'].fillna("").astype(str))
    if len(text.strip()) > 10:
        wc = WordCloud(width=600, height=400, background_color='white', colormap='Reds', stopwords=STOPWORDS).generate(text)
        # FIX: Convert to array manually to avoid NumPy asarray() error
        wc_array = np.array(wc.to_image()) 
        ax2.imshow(wc_array, interpolation='bilinear')
    ax2.set_title(f"Semantic Cloud (Comm #{comm_id})", size=15)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

# --- RUN ---
if len(comm_user_details) > 0:
    plot_validation(comm_user_details, human_baseline, TARGET_COMMUNITY)