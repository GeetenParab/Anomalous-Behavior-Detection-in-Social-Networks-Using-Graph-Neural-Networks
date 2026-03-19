# visualize_output.py

import torch
import os.path as osp
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser

# --- Import your project's code ---
from dataset import get_train_data
from model import BotGAT

# --- Defi
class BotGAT_WithEmbeddings(BotGAT):
    def forward(self, num_prop, edge_index, edge_type=None):
        n = self.linear_relu_num_prop(num_prop)
        x = n
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        embeddings = self.linear_relu_output1(x)
        output = self.linear_output2(embeddings)
        return embeddings, output



@torch.no_grad()
def generate_embeddings(model, data, device):
    model.eval()
    data = data.to(device)
    embeddings, _ = model(data.num_property_embedding,
                          data.edge_index,
                          data.edge_type)
    return embeddings.cpu()

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Twibot-22', help="Name of the dataset")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument('--mode', type=str, required=True, help="Model mode used for training (GAT, GCN, or RGCN)")
    parser.add_argument('--hidden_dim', type=int, default=32, help="Hidden dimension of the model")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("✅ Loading data...")
    data = get_train_data(args.dataset)
    
    print(f"✅ Loading {args.mode} model...")
    if args.mode == 'GAT':
        model = BotGAT_WithEmbeddings(hidden_dim=args.hidden_dim, num_prop_size=data.num_property_embedding.shape[-1]).to(device)
   
    else:
        raise ValueError("Invalid mode specified")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    print("✅ Generating node embeddings from the trained model...")
    # NOTE: The *entire* data object is moved to the GPU inside this function for processing
    output_embeddings = generate_embeddings(model, data, device)

    sample_size = 20000
    print(f"Data loaded. Taking a random sample of {sample_size} nodes.")
    indices = torch.randperm(len(data.y))[:sample_size]

    # --- CHANGE: Added .cpu() before .numpy() ---
    sample_labels = data.y[indices].cpu().numpy()
    
    sample_embeddings = output_embeddings[indices].numpy()
    
    label_map = {0: 'Human', 1: 'Bot'}
    sample_labels_named = [label_map[label] for label in sample_labels]
    
    print("✅ Sample created. Running t-SNE... (this may take a few minutes)")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    tsne_results = tsne.fit_transform(sample_embeddings)
    
    print("✅ t-SNE complete. Generating plot...")
    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    df['label'] = sample_labels_named

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette={"Human": "dodgerblue", "Bot": "red"},
        data=df,
        legend="full",
        alpha=0.6
    )

    plt.title('t-SNE Visualization of GNN Output Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)

    output_filename = 'tsne_output_visualization.png'
    plt.savefig(output_filename)
    print(f"✅ Plot saved as {output_filename}")

if __name__ == '__main__':
    main()