import torch
import os.path as osp
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser


from dataset import get_train_data
from model import BotGAT


class BotGAT_WithEmbeddings(BotGAT):
    # This forward pass MUST match the one used during training
    def forward(self, des, tweet, num_prop, cat_prop, edge_index, edge_type=None):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        x = torch.cat((d, t, n, c), dim=1)
        
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        
        # We define the layer before the final classification as the "embeddings"
        embeddings = self.linear_relu_output1(x)
        output = self.linear_output2(embeddings)
        return embeddings, output

@torch.no_grad()
def generate_embeddings(model, data, device):
    model.eval()
    data = data.to(device)
    # Pass ALL FOUR feature sets to the model
    embeddings, _ = model(data.des_embedding,
                          data.tweet_embedding,
                          data.num_property_embedding,
                          data.cat_properties_tensor, # Using the name we found before
                          data.edge_index,
                          data.edge_type)
    return embeddings.cpu()

def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cresci-2015', help="Name of the dataset")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument('--mode', type=str, required=True, help="Model mode used for training (GAT, GCN, or RGCN)")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Hidden dimension of the model")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("✅ Loading data...")
    data = get_train_data(args.dataset)
    
    
    required_attrs = ['des_embedding', 'tweet_embedding', 'num_property_embedding', 'cat_properties_tensor']
    for attr in required_attrs:
        if not hasattr(data, attr):
            raise AttributeError(f"FATAL: The data loader did not provide the required feature '{attr}'. Please fix your get_train_data() function.")
    print("✅ All required features found in the data object.")

    print(f"✅ Loading {args.mode} model...")
    if args.mode == 'GAT':
        # Instantiate the model with the correct sizes for ALL FOUR features
        model = BotGAT_WithEmbeddings(
            hidden_dim=args.hidden_dim,
            des_size=data.des_embedding.shape[-1],
            tweet_size=data.tweet_embedding.shape[-1],
            num_prop_size=data.num_property_embedding.shape[-1],
            cat_prop_size=data.cat_properties_tensor.shape[-1]
        ).to(device)
    else:
        raise ValueError("Invalid mode specified")

    # This should now work without errors
    print("✅ Loading checkpoint...")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print("✅ Checkpoint loaded successfully!")
    
    print("✅ Generating node embeddings from the trained model...")
    output_embeddings = generate_embeddings(model, data, device)

    sample_size = min(20000, len(data.y))
    print(f"Data loaded. Taking a random sample of {sample_size} nodes.")
    indices = torch.randperm(len(data.y))[:sample_size]

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