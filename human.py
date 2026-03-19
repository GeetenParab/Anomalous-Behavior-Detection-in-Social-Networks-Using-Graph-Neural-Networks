import torch
from torch_geometric.nn import GAE
from gaemodel import GATEncoder 
from dataset import get_train_data
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Data
data = get_train_data('cresci-2015')

# ==========================================================
# CRUCIAL STEP: Filter for Humans only for the "Normal" baseline
# ==========================================================
# We only use human indices for the message passing and reconstruction during training
human_mask = (data.y == 0)
human_indices = human_mask.nonzero(as_tuple=False).view(-1)

# 2. Edge-Level Split (applied to the full graph, but we filter later)
transform = RandomLinkSplit(
    num_val=0.05, 
    num_test=0.1, 
    is_undirected=True, 
    add_negative_train_samples=True, 
    split_labels=True
)
train_data, val_data, test_data = transform(data)

# 3. Initialize Model
encoder = GATEncoder(
    hidden_dim=128, 
    out_channels=64,
    num_prop_size=data.num_property_embedding.shape[-1],
    cat_prop_size=data.cat_properties_tensor.shape[-1]
).to(device)

model = GAE(encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def train():
    model.train()
    optimizer.zero_grad()
    
    z = model.encode(
        train_data.des_embedding.to(device), 
        train_data.tweet_embedding.to(device),
        train_data.num_property_embedding.to(device),
        train_data.cat_properties_tensor.to(device), 
        train_data.edge_index.to(device)
    )
    
    # We only calculate loss for human-to-human edges to define "Human Normality"
    # Filter edge_label_index to keep only those where both source and target are humans
    edge_index = train_data.pos_edge_label_index
    mask = human_mask[edge_index[0]] & human_mask[edge_index[1]]
    human_pos_edges = edge_index[:, mask]
    
    loss = model.recon_loss(z, human_pos_edges.to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

# Training Loop
for epoch in tqdm(range(1, 201), desc="Training Human-Only GAE"):
    loss = train()

# Save as a new checkpoint
torch.save(model.state_dict(), 'checkpoints/gae_human_baseline.pt')
print("Model trained on human normality and saved.")