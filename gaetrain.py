import torch
from torch_geometric.nn import GAE
from gaemodel import GATEncoder 
from dataset import get_train_data
from torch_geometric.transforms import RandomLinkSplit
from tqdm import tqdm
import os
# --- Configuration ---
TRIAL_MODE = False  # Set to False for the full 200-epoch run
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if TRIAL_MODE:
    MAX_EPOCHS = 5
    HIDDEN_DIM, OUT_CHANNELS = 32, 16
    LR = 1e-2
else:
    MAX_EPOCHS = 200
    HIDDEN_DIM, OUT_CHANNELS = 128, 64
    LR = 1e-3

# 1. Load Data
data = get_train_data('cresci-2015')

# 2. Edge-Level Split (Crucial for Unsupervised Learning)
# We add negative samples and split labels to help recon_loss
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
    hidden_dim=HIDDEN_DIM, 
    out_channels=OUT_CHANNELS,
    num_prop_size=data.num_property_embedding.shape[-1],
    cat_prop_size=data.cat_properties_tensor.shape[-1],
    dropout=0.3
).to(device)

model = GAE(encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

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
    
    # FIX: Pass pos_edge_label_index instead of edge_label_index
    loss = model.recon_loss(z, train_data.pos_edge_label_index.to(device))
    
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(data_obj):
    model.eval()
    z = model.encode(
        data_obj.des_embedding.to(device), 
        data_obj.tweet_embedding.to(device),
        data_obj.num_property_embedding.to(device),
        data_obj.cat_properties_tensor.to(device), 
        data_obj.edge_index.to(device)
    )
    
    # FIX: Pass both positive and negative indices
    return model.test(
        z, 
        data_obj.pos_edge_label_index.to(device), 
        data_obj.neg_edge_label_index.to(device)
    )

# --- Main Training Loop with Progress Bar ---
pbar = tqdm(range(1, MAX_EPOCHS + 1), desc="Training GAE", unit="epoch")

for epoch in pbar:
    loss = train()
    
    # Update progress bar with current metrics
    if epoch % 10 == 0 or TRIAL_MODE:
        auc, ap = evaluate(val_data)
        pbar.set_postfix({'loss': f'{loss:.4f}', 'AUC': f'{auc:.4f}', 'AP': f'{ap:.4f}'})

# Final Evaluation on Test Set
auc, ap = evaluate(test_data)
print(f'\nFinal Test Results - AUC: {auc:.4f}, AP: {ap:.4f}')

if not TRIAL_MODE:
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/gae_cresci15_auc_{auc:.2f}.pt')