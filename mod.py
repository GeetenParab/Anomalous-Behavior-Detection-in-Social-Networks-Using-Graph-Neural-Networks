import torch.nn as nn
import torch
from torch_geometric.nn.conv.gat_conv import GATConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv


class BotGAT(nn.Module):
    
    def __init__(self, hidden_dim, num_prop_size=5, dropout=0.3):
        super(BotGAT, self).__init__()

        
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, hidden_dim),
            nn.LeakyReLU()
        )
        
        
        self.linear_relu_input = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dim, 2)

        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    
    def forward(self, num_prop, edge_index, edge_type=None):
        n = self.linear_relu_num_prop(num_prop)
        
        x = n
        x = self.dropout(x)
        x = self.linear_relu_input(x)
        x = self.gat1(x, edge_index)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        return x