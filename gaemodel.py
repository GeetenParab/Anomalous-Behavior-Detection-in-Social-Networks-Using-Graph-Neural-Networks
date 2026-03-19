import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GATEncoder(nn.Module):
    def __init__(self, hidden_dim, out_channels, des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=1, dropout=0.3):
        super(GATEncoder, self).__init__()
        self.linear_relu_des = nn.Sequential(nn.Linear(des_size, hidden_dim // 4), nn.LeakyReLU())
        self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, hidden_dim // 4), nn.LeakyReLU())
        self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, hidden_dim // 4), nn.LeakyReLU())
      
        self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, hidden_dim // 4), nn.LeakyReLU())

        self.linear_relu_input = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())
        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        self.gat2 = GATConv(hidden_dim, out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, des, tweet, num_prop, cat_prop, edge_index):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        
        x = torch.cat((d, t, n, c), dim=1)
        x = self.dropout(self.linear_relu_input(x))
        x = self.dropout(torch.relu(self.gat1(x, edge_index)))
        z = self.gat2(x, edge_index)
        return z