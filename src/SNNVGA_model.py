# General Imports
import random as rd
import warnings 

# Pytorch
import torch
import torch.nn as nn

# Pytorch Geometric
from torch_geometric.nn import GCNConv

# Set seeds for reproducibility
seed= 1171  

class LinkPred(torch.nn.Module):
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()

        # Define GCN layers for encoding
        self.conv1 = GCNConv(in_dim, hidden_dim, normalize = True)
        self.mu = GCNConv(hidden_dim, out_dim, normaliz = True)
        self.logvar = GCNConv(hidden_dim, out_dim)

        # Define loss function
        self.loss_function = torch.nn.BCEWithLogitsLoss()

        # Define a decoder network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
            )
        
        # Initialize weights using Xavier initialization
        warnings.filterwarnings("ignore", category=UserWarning, message="nn.init.xavier_uniform is now deprecated in favor of nn.init.xavier_uniform_.")

        torch.nn.init.xavier_uniform((self.conv1.lin.weight), gain=nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform(self.mu.lin.weight)
        torch.nn.init.xavier_uniform(self.logvar.lin.weight)

    # Reparameterization trick to sample from the latent space
    def reparameterize (self, mu,logvar):
        std = torch.exp(0.5 * logvar)  
        eps = torch.randn_like(std)  
        z = mu + eps * std   

        return z
    
    # Encoding function using GCN layers
    def encoding (self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        
        return x

    # Decoding function to reconstruct the edge  
    def decoding (self, z, edge_index):
        Sub1_emb = z[edge_index[0]]
        Sub2_emb = z[edge_index[1]]
        edge_features = torch.cat((Sub1_emb, Sub2_emb), dim = -1)
        x_cons = self.decoder(edge_features).squeeze()
        
        return x_cons

    # Define the loss function
    def loss_BCE(self, pred, label):
        
        return self.loss_function(pred, label)

    # Forward pass through the network
    def forward(self, data ):
        x = self.encoding(data.x, data.edge_index)
        mu = self.mu(x, data.edge_index)
        logvar = self.logvar(x, data.edge_index)
        z = self.reparameterize (mu, logvar)
        x_cons = self.decoding(z, data.edge_label_index)
        
        return mu , logvar, z, x_cons
