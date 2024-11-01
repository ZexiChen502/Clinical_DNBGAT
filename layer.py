#%%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


#%%


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dims, heads, dropout):
        super(GAT, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout = dropout
        # input layer
        self.layers.append(GATConv(num_node_features, hidden_dims[0] // heads, heads=heads))

        # hidden layer
        for i in range(1, len(hidden_dims)):
            self.layers.append(GATConv(hidden_dims[i - 1], hidden_dims[i] // heads, heads=heads))

        # output layer
        self.layers.append(GATConv(hidden_dims[-1], num_classes, heads=1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layers[-1](x, edge_index)

        return x





