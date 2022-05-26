#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional
from math import pi as PI
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing, radius_graph


class SchNet(nn.Module):
    """
    Schnet by K.T.Schutt et al. 2017
    :param in_node_nf (int, required): Number of features for 'h' in the input
    :param out_node_nf (int, required): Number of features for 'h' in the output
    :params hidden_nf (int, optional): Hidden feature size. (default 128)
    :params num_filters (int, optional): The number of filters to use. (default 128)
    :params num_interactions (int, optional): The number of interaction blocks. (default 6)
    :params num_gaussians (int, optional): The number of guassians. (default 50)
    :params cutoff (float, optional): Cutoff distance for interatomic interactions. (default 10.0)
    :params max_num_neighbors (int, optional): The maximum number of neighbors to collect for each
        node within the 'cutoff' distance. (default 32)
    """
    def __init__(self, in_node_nf, out_node_nf, hidden_nf, num_filters, in_edge_nf,
                 num_interactions, num_gaussians: int=50, cutoff: float=10.0, max_num_neighbors: int=32):
        super().__init__()
        
        self.in_node_nf = in_node_nf
        self.out_node_nf = out_node_nf
        self.hidden_nf = hidden_nf
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.in_edge_nf = in_edge_nf

        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            self.interactions.append(InteractionBlock(hidden_nf, num_gaussians + in_edge_nf,
                                                      num_filters, cutoff))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf // 2),
            ShiftedSoftplus(),
            nn.Linear(hidden_nf // 2, out_node_nf)
        )
        

    @property
    def params(self):
        """
        Get the parameters to optimize.
        :return: The parameters to optimize.
        """
        return self.parameters()
    
    def forward(self, h, pos, edge_index, edge_input):
        h = self.embedding_in(h)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)  # calculate edge weight according to the distance
        edge_attr = self.distance_expansion(edge_weight)
        edge_attr = torch.cat([edge_attr, edge_input], dim=1)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.mlp(h)
        return h, pos


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, _nn, cutoff):
        super().__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.nn = _nn
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift