import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.stats import truncnorm
import pdb

class InfoGraph(torch.nn.Module):

    def __init__(self, config, rep_model):
        super(InfoGraph, self).__init__()
        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.mask_ratio = self.config.train.mask_ratio
        self.edge_types = 0 if config.model.no_edge_types else config.model.order + 1
        self.model = rep_model
        
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.SiLU(),
                                nn.Linear(self.hidden_dim, self.hidden_dim))
        
        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.SiLU(),
                                nn.Linear(self.hidden_dim, self.hidden_dim))

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def get_info_graph_score(self, node_rep, node2graph):
        node_rep_proj = self.node_dec(node_rep)
        graph_rep = scatter_mean(node_rep, node2graph, dim = -2)
        graph_rep_proj = self.graph_dec(graph_rep)
        pos_graph_rep = graph_rep_proj[node2graph]
        neg_index = self.get_neg_index(graph_rep.shape[0], 1)
        neg_graph_rep = graph_rep_proj[neg_index][node2graph]
        pos_score = torch.sum(pos_graph_rep * node_rep, dim=-1)
        neg_score = torch.sum(neg_graph_rep * node_rep, dim=-1)
        return pos_score, neg_score


    @torch.no_grad()
    def get_distance(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data

    @torch.no_grad()
    def get_neg_index(self, num, bias):
        idx = torch.arange(num) + bias
        idx[-bias:] = torch.arange(bias)
        return idx
      
    @torch.no_grad()
    def gen_edge_onehot(self, edge_types):
        if not self.edge_types:
            return None
        return F.one_hot(edge_types.long(), self.edge_types)

    def forward(self, data):
        """
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        self.device = data.pos.device
        node2graph = data.batch
        node_feature_input = data.node_feature.clone()
        pos_input = data.pos.clone()
        edge_attr = self.gen_edge_onehot(data.edge_type)
        pred_fea, pred_pos = self.model(node_feature_input, pos_input, data.edge_index, edge_attr)

        pos_score, neg_score = self.get_info_graph_score(pred_fea, node2graph)
        loss_pos = self.loss_fct(pos_score, torch.ones_like(pos_score))
        loss_neg = self.loss_fct(neg_score, torch.zeros_like(neg_score))
        return loss_pos.mean() + loss_neg.mean()
