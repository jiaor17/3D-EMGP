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

class GraphCL(torch.nn.Module):

    def __init__(self, config, rep_model):
        super(GraphCL, self).__init__()
        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.mask_ratio = self.config.train.mask_ratio
        self.edge_types = 0 if config.model.no_edge_types else config.model.order + 1
        self.model = rep_model
        
        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                nn.SiLU(),
                                nn.Linear(self.hidden_dim, self.hidden_dim))

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')


    def get_cl_score(self, x1, x2, node2graph1, node2graph2):
        # x2 = x2.detach()
        xg1 = scatter_mean(x1, node2graph1, dim=-2)
        xg2 = scatter_mean(x2, node2graph2, dim=-2)
        xg1 = self.graph_dec(xg1)
        xg2 = self.graph_dec(xg2)
        T = self.config.train.T
        batch, _ = xg1.size()
        xg1n = xg1.norm(dim=1)
        xg2n = xg2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', xg1, xg2) #/ torch.einsum('i,j->ij', xg1n, xg2n)
        sim_matrix = sim_matrix / T
        label = torch.arange(batch).to(sim_matrix.device).long()
        loss = self.loss_fct(sim_matrix, label)
        return loss


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
        data1, data2 = data
        self.device = data1.pos.device
        node2graph1 = data1.batch
        node2graph2 = data2.batch
        edge_attr1 = self.gen_edge_onehot(data1.edge_type)
        edge_attr2 = self.gen_edge_onehot(data2.edge_type)
        pred_fea1, pred_pos1 = self.model(data1.node_feature, data1.pos, data1.edge_index, edge_attr1)
        pred_fea2, pred_pos2 = self.model(data2.node_feature, data2.pos, data2.edge_index, edge_attr2)
        loss = self.get_cl_score(pred_fea1, pred_fea2, node2graph1, node2graph2)
        return loss.mean()
