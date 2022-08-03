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

loss_func = {
    "L1" : nn.L1Loss(reduction='none'),
    "L2" : nn.MSELoss(reduction='none'),
    "Cosine" : nn.CosineSimilarity(dim=-1, eps=1e-08),
    "CrossEntropy" : nn.CrossEntropyLoss(reduction='none')
}

class GemPretrain(torch.nn.Module):

    def __init__(self, config, rep_model):
        super(GemPretrain, self).__init__()
        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.mask_ratio = self.config.train.mask_ratio
        self.edge_types = 0 if config.model.no_edge_types else config.model.order + 1
        self.model = rep_model
        self.bond_dec = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                nn.SiLU(),
                                nn.Linear(self.hidden_dim, 1))
        
        self.angle_dec = nn.Sequential(nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                                nn.SiLU(),
                                nn.Linear(self.hidden_dim, 1))

        self.dist_dec = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                nn.SiLU(),
                                nn.Linear(self.hidden_dim, 30))
        

    @torch.no_grad()
    def get_distance(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data
      
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
        bond_row, bond_col = data.bond_index
        bond_pred = self.bond_dec(torch.cat([pred_fea[bond_row], pred_fea[bond_col]], dim=1)).squeeze(1)
        loss_bond = loss_func['L2'](bond_pred, data.bond)
        angle_1, angle_2, angle_3 = data.angle_index
        angle_pred = self.angle_dec(torch.cat([pred_fea[angle_1], pred_fea[angle_2], pred_fea[angle_3]], dim=1)).squeeze(1)
        loss_angle = loss_func['L2'](angle_pred, data.angle)
        dist_row, dist_col = data.dist_index
        dist_pred = self.dist_dec(torch.cat([pred_fea[dist_row], pred_fea[dist_col]], dim=1))
        loss_dist = loss_func['CrossEntropy'](dist_pred, data.dist)
        return loss_bond.mean() + loss_angle.mean() + loss_dist.mean()
