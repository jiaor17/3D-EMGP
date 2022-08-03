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

class EdgePred(torch.nn.Module):

    def __init__(self, config, rep_model):
        super(EdgePred, self).__init__()
        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.mask_ratio = self.config.train.mask_ratio
        self.edge_types = 0 if config.model.no_edge_types else config.model.order + 1
        self.model = rep_model
        
        self.projector = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.SiLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim))

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        

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
        edge_pos = data.edge_index[:,data.edge_type == 1]
        edge_neg = data.edge_index[:,data.edge_type != 1]

        num_samples = edge_pos.shape[-1] // 2

        pos_sample = torch.randperm(edge_pos.shape[-1])[:num_samples].to(edge_pos)
        neg_sample = torch.randperm(edge_neg.shape[-1])[:num_samples].to(edge_neg)

        node_feature_input = data.node_feature.clone()
        pos_input = data.pos.clone()
        edge_attr = self.gen_edge_onehot(data.edge_type)
        pred_fea, pred_pos = self.model(node_feature_input, pos_input, data.edge_index, edge_attr)

        # pred_logits = self.type_predictor(pred_fea[mask_nodes])
        node_rep = self.projector(pred_fea)
        pos_score = torch.sum(node_rep[edge_pos[0,pos_sample]] * node_rep[edge_pos[1,pos_sample]], dim = 1)
        neg_score = torch.sum(node_rep[edge_neg[0,neg_sample]] * node_rep[edge_neg[1,neg_sample]], dim = 1)
        loss_pos = self.loss_fct(pos_score, torch.ones_like(pos_score))
        loss_neg = self.loss_fct(neg_score, torch.zeros_like(neg_score))
        return loss_pos.mean() + loss_neg.mean()
