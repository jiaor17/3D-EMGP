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

class AttrMask(torch.nn.Module):

    def __init__(self, config, rep_model):
        super(AttrMask, self).__init__()
        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.mask_ratio = self.config.train.mask_ratio
        self.edge_types = 0 if config.model.no_edge_types else config.model.order + 1
        self.model = rep_model
        
        self.type_predictor = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.SiLU(),
                                    nn.Linear(self.hidden_dim, self.config.model.max_atom_type))

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

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
        node_range = scatter_min(torch.arange(node2graph.shape[0]).to(self.device),node2graph)[1].cpu().numpy()
        node_range = np.append(node_range, node2graph.shape[0])
        mask_nodes = [np.random.choice(num_node, max(1,int(num_node * self.mask_ratio)), replace = False) + idx_offset for num_node, idx_offset in zip(np.diff(node_range), node_range[:-1])]
        mask_nodes = np.concatenate(mask_nodes)
        mask_nodes = torch.from_numpy(mask_nodes).to(self.device)
        node_feature_input = data.node_feature.clone()
        pos_input = data.pos.clone()
        if self.config.model.charge_power == -1:
            node_feature_input[mask_nodes] = torch.zeros_like(mask_nodes)
        else:
            node_feature_input[mask_nodes] = F.one_hot(torch.zeros_like(mask_nodes),self.config.model.max_atom_type * (self.config.model.charge_power + 1)).to(data.node_feature)

        edge_attr = self.gen_edge_onehot(data.edge_type)

        pred_fea, pred_pos = self.model(node_feature_input, pos_input, data.edge_index, edge_attr)
        # pred_fea, pred_pos = self.model(node_feature_input, pos_input, batch=node2graph)

        pred_logits = self.type_predictor(pred_fea[mask_nodes])

        label = data.atom_type[mask_nodes]
        loss = self.loss_fct(pred_logits, label)
        return loss.mean()
