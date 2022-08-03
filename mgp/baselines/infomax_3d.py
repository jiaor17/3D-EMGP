import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.stats import truncnorm
from .gnns import GNN
from torch.nn.modules.loss import _Loss
from torch import Tensor
import pdb

def uniformity_loss(x1: Tensor, x2: Tensor, t=2) -> Tensor:
    sq_pdist_x1 = torch.pdist(x1, p=2).pow(2)
    uniformity_x1 = sq_pdist_x1.mul(-t).exp().mean().log()
    sq_pdist_x2 = torch.pdist(x2, p=2).pow(2)
    uniformity_x2 = sq_pdist_x2.mul(-t).exp().mean().log()
    return (uniformity_x1 + uniformity_x2) / 2


def cov_loss(x):
    batch_size, metric_dim = x.size()
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (batch_size - 1)
    off_diag_cov = cov.flatten()[:-1].view(metric_dim - 1, metric_dim + 1)[:, 1:].flatten()
    return off_diag_cov.pow_(2).sum() / metric_dim


def std_loss(x):
    std = torch.sqrt(x.var(dim=0) + 1e-04)
    return torch.mean(torch.relu(1 - std))


class NTXent(_Loss):
    '''
        Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
        Args:
            z1, z2: Tensor of shape [batch_size, z_dim]
            tau: Float. Usually in (0,1].
            norm: Boolean. Whether to apply normlization.
        '''

    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super(NTXent, self).__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum('ik,jk->ij', z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum('i,j->ij', z1_abs, z2_abs) + 1e-8)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class Infomax3D(torch.nn.Module):

    def __init__(self, config, rep_model):
        super(Infomax3D, self).__init__()
        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.mask_ratio = self.config.train.mask_ratio
        self.edge_types = 0 if config.model.no_edge_types else config.model.order + 1
        self.model = rep_model


        self.x_embedding = nn.Embedding(config.model.max_atom_type, config.model.hidden_dim)


        self.gnn_model = GNN(config.gnn_model.num_layer, config.model.hidden_dim, JK=config.gnn_model.JK, drop_ratio=config.gnn_model.dropout_ratio, \
                                gnn_type=config.gnn_model.gnn_type, num_atom_type = config.model.max_atom_type, num_bond_type = config.gnn_model.edge_type)

        self.loss_func = NTXent(tau = config.gnn_model.tau)
      
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
        data_3d, data_2d = data
        self.device = data_3d.pos.device
        node2graph = data_3d.batch
        node_feature_input = data_3d.node_feature.clone()
        pos_input = data_3d.pos.clone()
        edge_attr = self.gen_edge_onehot(data_3d.edge_type)
        pred_fea_3d, pred_pos = self.model(node_feature_input, pos_input, data_3d.edge_index, edge_attr)

        pred_fea_2d = self.gnn_model(data_2d.atom_type.clone(), data_2d.edge_index, data_2d.edge_type.clone())

        graph_rep_2d = scatter_mean(pred_fea_2d, node2graph, dim=-2)

        graph_rep_3d = scatter_mean(pred_fea_3d, node2graph, dim=-2)

        loss = self.loss_func(graph_rep_2d, graph_rep_3d)

        return loss
