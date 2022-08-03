import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.stats import truncnorm
from confgf import utils, layers
from .gnns import GNN
import pdb

class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, emb_dim, detach_target=True, beta=1):
        super(VariationalAutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.detach_target = detach_target
        self.beta = beta

        self.criterion = nn.MSELoss()

        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decoder(z)

        reconstruction_loss = self.criterion(y_hat, y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = reconstruction_loss + self.beta * kl_loss

        return loss

class GraphMVP(torch.nn.Module):

    def __init__(self, config, rep_model):
        super(GraphMVP, self).__init__()
        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.mask_ratio = self.config.train.mask_ratio
        self.edge_types = 0 if config.model.no_edge_types else config.model.order + 1
        self.model = rep_model

        self.gnn_model = GNN(config.gnn_model.num_layer, config.model.hidden_dim, JK=config.gnn_model.JK, drop_ratio=config.gnn_model.dropout_ratio, \
                                gnn_type=config.gnn_model.gnn_type, num_atom_type = config.model.max_atom_type, num_bond_type = config.gnn_model.edge_type)

        self.vae_2d_3d = VariationalAutoEncoder(config.model.hidden_dim)
        self.vae_3d_2d = VariationalAutoEncoder(config.model.hidden_dim)

        self.loss_cl = nn.BCEWithLogitsLoss(reduction='none')

    def get_cl_loss(self, X, Y):
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)
        neg_Y = torch.cat([Y[self.get_neg_index(len(Y), i + 1)]
                           for i in range(self.config.train.cl_neg_samples)], dim=0)
        neg_X = X.repeat((self.config.train.cl_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / self.config.train.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / self.config.train.T

        loss_pos = self.loss_cl(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = self.loss_cl(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = loss_pos + self.config.train.cl_neg_samples * loss_neg

        return CL_loss


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

        loss_cl_2d_3d = self.get_cl_loss(graph_rep_2d, graph_rep_3d)

        loss_cl_3d_2d = self.get_cl_loss(graph_rep_3d, graph_rep_2d)

        loss_ae_2d_3d = self.vae_2d_3d(graph_rep_2d, graph_rep_3d)

        loss_ae_3d_2d = self.vae_3d_2d(graph_rep_3d, graph_rep_2d)

        loss_cl = (loss_cl_2d_3d + loss_ae_3d_2d) / 2

        loss_ae = (loss_ae_2d_3d + loss_ae_3d_2d) / 2

        return loss_cl.mean() + loss_ae.mean()
