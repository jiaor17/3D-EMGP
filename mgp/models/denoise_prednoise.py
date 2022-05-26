import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from scipy.stats import truncnorm
from torch.autograd import grad
import math
import time

loss_func = {
    "L1" : nn.L1Loss(reduction='none'),
    "L2" : nn.MSELoss(reduction='none'),
    "Cosine" : nn.CosineSimilarity(dim=-1, eps=1e-08),
    "CrossEntropy" : nn.CrossEntropyLoss(reduction='none')
}

class EquivariantDenoisePred(torch.nn.Module):

    def __init__(self, config, rep_model):
        super(EquivariantDenoisePred, self).__init__()
        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.edge_types = 0 if config.model.no_edge_types else config.model.order + 1
        self.noise_type = config.model.noise_type
        self.pred_mode = config.model.pred_mode
        self.model = rep_model
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.hidden_dim, self.hidden_dim))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.hidden_dim, 1))

        self.noise_pred = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.hidden_dim, self.config.model.num_noise_level))
        

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)


    def get_energy(self, x, pos, edge_index, edge_attr, node2graph):
        xl, posl = self.model(x, pos, edge_index, edge_attr)
        xl = self.node_dec(xl)
        xg = scatter_add(xl, node2graph, dim = -2)
        e = self.graph_dec(xg)
        return e.squeeze(-1)

    def get_energy_and_rep(self, x, pos, edge_index, edge_attr, node2graph, return_pos = False):
        xl, posl = self.model(x, pos, edge_index, edge_attr)
        xl = self.node_dec(xl)
        xg = scatter_add(xl, node2graph, dim = -2)
        e = self.graph_dec(xg)
        if return_pos:
            return e.squeeze(-1), xg, posl
        return e.squeeze(-1), xg

    @torch.no_grad()
    def get_distance(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data

    @torch.no_grad()
    def truncated_normal(self, size, threshold=1):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        return torch.from_numpy(values)


    @torch.no_grad()
    def get_force_target(self, perturbed_pos, pos, node2graph):
        # s = - (pos_p @ (pos_p.T @ pos_p) - pos @ (pos.T @ pos_p)) / (torch.norm(pos_p.T @ pos_p) + torch.norm(pos.T @ pos_p))
        if self.noise_type == 'riemann':
            v = pos.shape[-1]
            center = scatter_mean(pos, node2graph, dim = -2) # B * 3
            perturbed_center = scatter_mean(perturbed_pos, node2graph, dim = -2) # B * 3
            pos_c = pos - center[node2graph]
            perturbed_pos_c = perturbed_pos - perturbed_center[node2graph]
            perturbed_pos_c_left = perturbed_pos_c.repeat_interleave(v,dim=-1)
            perturbed_pos_c_right = perturbed_pos_c.repeat([1,v])
            pos_c_left = pos_c.repeat_interleave(v,dim=-1)
            ptp = scatter_add(perturbed_pos_c_left * perturbed_pos_c_right, node2graph, dim = -2).reshape(-1,v,v) # B * 3 * 3     
            otp = scatter_add(pos_c_left * perturbed_pos_c_right, node2graph, dim = -2).reshape(-1,v,v) # B * 3 * 3     
            ptp = ptp[node2graph]
            otp = otp[node2graph]
            tar_force = - 2 * (perturbed_pos_c.unsqueeze(1) @ ptp - pos_c.unsqueeze(1) @ otp).squeeze(1) / (torch.norm(ptp,dim=(1,2)) + torch.norm(otp,dim=(1,2))).unsqueeze(-1).repeat([1,3])
            return tar_force
        else:
            return pos - perturbed_pos

    @torch.no_grad()
    def gen_edge_onehot(self, edge_types):
        if not self.edge_types:
            return None
        return F.one_hot(edge_types.long(), self.edge_types)

    @torch.no_grad()
    def fit_pos(self, perturbed_pos, pos, node2graph):
        v = pos.shape[-1]
        center = scatter_mean(pos, node2graph, dim = -2) # B * 3
        perturbed_center = scatter_mean(perturbed_pos, node2graph, dim = -2) # B * 3
        pos_c = pos - center[node2graph]
        perturbed_pos_c = perturbed_pos - perturbed_center[node2graph]
        pos_c = pos_c.repeat([1,v])
        perturbed_pos_c = perturbed_pos_c.repeat_interleave(v,dim=-1)
        H = scatter_add(pos_c * perturbed_pos_c, node2graph, dim = -2).reshape(-1,v,v) # B * 3 * 3
        U, S, V = torch.svd(H)
        # Rotation matrix
        R = V @ U.transpose(2,1)
        t = center - (perturbed_center.unsqueeze(1) @ R.transpose(2,1)).squeeze(1)
        R = R[node2graph]
        t = t[node2graph]
        p_aligned = (perturbed_pos.unsqueeze(1) @ R.transpose(2,1)).squeeze(1) + t
        return p_aligned

    @torch.no_grad()
    def perturb(self, pos, node2graph, used_sigmas, steps=1):
        if self.noise_type == 'riemann':
            pos_p = pos
            for t in range(1, steps + 1):
                alpha = 1 / (2 ** t)
                s = self.get_force_target(pos_p, pos, node2graph)
                pos_p = pos_p + alpha * s + torch.randn_like(pos) * math.sqrt(2 * alpha) * used_sigmas
            return pos_p
        elif self.noise_type == 'kabsch':
            pos_p = pos + torch.randn_like(pos) * used_sigmas
            pos_p = self.fit_pos(pos_p, pos, node2graph)
            return pos_p
        elif self.noise_type == 'gaussian':
            pos_p = pos + torch.randn_like(pos) * used_sigmas
            return pos_p


    def forward(self, data):
        """
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        self.device = self.sigmas.device

        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]
        
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device) # (num_graph)
        used_sigmas = self.sigmas[noise_level] # (num_graph)

        used_sigmas = used_sigmas[node2graph].unsqueeze(-1) # (num_nodes, 1)

        pos = data.pos

        perturbed_pos = self.perturb(pos, node2graph, used_sigmas, self.config.train.steps)

        target = self.get_force_target(perturbed_pos, pos, node2graph) / used_sigmas


        input_pos = perturbed_pos.clone()
        input_pos.requires_grad_(True)
        edge_attr = self.gen_edge_onehot(data.edge_type)

        energy, graph_rep_noise, pred_pos = self.get_energy_and_rep(data.node_feature, input_pos, data.edge_index, edge_attr, node2graph, return_pos = True)


        if self.pred_mode =='energy':
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
            dy = grad(
                    [energy],
                    [input_pos],
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
        
            pred_noise = (-dy).view(-1, 3)
            
        elif self.pred_mode == 'force':

            pred_noise = (pred_pos - perturbed_pos) * (1. / used_sigmas)

        loss_denoise = loss_func['L2'](pred_noise, target)

        loss_denoise = torch.sum(loss_denoise, dim = -1)

        loss_denoise = scatter_add(loss_denoise, node2graph)

        _, graph_rep_ori = self.get_energy_and_rep(data.node_feature, data.pos.clone(), data.edge_index, edge_attr, node2graph)

        graph_rep = torch.cat([graph_rep_ori, graph_rep_noise], dim=1)

        pred_scale = self.noise_pred(graph_rep)

        loss_pred_noise = loss_func['CrossEntropy'](pred_scale, noise_level)

        pred_scale_ = pred_scale.argmax(dim=1)

        return loss_denoise.mean(), loss_pred_noise.mean()
