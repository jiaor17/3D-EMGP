import os
import pickle
import copy
import json
from collections import defaultdict

import numpy as np
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from torch_scatter import scatter
from torch_sparse import SparseTensor

import random

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')


class BatchDatapoint:
    def __init__(self,
                 block_file,
                 n_samples,
                 ):
        self.block_file = block_file
        # deal with the last batch graph numbers.
        self.n_samples = n_samples
        self.datapoints = None

    def load_datapoints(self):
        
        self.datapoints = []

        with open(self.block_file, 'rb') as f:
            dp = pickle.load(f)
            self.datapoints = dp

        assert len(self.datapoints) == self.n_samples

    def shuffle(self):
        pass

    def clean_cache(self):
        del self.datapoints
        self.datapoints = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        assert self.datapoints is not None
        return self.datapoints[idx]

    def is_loaded(self):
        return self.datapoints is not None

class GEOMDataset(Dataset):
    def __init__(self, data,
                 graph_per_file=None, transforms = None):
        self.data = data

        self.len = 0
        for d in self.data:
            self.len += len(d)
        if graph_per_file is not None:
            self.sample_per_file = graph_per_file
        else:
            self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None
        self.transforms = transforms

    def shuffle(self, seed: int = None):
        pass

    def clean_cache(self):
        for d in self.data:
            d.clean_cache()

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx):
        dp_idx = idx // self.sample_per_file
        real_idx = idx % self.sample_per_file
        tar = self.data[dp_idx][real_idx].clone()
        if self.transforms:
            tar = self.transforms(tar)
        return tar

    def load_data(self, idx):
        dp_idx = int(idx / self.sample_per_file)
        if not self.data[dp_idx].is_loaded():
            self.data[dp_idx].load_datapoints()

    def count_loaded_datapoints(self):
        res = 0
        for d in self.data:
            if d.is_loaded():
                res += 1
        return res

def BFS(graph):
    num_node = len(graph.atom_type)
    # edge_set = set()
    edge_dict = {i:[] for i in range(num_node)}

    u_list, v_list = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    for u,v in zip(u_list, v_list):
        if u != v:
            edge_dict[u].append(v)
            edge_dict[v].append(u)
    
    visited_list = []
    unvisited_set = set([i for i in range(num_node)])
    sets = 0
    while len(unvisited_set) > 0:
        u = random.sample(unvisited_set, 1)[0]
        queue = [u]
        sets += 1
        while len(queue):
            u = queue.pop(0)
            if u in visited_list:
                continue
            visited_list.append(u)
            unvisited_set.remove(u)

            for v in edge_dict[u]:
                if v not in visited_list:
                    queue.append(v)
    assert len(visited_list) == num_node
    # return visited_list
    return torch.LongTensor(visited_list)

class GEOMDatasetGPT(GEOMDataset):
    def __init__(self, data=None, graph_per_file=None, transforms = None):
        super(GEOMDatasetGPT, self).__init__(data, graph_per_file, transforms)

    def __getitem__(self, idx):
        # d = self.data[idx]
        dp_idx = idx // self.sample_per_file
        real_idx = idx % self.sample_per_file
        d = self.data[dp_idx][real_idx]
        if not hasattr(d,"order"):
            d.order = BFS(d)
        pred = random.randint(2, len(d.atom_type) - 1)
        given_nodes = d.order[:pred]
        given_edges, given_types = subgraph(given_nodes, d.edge_index, d.edge_type, relabel_nodes=True, num_nodes = len(d.atom_type))
        data = Data(atom_type=d.atom_type[given_nodes], pos=d.pos[given_nodes], edge_index=given_edges, edge_type=given_types, pred=d.atom_type[pred])
        if self.transforms is not None:
            data = self.transforms(data)        
        return data

class GEOMDatasetMVP(GEOMDataset):
    def __init__(self, data=None, graph_per_file=None, transforms = None, mask_ratio = 0.15):
        super(GEOMDatasetMVP, self).__init__(data, graph_per_file, transforms)
        self.mask_ratio = mask_ratio

    def __getitem__(self, idx):
        # d = self.data[idx]
        dp_idx = idx // self.sample_per_file
        real_idx = idx % self.sample_per_file
        d = self.data[dp_idx][real_idx]
        order = BFS(d)
        preserve = max(int(len(d.atom_type) * (1 - self.mask_ratio)), 3)
        given_nodes = order[:preserve]
        given_edges, given_types = subgraph(given_nodes, d.edge_index, d.edge_type, relabel_nodes=True, num_nodes = len(d.atom_type))
        data = Data(atom_type=d.atom_type[given_nodes], pos=d.pos[given_nodes], edge_index=given_edges, edge_type=given_types)
        data_2d = data.clone()
        if self.transforms is not None:
            data = self.transforms(data)        
        return data, data_2d

class GEOMDatasetCL(GEOMDataset):

    def __init__(self, data=None, graph_per_file=None, transforms = None):
        super(GEOMDatasetCL, self).__init__(data, graph_per_file, transforms)
        self.aug_strength = 0.2
        self.aug_prob = None
        self.augmentations = [self.node_drop, self.subgraph,
                              self.edge_pert, self.attr_mask, self.no_aug]

    def set_aug_prob(self, aug_prob):
        self.aug_prob = aug_prob

    def no_aug(self,data):
        d = Data(atom_type=data.atom_type, 
                    pos=data.pos, 
                    edge_index=data.edge_index, 
                    edge_type=data.edge_type)
        return d        

    def node_drop(self, data):

        node_num = data.num_nodes
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num * self.aug_strength)

        idx_perm = np.random.permutation(node_num)
        idx_nodrop = idx_perm[drop_num:].tolist()
        idx_nodrop.sort()

        edge_idx, edge_type = subgraph(subset=idx_nodrop,
                                       edge_index=data.edge_index,
                                       edge_attr=data.edge_type,
                                       relabel_nodes=True,
                                       num_nodes=node_num)

        d = Data(atom_type=data.atom_type[idx_nodrop], 
                    pos=data.pos[idx_nodrop], 
                    edge_index=edge_idx, 
                    edge_type=edge_type)
        return d

    def edge_pert(self, data):
        node_num = data.num_nodes
        _, edge_num = data.edge_index.size()
        pert_num = int(edge_num * self.aug_strength)

        # delete edges
        idx_drop = np.random.choice(edge_num, (edge_num - pert_num),
                                    replace=False)
        edge_index = data.edge_index[:, idx_drop]
        edge_type = data.edge_type[idx_drop]

        # add edges
        adj = torch.ones((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 0
        # edge_index_nonexist = adj.nonzero(as_tuple=False).t()
        edge_index_nonexist = torch.nonzero(adj, as_tuple=False).t()
        idx_add = np.random.choice(edge_index_nonexist.shape[1],
                                   pert_num, replace=False)

        edge_index_add = edge_index_nonexist[:, idx_add]
        edge_type_add = torch.ones(edge_index_add.shape[1])
        edge_index = torch.cat((edge_index, edge_index_add), dim=1)
        edge_type = torch.cat((edge_type, edge_type_add), dim=0)

        d = Data(atom_type=data.atom_type, 
                    pos=data.pos, 
                    edge_index=edge_index, 
                    edge_type=edge_type)
        return d

    def attr_mask(self, data):

        _x = data.atom_type.clone()
        node_num = data.num_nodes
        mask_num = int(node_num * self.aug_strength)

        token = data.atom_type.float().mean(dim=0).long()
        idx_mask = np.random.choice(
            node_num, mask_num, replace=False)

        _x[idx_mask] = token
        d = Data(atom_type=_x, 
            pos=data.pos, 
            edge_index=data.edge_index, 
            edge_type=data.edge_type)
        return d

    def subgraph(self, data):

        order = BFS(data)

        node_num = data.num_nodes

        idx_preserve = max(int(node_num * (1 - self.aug_strength)), 3)

        idx_nondrop = order[:idx_preserve]

        edge_idx, edge_type = subgraph(subset=idx_nondrop,
                                       edge_index=data.edge_index,
                                       edge_attr=data.edge_type,
                                       relabel_nodes=True,
                                       num_nodes=node_num)

        d = Data(atom_type=data.atom_type[idx_nondrop], 
                    pos=data.pos[idx_nondrop], 
                    edge_index=edge_idx, 
                    edge_type=edge_type)
        return d

    def __getitem__(self, idx):
        # d = self.data[idx]
        dp_idx = idx // self.sample_per_file
        real_idx = idx % self.sample_per_file
        d = self.data[dp_idx][real_idx]

        data1, data2 = d.clone(), d.clone()

        if self.aug_prob is None:
            n_aug = np.random.choice(25, 1)[0]
            n_aug1, n_aug2 = n_aug // 5, n_aug % 5
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        else:
            n_aug = np.random.choice(25, 1, p=self.aug_prob)[0]
            n_aug1, n_aug2 = n_aug // 5, n_aug % 5
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        if self.transforms is not None:
            data1 = self.transforms(data1)   
            data2 = self.transforms(data2)
        return data1, data2



class AtomOnehot:

    # global process_input
    def __init__(self,max_atom_type=100, charge_power=2):
        self.max_atom_type = max_atom_type
        self.charge_power = charge_power

    def __call__(self,data):
        atom_type = data.atom_type
        if self.charge_power == -1:
            data.node_feature = atom_type
        else:
            one_hot = F.one_hot(atom_type, self.max_atom_type)
            charge_tensor = (atom_type.unsqueeze(-1) / self.max_atom_type).pow(
                torch.arange(self.charge_power + 1., dtype=torch.float32))
            charge_tensor = charge_tensor.view(atom_type.shape + (1, self.charge_power + 1))
            atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
            data.node_feature = atom_scalars
        return data

class Cutoff:

    # global cutoff, gen_fully_connected
    def __init__(self,cutoff_length=5.0):
        self.cutoff_length = cutoff_length

    def __call__(self,data):
        if not self.cutoff_length:
            ans = 1 - torch.eye(data.pos.shape[0], dtype=torch.int32)
            data.edge_index,_ = dense_to_sparse(ans)
            data.edge_type = torch.zeros([data.edge_index.shape[1]],dtype=torch.long)
            return data
        else:
            cutoff_length = self.cutoff_length
            pos = data.pos # (N,3)
            pos_l, pos_r = pos.unsqueeze(1), pos.unsqueeze(0)
            pairwise_vec = pos_l - pos_r # (N,N,3)
            pairwise_distance = torch.norm(pairwise_vec, p = 2, dim = -1)
            ans = (pairwise_distance <= cutoff_length) * (~torch.eye(pos.shape[0], dtype=torch.bool))
            data.edge_index = torch.nonzero(ans).T
            data.edge_type = torch.zeros([data.edge_index.shape[1]],dtype=torch.long)
            return data
    
class EdgeHop:

    # global gen_graph

    def __init__(self,max_hop=3):
        self.max_hop = max_hop

    def binarize(self,x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):

        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    self.binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]
        for i in range(2, order+1):
            adj_mats.append(self.binarize(adj_mats[i-1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)
        for i in range(1, order+1):
            order_mat += (adj_mats[i] - adj_mats[i-1]) * i
        return order_mat

    def __call__(self,data):
        # print(data.edge_type.max())
        if data.edge_index.shape[1] == 0:
            ans = 1 - torch.eye(data.pos.shape[0], dtype=torch.long)
        else:
            type_mat = to_dense_adj(data.edge_index,max_num_nodes=data.pos.shape[0]).squeeze(0)
            adj_order = self.get_higher_order_adj_matrix(type_mat, self.max_hop)
            # adj_order = type_mat
            fc = 1 - torch.eye(data.pos.shape[0], dtype=torch.long)
            ans = adj_order + fc
        data.edge_index, edge_type = dense_to_sparse(ans)
        data.edge_type = edge_type - 1
        return data