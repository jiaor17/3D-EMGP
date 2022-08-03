import os
import pickle
import copy
import json
from collections import defaultdict

import numpy as np
import random


import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_scatter import scatter
from torch_sparse import SparseTensor

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType, BondType
from rdkit import RDLogger
import networkx as nx
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')
import argparse
from multiprocessing import Pool, set_start_method
set_start_method("spawn",force=True)

BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}

def rdmol_to_data(mol:Mol, smiles=None):
    assert mol.GetNumConformers() == 1
    N = mol.GetNumAtoms()

    pos = torch.tensor(mol.GetConformer(0).GetPositions(), dtype=torch.float32)

    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        hybridization = atom.GetHybridization()
        sp.append(1 if hybridization == HybridizationType.SP else 0)
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float32)

    num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)

    data = Data(atom_type=z, pos=pos, edge_index=edge_index, edge_type=edge_type,
                rdmol=copy.deepcopy(mol), smiles=smiles)
    return data

def load_confs_from_filelist(base_path, file_list, conf_per_mol, worker_id):
    bad_case = 0
    res = []
    for i,f in enumerate(file_list):
        if i % 10000 == 0:
            print('worker %d, processed %d files' % (worker_id, i))
        with open(os.path.join(base_path, f), 'rb') as fin:
            mol = pickle.load(fin)        
        if mol.get('uniqueconfs') > len(mol.get('conformers')):
            bad_case += 1
            continue
        if mol.get('uniqueconfs') <= 0:
            bad_case += 1
            continue

        datas = []
        smiles = mol.get('smiles')

        conf_num = min(mol.get('uniqueconfs'), conf_per_mol)
        all_weights = np.array([_.get('boltzmannweight', -1.) for _ in mol.get('conformers')])
        descend_conf_id = (-all_weights).argsort()
        conf_ids = descend_conf_id[:conf_num]

        for conf_id in conf_ids:
            conf_meta = mol.get('conformers')[conf_id]
            data = rdmol_to_data(conf_meta.get('rd_mol'), smiles=smiles)
            labels = {
                'totalenergy': conf_meta['totalenergy'],
                'boltzmannweight': conf_meta['boltzmannweight'],
            }
            for k, v in labels.items():
                data[k] = torch.tensor([v], dtype=torch.float32)
            data['idx'] = torch.tensor([i], dtype=torch.long)
            datas.append(data)
        
        assert len(datas) == conf_num
        res.extend(datas)
    print('worker %d, processed %d files' % (worker_id, len(file_list)))
    return res, bad_case

def idx2list(lis, idx):
    return [lis[_] for _ in idx]

def gen_train_val(base_path, datasets, conf_per_mol=10, val_num = 200, workers = 10, seed=None, test_mask=None):
    # set random seed
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    
    test_mask = test_mask or []

    smiles_list = []
    pickle_path_list = []
    num_mols = 0    
    num_confs = 0   

    # read summary file
    for dataset_name in datasets:
        summary_path = os.path.join(base_path, 'summary_%s.json' % dataset_name)
        with open(summary_path, 'r') as f:
            summ = json.load(f)

        # filter valid pickle path
    
        for smiles, meta_mol in summ.items():
            u_conf = meta_mol.get('uniqueconfs')
            if u_conf is None:
                continue
            pickle_path = meta_mol.get('pickle_path')
            if pickle_path is None:
                continue
            if u_conf < 1:
                continue
            if smiles in test_mask:
                continue
            num_mols += 1
            num_confs += conf_per_mol
            smiles_list.append(smiles)
            pickle_path_list.append(pickle_path)

    print('pre-filter: find %d molecules with %d confs' % (num_mols, num_confs))

    split_indexes = list(range(len(pickle_path_list)))
    random.shuffle(split_indexes)
    val = split_indexes[:val_num]
    train = split_indexes[val_num:]

    if workers == 1:
        print('start processing val data')
        val_data, val_bad = load_confs_from_filelist(base_path, idx2list(pickle_path_list, val), conf_per_mol, 0)
        print('start processing val data')
        train_data, train_bad = load_confs_from_filelist(base_path, idx2list(pickle_path_list, train), conf_per_mol, 0)
    else:
        print('start processing val data')
        val_p = Pool(processes=workers)
        val_f_per_w = int(np.ceil(len(val) / workers))
        val_wks = []
        for i in range(workers):
            tar_idx = val[i * val_f_per_w : (i+1) * val_f_per_w]
            wk = val_p.apply_async(
                load_confs_from_filelist, (base_path, idx2list(pickle_path_list, tar_idx), conf_per_mol, i)
            )
            val_wks.append(wk)
        
        val_p.close()
        val_p.join()
        val_data = []
        val_bad = 0
        for wk in val_wks:
            _val_data, _val_bad = wk.get()
            val_data.extend(_val_data)
            val_bad += _val_bad

        print('start processing train data')
        train_p = Pool(processes=workers)
        train_f_per_w = int(np.ceil(len(train) / workers))
        train_wks = []
        for i in range(workers):
            tar_idx = train[i * train_f_per_w : (i+1) * train_f_per_w]
            wk = train_p.apply_async(
                load_confs_from_filelist, (base_path, idx2list(pickle_path_list, tar_idx), conf_per_mol, i)
            )
            train_wks.append(wk)
        
        train_p.close()
        train_p.join()  
        train_data = []
        train_bad = 0
        for wk in train_wks:
            _train_data, _train_bad = wk.get()
            train_data.extend(_train_data)
            train_bad += _train_bad  

    print('train size: %d molecules with %d confs, %d bad cased filted.' % (len(train) - train_bad, len(train_data), train_bad))
    print('val size: %d molecules with %d confs, %d bad cased filted.' % (len(val) - val_bad, len(val_data), val_bad))

    return train_data, val_data

def gen_GEOM_blocks(base_path, datasets, output_dir, conf_per_mol=10, val_num = 200, train_block_size = 100000, workers = 10, seed=None, test_mask=None):
    train_data, val_data = gen_train_val(base_path, datasets, conf_per_mol, val_num, workers, seed, test_mask)
    if seed is None:
        seed = 2021
    np.random.seed(seed)
    random.seed(seed)
    block_num = len(train_data) // train_block_size + (len(train_data) % train_block_size != 0)
    train_idx = np.arange(block_num * train_block_size)
    np.random.shuffle(train_idx)
    os.makedirs(os.path.join(base_path,'..',output_dir), exist_ok=True)
    for i in range(block_num):
        block_idx = train_idx[i * train_block_size : (i + 1) * train_block_size]
        train_block = idx2list(train_data, block_idx)
        with open(os.path.join(base_path,'..',output_dir,'train_block_%d.pkl'%i), 'wb') as f:
            pickle.dump(train_block, f)

    val_size = len(val_data)
    with open(os.path.join(base_path,'..',output_dir,'val_block.pkl'), 'wb') as f:
        pickle.dump(val_data, f)

    with open(os.path.join(base_path,'..',output_dir,'summary.json'), 'w') as f:
        json.dump({
            'train block num' : block_num,
            'train block size': train_block_size,
            'val block num' : 1,
            'val block size': val_size
        }, f)


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
        # print(idx)
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

example = {'charge','ensembleenergy','ensembleentropy','ensemblefreeenergy','lowestenergy', 'poplowestpct', 'temperature', 'totalconfs', 'uniqueconfs'}

def gen_summary(base_path, pkl_dir):
    files = os.listdir(os.path.join(base_path,pkl_dir))
    tar_dic = {}
    for f in files:
        tk = '.'.join(f.split('.')[:-1])
        tv = {}
        with open(os.path.join(base_path,pkl_dir,f),'rb') as fin:
            p = pickle.load(fin)
        for k,v in p.items():
            if k in example:
                tv[k] = v
        tv['pickle_path'] = os.path.join(pkl_dir, f)
        tar_dic[tk] = tv
    with open(os.path.join(base_path, 'summary_%s.json' % pkl_dir), 'w') as f:
        json.dump(tar_dic, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeomData')
    parser.add_argument('--base_path', type=str, help='path of dataset', required=True)
    parser.add_argument('--datasets', nargs = '+', help='dataset names', required=True)
    parser.add_argument('--output', type=str, help='output dir', required=True)
    parser.add_argument('--val_num', type=int, help='maximum moleculars for validation', default = 1000)
    parser.add_argument('--conf_num', type=int, help='maximum sampled conformations per molecular', default = 10)
    parser.add_argument('--num_workers', type=int, help='workers number', default = 10)
    parser.add_argument('--block_size', type=int, help='conformations per block', default = 100000)
    parser.add_argument('--test_smiles', type=str, help='txt files of smiles for test, split by \\n', default = '')
    args = parser.parse_args()
    if len(args.test_smiles) > 0:
        with open(args.test_smiles, 'r') as f:
            l = f.readlines()
        test_mask = [_.strip() for _ in l]
    else:
        test_mask = None
    gen_GEOM_blocks(args.base_path, args.datasets, args.output, val_num=args.val_num, train_block_size=args.block_size, workers = args.num_workers, conf_per_mol = args.conf_num, test_mask = test_mask)