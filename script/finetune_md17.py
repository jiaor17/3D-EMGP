import torch
from torch import nn, optim
import argparse
import sys
sys.path.append('.')
import os
from mgp import layers
import yaml
from easydict import EasyDict
from collections import OrderedDict
import random
import numpy as np
import pickle as pkl
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import coalesce
from data.md17 import MD17, get_mean_std, get_dataloaders


parser = argparse.ArgumentParser(description='MD17 Example')
parser.add_argument('--config_path', type=str, default='.', metavar='N',
                    help='Path of config yaml.')
parser.add_argument('--molecule', type=str, default='', metavar='N',
                    help='Molecule to simulate.')
parser.add_argument('--restore_path', type=str, default='', metavar='N',
                    help='Restore path.')
parser.add_argument('--model_name', type=str, default='', metavar='N',
                    help='Model name.')
args = parser.parse_args()

device = torch.device("cuda")
dtype = torch.float32

with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)

if args.molecule != '':
    config.data.molecule = args.molecule

if args.restore_path != '':
    config.train.restore_path = args.restore_path

config.data.base_path = os.path.join(config.data.base_path, config.data.molecule)

if args.model_name != '':
    config.model.name = args.model_name

os.makedirs(config.train.save_path + "/" + config.model.name, exist_ok=True)

# fix seed
seed = config.train.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print('fix seed:', seed)


def load_model(model, model_path):
    state = torch.load(model_path, map_location=device)
    new_dict = OrderedDict()
    for k, v in state['model'].items():
        if k.startswith('module.model.'):
            new_dict[k[13:]] = v
        if k.startswith('model.'):
            new_dict[k[6:]] = v

    model.load_state_dict(new_dict, strict=False)
    return new_dict

loss_l1 = nn.L1Loss()
loss_l2 = nn.MSELoss()


def process_input(atom_type, max_atom_type=10, charge_power=2):
    one_hot = nn.functional.one_hot(atom_type, max_atom_type)
    charge_tensor = (atom_type.unsqueeze(-1) / max_atom_type).pow(
        torch.arange(charge_power + 1., dtype=torch.float32).to(atom_type))
    charge_tensor = charge_tensor.view(atom_type.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(atom_type.shape + (-1,))
    return atom_scalars

def binarize(x):
    return torch.where(x > 0.5, torch.ones_like(x), torch.zeros_like(x))

def get_higher_order_adj_matrix(adj, order):
    """
    Args:
        adj:        (N, N)
        type_mat:   (N, N)
    """

    adj_mats = [torch.eye(adj.size(0), device=adj.device), \
                binarize(adj + torch.eye(adj.size(0), device=adj.device))]
    for i in range(2, order+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    order_mat = torch.zeros_like(adj).float()

    for i in range(1, order+1):
        order_mat += (adj_mats[i] - adj_mats[i-1]) * i
    return order_mat.long()

def add_high_order_edges(adj_mat):
    adj_order = get_higher_order_adj_matrix(adj_mat,3)
    type_highorder = torch.where(adj_order > 1, adj_order, torch.zeros_like(adj_order))
    type_mat = adj_mat
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder
    edge_index, edge_type = dense_to_sparse(type_new)
    return edge_index, edge_type

def gen_adj_matrix(pos):
    # pos : batch * n_nodes * 3
    nodes = pos.shape[0]
    adj = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1) # n * n
    return (adj <= 1.6) & (~torch.eye(nodes).bool())

def gen_fully_connected_with_hop(pos):
    nodes = pos.shape[0]
    adj = torch.norm(pos.unsqueeze(0) - pos.unsqueeze(1), p=2, dim=-1) # n * n
    adj = (adj <= 1.6) & (~torch.eye(nodes).bool())
    adj_order = get_higher_order_adj_matrix(adj,3)
    type_highorder = torch.where(adj_order > 1, adj_order, torch.zeros_like(adj_order))
    fc_mask = torch.ones(nodes,nodes).bool() & (~torch.eye(nodes).bool())
    type_new = adj + type_highorder + fc_mask
    edge_index, edge_type = dense_to_sparse(type_new)
    return edge_index, edge_type - 1

dataset = MD17(config.data.base_path, dataset_arg = config.data.molecule)

# Preprocess the 2D graph features for the first frame, the 2D topological structure does not change in one trajectory.

standard_mol = dataset[0]
standard_edge_index, standard_edge_type = gen_fully_connected_with_hop(standard_mol.pos)
standard_node_fea = process_input(standard_mol.z, max_atom_type=config.model.max_atom_type, charge_power=config.model.charge_power)
def pre_transform(data_ori):
    data = data_ori.clone()
    data.edge_index, data.edge_type, data.input = standard_edge_index, standard_edge_type, standard_node_fea
    return data

dataset.transform = pre_transform

dataloaders = get_dataloaders(dataset, num_train = config.data.num_train, num_val = config.data.num_val, num_workers = config.train.num_workers, batch_size = config.train.batch_size,\
                test_batch_size = config.test.test_batch_size, idx_dir = config.data.base_path)
mean, std = get_mean_std(dataloaders)
print(mean, std)

model = layers.EGNN_md_last(in_node_nf=config.model.max_atom_type * (config.model.charge_power + 1),
                             in_edge_nf=4, hidden_nf=config.model.hidden_dim, n_layers=config.model.n_layers, 
                             attention=config.model.attention, mean=mean, std=std).to(device)

print(model)
print(sum(p.numel() for p in model.parameters()))

if config.train.restore_path:
    encoder_param = load_model(model, config.train.restore_path)
    print('load model from', config.train.restore_path)


optimizer = optim.Adam([param for name, param in model.named_parameters()], lr=config.train.lr, weight_decay=float(config.train.weight_decay))
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.train.factor,
            patience=config.train.patience,
            min_lr=float(config.train.min_lr),
        )

def train(epoch, loader, config, partition='train'):
    # lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[], 'energy': 0, 'energy_arr':[], 'force': 0, 'force_arr':[]}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()
 
        label = data.y.to(device, dtype).squeeze(1)
        dy = data.dy.to(device, dtype)
        nodes = data.input.to(device, dtype)
        edges = data.edge_index.to(device)
        edge_attr = nn.functional.one_hot(data.edge_type, 4).to(device,dtype)
        atom_positions = data.pos.to(device, dtype)
        batch = data.batch.to(device)
        pred, pdy = model(h=nodes, x=atom_positions, edges=edges, edge_attr=edge_attr, batch = batch, md_type = 'gradient')
        batch_size = len(pred)
        if partition == 'train':
            loss_energy = loss_l1(pred, label)
            loss_force = loss_l1(dy, pdy)
            loss = loss_energy * config.train.energy_weight + loss_force * config.train.force_weight
            loss.backward()
            optimizer.step()
        else:
            loss_energy = loss_l1(pred, label)
            loss_force = loss_l1(dy, pdy)
            loss = loss_energy * config.train.energy_weight + loss_force * config.train.force_weight


        res['loss'] += loss.item() * batch_size
        res['energy'] += loss_energy.item() * batch_size
        res['force'] += loss_force.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())
        res['energy_arr'].append(loss_energy.item())
        res['force_arr'].append(loss_force.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % config.train.log_interval == 0:
            print(prefix + 'Epoch: {:4d} | Iter: {:4d} | loss: {:.4f} | energy: {:.4f} | force: {:.4f} | lr: {:.5f}'
                  .format(epoch, i, sum(res['loss_arr'][-10:]) / len(res['loss_arr'][-10:]), sum(res['energy_arr'][-10:]) / len(res['energy_arr'][-10:]),
                          sum(res['force_arr'][-10:]) / len(res['force_arr'][-10:]), optimizer.state_dict()['param_groups'][0]['lr']))
    return res['loss'] / res['counter'], res['energy'] / res['counter'], res['force'] / res['counter']


if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    all_train_loss, all_val_loss, all_test_loss = [], [], []

    for epoch in range(0, config.train.epochs):
        train_loss, train_energy, train_force = train(epoch, dataloaders['train'], config, partition='train')
        all_train_loss.append(train_loss)
        if epoch % config.test.test_interval == 0:
            val_loss, val_energy, val_force = train(epoch, dataloaders['val'], config, partition='valid')
            lr_scheduler.step(val_loss)
            res['epochs'].append(epoch)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_epoch'] = epoch
                state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch
                }
                torch.save(state, config.train.save_path + "/" + config.model.name + "/checkpoint_best.pth")
            print("Val loss: %.4f \t epoch %d" % (val_loss, epoch))
            print("Best: val loss: %.4f \t epoch %d"
                  % (res['best_val'], res['best_epoch']))
            all_val_loss.append(val_loss)
        path = config.train.save_path + "/" + config.model.name
        loss_file = path + '/loss.pkl'

    best_state = torch.load(config.train.save_path + "/" + config.model.name + "/checkpoint_best.pth", map_location=device)
    model.load_state_dict(best_state['model'])

    test_loss, test_energy, test_force = train(epoch, dataloaders['test'], config, partition='test')
    print("Test loss: %.4f \t energy: %.4f \t force %.4f" % (test_loss, test_energy, test_force))
    with open(loss_file, 'wb') as f:
        pkl.dump((all_train_loss, all_val_loss, test_loss, test_energy, test_force), f)


