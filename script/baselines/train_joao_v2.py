#coding: utf-8

import argparse
import numpy as np
import random
import os
import pickle
import yaml
from easydict import EasyDict

import torch
import sys
sys.path.append('.')
from torch_geometric.transforms import Compose
from torch_geometric.data import DataLoader
from mgp import utils, layers, baselines
from torch_geometric.nn import DataParallel
import torch.multiprocessing as mp
import pdb
from time import time

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


from data.dataset import BatchDatapoint, GEOMDatasetCLProj, AtomOnehot, EdgeHop, Cutoff
import json

torch.multiprocessing.set_sharing_strategy('file_system')


def train(rank, config, world_size, verbose=1):

    print('Rank: ',rank)
    if rank != 0:
        verbose = 0


    train_start = time()

    if config.model.no_edge_types:
        transform = Compose([
            Cutoff(cutoff_length=config.model.cutoff),
            AtomOnehot(max_atom_type=config.model.max_atom_type, charge_power=config.model.charge_power),
            ])
    else:
        transform = Compose([
            EdgeHop(max_hop=config.model.order),
            AtomOnehot(max_atom_type=config.model.max_atom_type, charge_power=config.model.charge_power),
            ])

     # set random seed
    np.random.seed(config.train.seed)
    random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.train.seed)        
        torch.cuda.manual_seed_all(config.train.seed)
    torch.backends.cudnn.benchmark = True

    data_dir = config.data.block_dir

    with open(os.path.join(data_dir,'summary.json'),'r') as f:
        summ = json.load(f)
    
    train_block_num = summ['train block num']
    train_block_size = summ['train block size']
    val_block_size = summ['val block size']
    val_block = BatchDatapoint(os.path.join(data_dir,'val_block.pkl'),val_block_size)
    val_block.load_datapoints()
    val_dataset = GEOMDatasetCLProj([val_block],val_block_size, transforms=transform)

    train_blocks = [BatchDatapoint(os.path.join(data_dir,'train_block_%d.pkl'%i),train_block_size) for i in range(train_block_num)]

    for d in train_blocks:
        d.load_datapoints()

    train_dataset = GEOMDatasetCLProj(train_blocks, train_block_size, transforms=transform)

    edge_types = 0 if config.model.no_edge_types else config.model.order + 1
    rep = layers.EGNN_last(in_node_nf=config.model.max_atom_type * (config.model.charge_power + 1),
                                    in_edge_nf=edge_types, hidden_nf=config.model.hidden_dim, n_layers=config.model.n_layers,
                                    attention=config.model.attention, use_layer_norm = config.model.layernorm)
    model = baselines.GraphCLProj(config, rep)

    num_epochs = config.train.epochs
    if world_size == 1:
        dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
                            num_workers=config.train.num_workers, pin_memory = False)
        valloader = DataLoader(val_dataset, batch_size=config.train.batch_size, \
                            shuffle=False, num_workers=config.train.num_workers)
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size,
                                        rank=rank)
        dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size,
                                sampler=train_sampler, num_workers=config.train.num_workers, pin_memory = False)

        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size,
                                        rank=rank, shuffle=False)        

        valloader = DataLoader(val_dataset, batch_size=config.train.batch_size,
                                sampler=val_sampler, num_workers=config.train.num_workers)
  
    model = model.to(rank)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True) 
    optimizer = utils.get_optimizer(config.train.optimizer, model)
    scheduler = utils.get_scheduler(config.train.scheduler, optimizer)
    train_losses = []
    val_losses = []
    ckpt_list = []
    max_ckpt_maintain = 10
    best_loss = 100.0
    start_epoch = 0
    print(f'Rank {rank} start training...')

    aug_probs = torch.FloatTensor(np.ones(25) / 25.).to(rank)
    for epoch in range(num_epochs):
        #train
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_start = time()
        batch_losses = []
        batch_cnt = 0
        if world_size > 1:
            dist.broadcast(aug_probs, src = 0)
        dataloader.dataset.set_aug_prob(aug_probs.detach().cpu().numpy())
        for batch in dataloader:
            batch_cnt += 1
            batch = [_.to(rank) for _ in batch]
            loss = model(batch)
            if not loss.requires_grad:
                raise RuntimeError("loss doesn't require grad")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip)
            if not norm.isnan():
                optimizer.step()
            batch_losses.append(loss.item())

            if verbose and (batch_cnt % config.train.log_interval == 0 or (epoch==0 and batch_cnt <= 10)):
                print('Epoch: %d | Step: %d | loss: %.5f| GradNorm: %.5f | Lr: %.5f' % \
                                    (epoch + start_epoch, batch_cnt, batch_losses[-1], norm.item(), optimizer.param_groups[0]['lr']))


        average_loss = sum(batch_losses) / len(batch_losses)
        train_losses.append(average_loss)

        if verbose:
            print('Epoch: %d | Train Loss: %.5f | Time: %.5f' % (epoch + start_epoch, average_loss, time() - epoch_start))
        
        scheduler.step()

        if world_size > 1:
            dist.barrier()

        if rank == 0:
            
            val_losses.append(average_loss)

            # joao
            aug_prob = aug_probs.detach().cpu().numpy()
            loss_aug = np.zeros(25)
            for n in range(25):
                _aug_prob = np.zeros(25)
                _aug_prob[n] = 1
                valloader.dataset.set_aug_prob(_aug_prob)
                count = 0

                with torch.no_grad():
                    for step, (batch) in enumerate(valloader):
                        batch = [_.to(rank) for _ in batch] 
                        loss = model(batch)
                        loss_aug[n] += loss.item()
                        count += 1
                loss_aug[n] /= count
            beta = 1
            gamma = config.train.gamma_joao

            b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1 / 25))
            mu_min, mu_max = b.min() - 1 / 25, b.max() - 1 / 25
            mu = (mu_min + mu_max) / 2

            # bisection method
            while abs(np.maximum(b - mu, 0).sum() - 1) > 1e-2:
                if np.maximum(b - mu, 0).sum() > 1:
                    mu_min = mu
                else:
                    mu_max = mu
                mu = (mu_min + mu_max) / 2

            aug_prob = np.maximum(b - mu, 0)
            aug_prob /= aug_prob.sum()

            print("Aug Prob: ")
            print(aug_prob.reshape(5,5))

            aug_probs = torch.FloatTensor(aug_prob).to(rank)

            if val_losses[-1] < best_loss:
                best_loss = val_losses[-1]
                if config.train.save:
                    state = {
                        "model": model.state_dict(),
                        "config": config,
                        'cur_epoch': epoch + start_epoch,
                        'best_loss': best_loss,
                    }
                    epoch = str(epoch) if epoch is not None else ''
                    checkpoint = os.path.join(config.train.save_path, 'checkpoint%s' % epoch)

                    if len(ckpt_list) >= max_ckpt_maintain:
                        try:
                            os.remove(ckpt_list[0])
                        except:
                            print('Remove checkpoint failed for', ckpt_list[0])
                        ckpt_list = ckpt_list[1:]
                        ckpt_list.append(checkpoint)
                    else:
                        ckpt_list.append(checkpoint)

                    torch.save(state, checkpoint)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        best_loss = best_loss
        start_epoch = start_epoch + num_epochs               
        print('optimization finished.')
        print('Total time elapsed: %.5fs' % (time() - train_start))
    if world_size > 1:
        dist.destroy_process_group()





def main():

    torch.set_printoptions(profile="full")

    parser = argparse.ArgumentParser(description='mgp')
    parser.add_argument('--config_path', type=str, help='path of dataset', required=True)
    parser.add_argument('--seed', type=int, default=2021, help='overwrite config seed')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    if args.seed != 2021:
        config.train.seed = args.seed

    if config.train.save and config.train.save_path is not None:
        config.train.save_path = os.path.join(config.train.save_path, config.model.name, 'JOAOv2')
        if not os.path.exists(config.train.save_path):
            os.makedirs(config.train.save_path, exist_ok=True)

    print(config)

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')

    if world_size > 1:
        dist.init_process_group('nccl', rank=args.local_rank, world_size=world_size)
    train(args.local_rank, config, world_size)

if __name__ == '__main__':
    main()


