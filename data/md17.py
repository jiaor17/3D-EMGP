import torch
from torch_geometric.data import InMemoryDataset, download_url, Data
import numpy as np
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from tqdm import tqdm
import pickle as pkl
import os

class MD17(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"

    molecule_files = dict(
        aspirin="aspirin_dft.npz",
        benzene="benzene2017_dft.npz",
        ethanol="ethanol_dft.npz",
        malonaldehyde="malonaldehyde_dft.npz",
        naphthalene="naphthalene_dft.npz",
        salicylic_acid="salicylic_dft.npz",
        toluene="toluene_dft.npz",
        uracil="uracil_dft.npz",
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(MD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == "all":
            dataset_arg = ",".join(MD17.available_molecules)
        self.molecules = dataset_arg.split(",")

        if len(self.molecules) > 1:
            print(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super(MD17, self).__init__(root, transform, pre_transform)

        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )

    def len(self):
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        data = super(MD17, self).get(idx - self.offsets[data_idx])
        if self.transform:
            return self.transform(data)
        else:
            return data

    @property
    def raw_file_names(self):
        return [MD17.molecule_files[mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f"md17-{mol}.pt" for mol in self.molecules]

    def download(self):
        for file_name in self.raw_file_names:
            download_url(MD17.raw_url + file_name, self.raw_dir)


    def process(self):
        for path in self.raw_paths:
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["z"]).long()
            positions = torch.from_numpy(data_npz["R"]).float()
            energies = torch.from_numpy(data_npz["E"]).float()
            forces = torch.from_numpy(data_npz["F"]).float()

            samples = []
            for pos, y, dy in zip(positions, energies, forces):
                samples.append(Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy))

            if self.pre_filter is not None:
                samples = [data for data in samples if self.pre_filter(data)]

            if self.pre_transform is not None:
                samples = [self.pre_transform(data) for data in tqdm(samples)]

            data, slices = self.collate(samples)
            torch.save((data, slices), self.processed_paths[0])


def get_dataloaders(dataset, num_train, num_val, batch_size, test_batch_size, num_workers, idx_dir):
    idx_file = os.path.join(idx_dir,'idx.pkl')
    if os.path.exists(idx_file):
        with open(idx_file,'rb') as f:
            idx = pkl.load(f)
    else:
        size = len(dataset)
        idx = np.arange(size)
        np.random.shuffle(idx)
        with open(idx_file,'wb') as f:
            pkl.dump(idx, f)
    idx = torch.from_numpy(idx)
    train_idx = idx[:num_train]
    val_idx = idx[num_train:num_train + num_val]
    test_idx = idx[num_train + num_val:]
    train_set = dataset[train_idx]
    val_set = dataset[val_idx]
    test_set = dataset[test_idx]
    train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    val_loader = DataLoader(
            dataset=val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    test_loader = DataLoader(
            dataset=test_set,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    return {
        'train':train_loader,
        'val':val_loader,
        'test':test_loader,
        'idx':idx
    }

def get_mean_std(dataloaders):
    val_loader = dataloaders['val']
    ys = torch.cat([batch.y.squeeze() for batch in val_loader])
    return ys.mean(), ys.std()