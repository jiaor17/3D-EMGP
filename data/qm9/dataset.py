from torch.utils.data import DataLoader
from data.qm9.data.utils import initialize_datasets
from data.qm9.data.collate import collate_fn
import torch

def retrieve_dataloaders(datadir, batch_size, num_workers=1):
    # Initialize dataloader
    num_train, num_val, num_test, datasets, num_species, charge_scale = initialize_datasets(
        datadir, 'qm9', subtract_thermo=True, force_download=False)
    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                 'lumo': 27.2114}

    print('num train/val/test:', num_train, num_val, num_test)

    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)


    # Construct PyTorch dataloaders from datasets
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn)
                         for split, dataset in datasets.items()}

    return dataloaders, charge_scale



def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


if __name__ == '__main__':
    '''
    dataloader = retrieve_dataloaders(batch_size=64)
    for i, batch in enumerate(dataloader['train']):
        print(i)
    '''
