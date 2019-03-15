# where data loading and preprocessing code should go
from torch.utils.data import DataLoader
from .ExampleDataset import ExampleDataset


def get_train_test_dataloaders(batch_size):
    # create Dataset objects
    train_dataset = None
    val_dataset = None
    test_dataset = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    dataloaders = (train_dataloader, val_dataloader, test_dataloader)
    datasets = (train_dataset, val_dataset, test_dataset)
    return datasets, dataloaders
