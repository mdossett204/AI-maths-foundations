from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset

def get_imdb_ds(seed: int, train_size: int, val_size: int, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the IMDB dataset, shuffles, and creates Train, Validation, and Test subsets.
    Validation is taken from the original Train split to keep Test pure.
    """
    # Load the dataset (this will use the cache if already downloaded)
    imdb_ds = load_dataset("stanfordnlp/imdb")
    
    # Shuffle train and split into Train and Validation
    shuffled_train = imdb_ds["train"].shuffle(seed=seed)
    train_subset = shuffled_train.select(range(train_size))
    val_subset = shuffled_train.select(range(train_size, train_size + val_size))
    
    # Test subset
    test_subset = imdb_ds["test"].shuffle(seed=seed).select(range(test_size))
    
    return pd.DataFrame(train_subset), pd.DataFrame(val_subset), pd.DataFrame(test_subset)

def create_dataloaders(
    X_train: torch.Tensor, 
    y_train: torch.Tensor, 
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor, 
    y_test: torch.Tensor, 
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wraps tensors into TensorDatasets and creates DataLoaders.
    """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def save_tensors(
    X_train: torch.Tensor, 
    y_train: torch.Tensor, 
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor, 
    y_test: torch.Tensor, 
    filepath: str
):
    """
    Saves the processed tensors to a file.
    """
    torch.save({
        'X_train': X_train, 
        'y_train': y_train, 
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test, 
        'y_test': y_test
    }, filepath)
    print(f"Tensors saved to '{filepath}'")

def load_tensors_and_create_dataloaders(filepath: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads tensors from a file and recreates DataLoaders.
    """
    loaded_data = torch.load(filepath)
    
    X_train = loaded_data['X_train']
    y_train = loaded_data['y_train']
    X_val = loaded_data['X_val']
    y_val = loaded_data['y_val']
    X_test = loaded_data['X_test']
    y_test = loaded_data['y_test']
    
    return create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size)
