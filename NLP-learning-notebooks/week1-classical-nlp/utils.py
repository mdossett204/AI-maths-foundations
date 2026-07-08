from typing import Dict, Tuple
from copy import deepcopy
import gc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import scipy.sparse
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


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

def get_newsgroups_ds(remove: Tuple[str] = ('headers', 'footers', 'quotes'), val_size: float = 0.1, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """
    Loads the 20 Newsgroups dataset.
    Splits the original 'train' set into 'train' and 'validation'.
    Returns Train DF, Val DF, Test DF, and the list of class names.
    """
    # Fetch data
    newsgroups_train_full = fetch_20newsgroups(subset='train', remove=remove)
    newsgroups_test = fetch_20newsgroups(subset='test', remove=remove)

    # Create DataFrames
    full_train_df = pd.DataFrame({'text': newsgroups_train_full.data, 'label': newsgroups_train_full.target})
    test_df = pd.DataFrame({'text': newsgroups_test.data, 'label': newsgroups_test.target})
    
    # Split Train into Train/Val
    train_df, val_df = train_test_split(full_train_df, test_size=val_size, random_state=seed, stratify=full_train_df['label'])
    
    return train_df, val_df, test_df, newsgroups_train_full.target_names

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


def save_sklearn_features(X_train, y_train, X_val, y_val, X_test, y_test, base_filename='imdb_sklearn'):
    """
    Saves sparse matrices using scipy and labels using numpy.
    """
    # 1. Save Sparse Matrices (efficient storage for CSR matrices)
    scipy.sparse.save_npz(f"{base_filename}_X_train.npz", X_train)
    scipy.sparse.save_npz(f"{base_filename}_X_val.npz", X_val)
    scipy.sparse.save_npz(f"{base_filename}_X_test.npz", X_test)
    
    # 2. Save Labels (as numpy arrays)
    # We use np.savez to bundle all labels into one file
    # If y_train is a Pandas Series, we save the underlying numpy array
    np.savez(f"{base_filename}_labels.npz", 
             y_train=y_train if isinstance(y_train, np.ndarray) else y_train.values, 
             y_val=y_val if isinstance(y_val, np.ndarray) else y_val.values, 
             y_test=y_test if isinstance(y_test, np.ndarray) else y_test.values)
             
    print(f"Sparse features and labels saved with prefix '{base_filename}'")

def load_sklearn_features(base_filename='imdb_sklearn'):
    """
    Loads the sparse matrices and labels for Day 2.
    """
    # 1. Load Sparse Matrices
    X_train = scipy.sparse.load_npz(f"{base_filename}_X_train.npz")
    X_val = scipy.sparse.load_npz(f"{base_filename}_X_val.npz")
    X_test = scipy.sparse.load_npz(f"{base_filename}_X_test.npz")
    
    # 2. Load Labels
    labels = np.load(f"{base_filename}_labels.npz")
    
    return X_train, labels['y_train'], X_val, labels['y_val'], X_test, labels['y_test']


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    Plots loss and accuracy curves.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Acc')
    plt.plot(epochs, val_accs, 'r-', label='Validation Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

def tune_sklearn_model(model, param_name, param_values, X_train, y_train, X_val, y_val):
    """
    Generic function to tune a single hyperparameter for any Sklearn model.
    """
    best_acc = 0
    best_param = None
    best_model = None
    
    print(f"--- Tuning {model.__class__.__name__} ({param_name}) ---")
    
    for value in param_values:
        # Set the parameter dynamically
        params = {param_name: value}
        model.set_params(**params)
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        
        print(f"{param_name}={value}: Val Acc = {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_param = value
            # We don't deepcopy here to save memory, but in production you might want to
            
    print(f"🏆 Best {param_name}: {best_param} (Acc: {best_acc:.4f})\n")
    return best_param, best_acc

def clean_memory(vars_to_delete: list = None, scope: dict = None, verbose: bool = True):
    """
    Cleans memory. To delete variables, pass a list of names and globals().
    Example: utils.clean_memory(['model', 'X_train'], globals())
    """
    if vars_to_delete and scope:
        for var in vars_to_delete:
            if var in scope:
                del scope[var]
    
    gc.collect()
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    if verbose:
        print("Memory cleaned.")

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def check_early_stop(
        patience:int, 
        val_loss:float, 
        model:nn.Module, 
        best_model_state: Dict, 
        lowest_loss:float, 
        counter: int) -> Tuple:
    early_stop = False
    if val_loss < lowest_loss:
        lowest_loss = val_loss
        best_model_state = deepcopy(model.state_dict())
        counter = 0 
    else:
        counter += 1

    if counter >= patience:
        early_stop = True
    return best_model_state, lowest_loss, counter, early_stop

def get_correct_predictions(logits: torch.tensor, y:torch.tensor, is_binary:bool=False)-> float:
    if is_binary:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    else:
        preds = torch.argmax(logits, dim=1)
    return (preds == y).float().sum().item()

def evaluate_model(
        model: nn.Module, 
        data_loader:DataLoader, 
        criterion: nn.Module, 
        device: torch.device,
        is_binary:bool=False) -> Tuple:
    total_size = len(data_loader.dataset)
    if total_size == 0: return 0.0, 0.0
    model.eval()
    avg_loss = 0 
    accuracy = 0 
    # eval disables dropout/batchnorm behavior; no_grad saves memory and skips gradient computation
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logit = model(x)
            loss = criterion(logit, y)
            avg_loss += loss.item()
            accuracy += get_correct_predictions(logit, y, is_binary)
    avg_loss/=len(data_loader)
    accuracy/=total_size
    return avg_loss, accuracy


def train_test_model(
        device: torch.device, 
        model: nn.Module, 
        criterion:nn.Module,
        train_loader:DataLoader, 
        val_loader:DataLoader, 
        patience:int, 
        num_epochs:int, 
        lr:float, 
        l2:float,
        is_binary:bool=False) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=l2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3) 
    best_model_state = deepcopy(model.state_dict())
    lowest_loss = float("inf")
    counter = 0 
    total_train_size = len(train_loader.dataset)
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(num_epochs):
        model.train()
        avg_train_loss = 0 
        train_accuracy = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
            train_accuracy += get_correct_predictions(logit, y, is_binary)
        avg_train_loss /= len(train_loader)
        train_accuracy /= total_train_size
        train_losses.append(avg_train_loss)
        train_accs.append(train_accuracy)
        avg_val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device, is_binary)
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)
        lr_scheduler.step(avg_val_loss)
        best_model_state, lowest_loss, counter, early_stop = check_early_stop(
            patience, avg_val_loss, model, best_model_state, lowest_loss, counter)
        if epoch % 10 == 0: 
            print(f"Epoch number {epoch} train_loss: {avg_train_loss:.4f} and val_loss: {avg_val_loss:.4f}")
            print(f"Epoch number {epoch} train_accuracy: {train_accuracy:.4f} and val_accuracy: {val_accuracy:.4f}")
        if early_stop:
            print(f"Early stopping at epoch number {epoch}")
            print(f"Train_loss: {avg_train_loss:.4f} and val_loss: {avg_val_loss:.4f}")
            break
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    model.load_state_dict(best_model_state)
    return model 