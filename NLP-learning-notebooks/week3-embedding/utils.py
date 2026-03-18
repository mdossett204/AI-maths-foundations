import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union
import textwrap
import numpy as np
from copy import deepcopy
import gc
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

def save_json(file_name:str, content: Union[Dict, List]) -> None:
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)

def load_json(file_name: str) -> Union[Dict, list]:
    with open(file_name, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content

def _as_long_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(dtype=torch.long)
    return torch.as_tensor(x, dtype=torch.long)

def save_torch_dataset(input_ids:List[List[int]], labels:List[int], 
                       attention_masks:List[List[int]], file_name:str) -> None:
    input_ids = _as_long_tensor(input_ids)
    labels = _as_long_tensor(labels)
    attention_masks = _as_long_tensor(attention_masks)

    torch.save(
        {"input_ids": input_ids,
         "labels": labels,
         "attention_mask":attention_masks,
         }, file_name
    )

def load_torch_dataset(file_name:str) -> Tuple:
    data = torch.load(file_name)
    return data["input_ids"], data["labels"], data["attention_mask"]


def create_vocab_mapping(bpe_merges: List[Tuple[str,str]], pad_token:str="[PAD]") -> Dict[str,int]:
    vocab_id_mapping = {str(i): i for i in range(256)}
    current_id = 256
    for a, b in bpe_merges:
        token = a+b
        if token not in vocab_id_mapping:
            vocab_id_mapping[token] = current_id
            current_id += 1
    vocab_id_mapping[pad_token] = current_id
    return vocab_id_mapping

def convert_text_to_token_ids(text:str, bpe_merges: List[Tuple[str,str]], bpe_vocab: Dict[str,int], max_seq:int, pad_token:str="[PAD]") -> Tuple:
    tokenized_text = bpe_tokenizer(text, bpe_merges, byte_level=True)
    n = len(tokenized_text)
    attention_mask = [1] * n if n <= max_seq else [1]*max_seq
    if n < max_seq:
        tokenized_text.extend([pad_token for _ in range(max_seq-n)])
        attention_mask.extend([0 for _ in range(max_seq-n)])
    token_ids = [bpe_vocab[token] for token in tokenized_text[:max_seq]]
    return token_ids, attention_mask

def get_imdb_corpus():
    """
    Loads the IMDB dataset and extracts text + label from labeled splits (train, test).
    
    Returns:
        tuple[list[str], list[int]]: Texts and corresponding labels.
    """
    # Load the dataset (this will use the cache if already downloaded)
    dataset = load_dataset("imdb")
    
    texts = []
    labels = []
    # Use only labeled splits for training (train/test)
    for split in ("train", "test"):
        texts.extend(dataset[split]["text"])
        labels.extend(dataset[split]["label"])
        
    return texts, labels



def word_tokenizer(text: str) -> List[str]:
    """
    Split input text into words and punctuation tokens.
    Example: "don't stop!" -> ["don", "'", "t", "stop", "!"]
    """
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

def character_tokenizer(text:str) -> List[str]:
    """
    split the input text by character boundary.
    """
    return list(text)

def byte_level_tokenizer(text: str) -> List[str]:
    """
    Return a list of UTF-8 bytes (0-255) representing the input text.
    This is the simplest byte-level tokenizer (no merges).
    """
    return list(text.encode("utf-8"))

def train_bpe(
    corpus: List[str],
    vocab_size: int = 200,
    min_freq: int = 2,
    byte_level: bool = False,
    verbose=False
) -> List[Tuple[str, str]]:
    """
    Train a basic BPE merge list on a word-level corpus.
    Returns a list of merges (pair tuples) in order.
    """
    vocab = Counter()
    for text in corpus:
        for word in word_tokenizer(text):
            if not word.strip():
                continue
            if byte_level:
                # Use byte-level symbols for each word
                symbols = [str(b) for b in byte_level_tokenizer(word)]
            else:
                symbols = character_tokenizer(word)
            vocab[" ".join(symbols) + " </w>"] += 1

    initial_vocab_size = len({tok for w in vocab for tok in w.split()})
    merges = []
    while initial_vocab_size + len(merges) < vocab_size:
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        if pairs[best] < min_freq:
            break

        merges.append(best)
        if verbose:
            print(merges)
        replacement = "".join(best)

        new_vocab = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            i = 0 
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best[0] and symbols[i+1] == best[1]:
                    new_symbols.append(replacement)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_vocab[" ".join(new_symbols)] += freq
        vocab = new_vocab
    return merges

def _apply_bpe_to_word(word: str, merges: List[Tuple[str, str]], byte_level: bool = False) -> List[str]:
    if byte_level:
        tokens = [str(b) for b in byte_level_tokenizer(word)] + ["</w>"]
    else:
        tokens = list(word) + ["</w>"]
    if not merges:
        return [t for t in tokens if t != "</w>"]

    for a, b in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
        if len(tokens) == 2: # one real token plus </w> 
            break

    return [t for t in tokens if t != "</w>"]

def bpe_tokenizer(text: str, merges: List[Tuple[str, str]], byte_level: bool = False) -> List[str]:
    """
    Tokenize text using learned BPE merges (word-level).
    """
    tokens = []
    for word in word_tokenizer(text):
        tokens.extend(_apply_bpe_to_word(word, merges, byte_level=byte_level))
    return tokens

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
        for x, mask, y in data_loader:
            x, mask, y = x.to(device), mask.to(device), y.unsqueeze(-1).to(device)
            logit = model(x,mask)
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
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience) 
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
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.unsqueeze(-1).to(device)
            optimizer.zero_grad()
            logit = model(x, mask)
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
        if epoch % 2 == 0: 
            print(f"Epoch number {epoch} train_loss: {avg_train_loss:.4f} and val_loss: {avg_val_loss:.4f}")
            print(f"Epoch number {epoch} train_accuracy: {train_accuracy:.4f} and val_accuracy: {val_accuracy:.4f}")
        if early_stop:
            print(f"Early stopping at epoch number {epoch}")
            print(f"Train_loss: {avg_train_loss:.4f} and val_loss: {avg_val_loss:.4f}")
            break
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    model.load_state_dict(best_model_state)
    return model  

def bpe_batch_encode(sentences:List[str], 
                     bpe_merges: List[Tuple[str,str]], 
                     bpe_vocab: Dict[str, int], 
                     max_seq:int, 
                     pad_token:str ="[PAD]"):
    input_ids = []
    attention_masks = []
    for text in sentences:
        token_ids, attn = convert_text_to_token_ids(
            text,
            bpe_merges=bpe_merges,
            bpe_vocab=bpe_vocab,
            max_seq=max_seq,
            pad_token=pad_token
        )
        input_ids.append(token_ids)
        attention_masks.append(attn)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long)


def plot_heatmap(sim:torch.Tensor, title: str, labels: List[str]):
    sim = sim.cpu().numpy()
    wrap = lambda s: "\n".join(textwrap.wrap(s, width=18))
    wrapped = [wrap(s) for s in labels]

    plt.figure(figsize=(7,6))
    plt.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(wrapped)), wrapped, rotation=45, ha="right")
    plt.yticks(range(len(wrapped)), wrapped)
    plt.title(title)

    # annotate values
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            plt.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    plt.show()


def get_dan_sent_embed(embedding_layer: nn.Module, 
                       input_ids: torch.tensor, 
                       attention_mask: torch.tensor) -> torch.tensor:
    # input_ids: [B, S]
    # attention_mask: [B, S]
    with torch.no_grad():
        emb = embedding_layer(input_ids)                 # [B, S, D]
        mask = attention_mask.unsqueeze(-1).float()      # [B, S, 1]
        masked = emb * mask
        sent_emb = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [B, D]
    return sent_emb

def load_glove_txt(path:str, dim:int) -> Dict:
    word_to_vec = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = parts[1:]
            if len(vec) != dim:
                continue
            word_to_vec[word] = torch.tensor([float(x) for x in vec])
    return word_to_vec

def get_glove_sent_embed(sentence:str, glove:Dict, dim:int) -> torch.tensor:
    words = sentence.lower().split()
    vecs = [glove[w] for w in words if w in glove]
    if not vecs:
        return torch.zeros(dim)
    return torch.stack(vecs).mean(dim=0)

def get_sbert_embed(sbert: SentenceTransformer, sentences:List[str]) -> torch.tensor:
    sbert_embed = sbert.encode(sentences, convert_to_tensor=True)
    return sbert_embed 

def cos_sim_matrix(x):
    x = F.normalize(x, dim=1)
    return x @ x.T