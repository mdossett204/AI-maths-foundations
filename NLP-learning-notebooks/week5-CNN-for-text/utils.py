import gc
import json
import re
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def save_json(file_name: str, content: Union[Dict, List]) -> None:
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


def load_torch_dataset(file_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(file_name, weights_only=True)
    padding_mask = data.get("padding_mask", data.get("attention_mask"))
    return data["input_ids"], data["labels"], padding_mask


def word_tokenizer(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def byte_level_tokenizer(text: str) -> List[int]:
    return list(text.encode("utf-8"))


def _apply_bpe_to_word(word: str, merges: List[Tuple[str, str]], byte_level: bool = False) -> List[str]:
    if byte_level:
        tokens = [str(b) for b in byte_level_tokenizer(word)] + ["</w>"]
    else:
        tokens = list(word) + ["</w>"]

    if not merges:
        return [token for token in tokens if token != "</w>"]

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
        if len(tokens) == 2:
            break

    return [token for token in tokens if token != "</w>"]


def bpe_tokenizer(text: str, merges: List[Tuple[str, str]], byte_level: bool = False) -> List[str]:
    tokens = []
    for word in word_tokenizer(text):
        tokens.extend(_apply_bpe_to_word(word, merges, byte_level=byte_level))
    return tokens


def convert_text_to_token_ids(
    text: str,
    bpe_merges: List[Tuple[str, str]],
    bpe_vocab: Dict[str, int],
    max_seq: int,
    pad_token: str = "[PAD]",
) -> Tuple[List[int], List[int]]:
    tokenized_text = bpe_tokenizer(text, bpe_merges, byte_level=True)
    n = len(tokenized_text)
    padding_mask = [1] * min(n, max_seq)
    if n < max_seq:
        tokenized_text.extend([pad_token] * (max_seq - n))
        padding_mask.extend([0] * (max_seq - n))
    token_ids = [bpe_vocab[token] for token in tokenized_text[:max_seq]]
    return token_ids, padding_mask


def bpe_batch_encode(
    sentences: List[str],
    bpe_merges: List[Tuple[str, str]],
    bpe_vocab: Dict[str, int],
    max_seq: int,
    pad_token: str = "[PAD]",
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = []
    padding_masks = []
    for text in sentences:
        token_ids, padding_mask = convert_text_to_token_ids(
            text=text,
            bpe_merges=bpe_merges,
            bpe_vocab=bpe_vocab,
            max_seq=max_seq,
            pad_token=pad_token,
        )
        input_ids.append(token_ids)
        padding_masks.append(padding_mask)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(padding_masks, dtype=torch.long)


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, "b-", label="Training Acc")
    plt.plot(epochs, val_accs, "r-", label="Validation Acc")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def clean_memory(vars_to_delete: list = None, scope: dict = None, verbose: bool = True):
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
    patience: int,
    val_loss: float,
    model: nn.Module,
    best_model_state: Dict,
    lowest_loss: float,
    counter: int,
) -> Tuple:
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


def get_correct_predictions(logits: torch.Tensor, y: torch.Tensor, is_binary: bool = False) -> float:
    if is_binary:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    else:
        preds = torch.argmax(logits, dim=1)
    return (preds == y).float().sum().item()


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_binary: bool = False,
) -> Tuple[float, float]:
    total_size = len(data_loader.dataset)
    if total_size == 0:
        return 0.0, 0.0
    model.eval()
    avg_loss = 0.0
    accuracy = 0.0
    with torch.no_grad():
        for x, mask, y in data_loader:
            x, mask, y = x.to(device), mask.to(device), y.unsqueeze(-1).to(device)
            logit = model(x, mask)
            loss = criterion(logit, y)
            avg_loss += loss.item()
            accuracy += get_correct_predictions(logit, y, is_binary)
    avg_loss /= len(data_loader)
    accuracy /= total_size
    return avg_loss, accuracy


def train_test_model(
    device: torch.device,
    model: nn.Module,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    patience: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    is_binary: bool = False,
    save_path: str = None,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience
    )
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
        avg_train_loss = 0.0
        train_accuracy = 0.0

        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.unsqueeze(-1).to(device)
            optimizer.zero_grad()
            logit = model(x, mask)
            loss = criterion(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            patience, avg_val_loss, model, best_model_state, lowest_loss, counter
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f}"
        )

        if save_path is not None:
            torch.save(best_model_state, save_path)

        if early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    model.load_state_dict(best_model_state)
    return model


def load_pretrained_embedding(model: nn.Module, state_path: str, key: str = "embedding.weight") -> None:
    state = torch.load(state_path, map_location="cpu")
    if key not in state:
        raise KeyError(f"Could not find '{key}' in {state_path}")
    model.embedding.weight.data.copy_(state[key])


def build_conv_mask(attention_mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    window_mass = F.avg_pool1d(
        attention_mask.float().unsqueeze(1), kernel_size=kernel_size, stride=1
    ).squeeze(1)
    return window_mass == 1.0


def predict_probabilities(model: nn.Module, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, mask)
        return torch.sigmoid(logits).squeeze(-1)


def plot_feature_map_heatmap(
    feature_map: torch.Tensor,
    kernel_size: int,
    title: str = "",
    max_filters: int = 12,
) -> None:
    tensor = feature_map.detach().cpu()
    if tensor.dim() != 2:
        raise ValueError("feature_map must have shape [num_filters, num_windows]")
    tensor = tensor[:max_filters]

    plt.figure(figsize=(12, max(4, 0.5 * tensor.shape[0])))
    plt.imshow(tensor, aspect="auto", cmap="viridis")
    plt.colorbar(label="Activation")
    plt.xlabel("Sliding-window position")
    plt.ylabel("Filter index")
    plt.title(title or f"Kernel size {kernel_size} feature map")
    plt.tight_layout()
    plt.show()

