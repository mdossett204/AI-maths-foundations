import gc
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def save_json(file_name: str, content: Union[Dict, List]) -> None:
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def load_json(file_name: str) -> Union[Dict, List]:
    with open(file_name, "r", encoding="utf-8") as f:
        return json.load(f)


def load_torch_dataset(file_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.load(file_name, weights_only=True)
    padding_mask = data.get("padding_mask", data.get("attention_mask"))
    return data["input_ids"], data["labels"], padding_mask


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


def word_tokenizer(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def byte_level_tokenizer(text: str) -> List[int]:
    return list(text.encode("utf-8"))


def _apply_bpe_to_word(
    word: str,
    merges: List[Tuple[str, str]],
    byte_level: bool = False,
) -> List[str]:
    if byte_level:
        tokens = [str(byte) for byte in byte_level_tokenizer(word)] + ["</w>"]
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


def bpe_tokenizer(
    text: str,
    merges: List[Tuple[str, str]],
    byte_level: bool = True,
) -> List[str]:
    tokens = []
    for word in word_tokenizer(text):
        tokens.extend(_apply_bpe_to_word(word, merges, byte_level=byte_level))
    return tokens


def encode_bpe_sequence(
    text: str,
    bpe_merges: List[Tuple[str, str]],
    bpe_vocab: Dict[str, int],
    max_len: int,
    pad_token: str = "[PAD]",
    byte_level: bool = True,
) -> Tuple[List[int], List[int], List[str]]:
    tokens = bpe_tokenizer(text, bpe_merges, byte_level=byte_level)[:max_len]
    ids = [bpe_vocab[token] for token in tokens]
    mask = [1] * len(ids)
    pad_id = bpe_vocab[pad_token]

    if len(ids) < max_len:
        pad_size = max_len - len(ids)
        ids.extend([pad_id] * pad_size)
        mask.extend([0] * pad_size)
        tokens.extend([pad_token] * pad_size)

    return ids, mask, tokens


def bpe_batch_encode(
    texts: List[str],
    bpe_merges: List[Tuple[str, str]],
    bpe_vocab: Dict[str, int],
    max_len: int,
    pad_token: str = "[PAD]",
    byte_level: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    all_ids = []
    all_masks = []
    all_tokens = []
    for text in texts:
        ids, mask, tokens = encode_bpe_sequence(
            text=text,
            bpe_merges=bpe_merges,
            bpe_vocab=bpe_vocab,
            max_len=max_len,
            pad_token=pad_token,
            byte_level=byte_level,
        )
        all_ids.append(ids)
        all_masks.append(mask)
        all_tokens.append(tokens)
    return (
        torch.tensor(all_ids, dtype=torch.long),
        torch.tensor(all_masks, dtype=torch.long),
        all_tokens,
    )


def sinusoidal_position_encoding(
    seq_len: int,
    d_model: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if d_model % 2 != 0:
        raise ValueError("d_model must be even for sinusoidal position encoding.")

    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: Optional[torch.device] = None,
    base: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even to apply rotary embeddings.")

    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    angles = torch.outer(positions, inv_freq)
    cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.size(-2)
    cos = cos[..., :seq_len, :].to(dtype=x.dtype, device=x.device)
    sin = sin[..., :seq_len, :].to(dtype=x.dtype, device=x.device)
    return (x * cos) + (rotate_half(x) * sin)


def build_causal_mask(
    query_len: int,
    key_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    key_len = key_len if key_len is not None else query_len
    return torch.tril(torch.ones(query_len, key_len, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


def _expand_padding_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(1)


def combine_attention_masks(
    padding_mask: Optional[torch.Tensor] = None,
    causal_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    expanded_padding_mask = _expand_padding_mask(padding_mask)
    if expanded_padding_mask is None:
        return causal_mask
    if causal_mask is None:
        return expanded_padding_mask
    return expanded_padding_mask & causal_mask


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = query.size(-1) ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if attention_mask is not None:
        scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)
    if attention_mask is not None:
        weights = weights.masked_fill(~attention_mask, 0.0)

    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=training)

    output = torch.matmul(weights, value)
    return output, weights


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    return x.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.use_rope = use_rope
        self.is_causal = is_causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        context = x if context is None else context
        key_mask = attention_mask if context_mask is None and context is x else context_mask

        query = _split_heads(self.q_proj(x), self.num_heads)
        key = _split_heads(self.k_proj(context), self.num_heads)
        value = _split_heads(self.v_proj(context), self.num_heads)

        if self.use_rope:
            q_cos, q_sin = build_rope_cache(
                seq_len=query.size(-2),
                head_dim=self.head_dim,
                device=query.device,
            )
            k_cos, k_sin = build_rope_cache(
                seq_len=key.size(-2),
                head_dim=self.head_dim,
                device=key.device,
            )
            query = apply_rope(query, q_cos, q_sin)
            key = apply_rope(key, k_cos, k_sin)

        causal_mask = None
        if self.is_causal:
            causal_mask = build_causal_mask(
                query_len=query.size(-2),
                key_len=key.size(-2),
                device=query.device,
            )

        combined_mask = combine_attention_masks(
            padding_mask=key_mask,
            causal_mask=causal_mask,
        )

        attended, attention_weights = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=combined_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        output = self.out_proj(_merge_heads(attended))

        if return_attention:
            return output, attention_weights
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        enable_cross_attention: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
            is_causal=is_causal,
        )
        self.enable_cross_attention = enable_cross_attention
        if enable_cross_attention:
            self.cross_attn_norm = nn.LayerNorm(d_model)
            self.cross_attn = MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                use_rope=False,
                is_causal=False,
            )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, hidden_dim=mlp_hidden_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        self_attn_input = self.self_attn_norm(x)
        if return_attentions:
            self_attn_output, self_attn_weights = self.self_attn(
                x=self_attn_input,
                attention_mask=attention_mask,
                return_attention=True,
            )
        else:
            self_attn_output = self.self_attn(
                x=self_attn_input,
                attention_mask=attention_mask,
                return_attention=False,
            )
            self_attn_weights = None
        x = x + self_attn_output

        cross_attn_weights = None
        if self.enable_cross_attention and context is not None:
            cross_attn_input = self.cross_attn_norm(x)
            if return_attentions:
                cross_attn_output, cross_attn_weights = self.cross_attn(
                    x=cross_attn_input,
                    attention_mask=None,
                    context=context,
                    context_mask=context_mask,
                    return_attention=True,
                )
            else:
                cross_attn_output = self.cross_attn(
                    x=cross_attn_input,
                    attention_mask=None,
                    context=context,
                    context_mask=context_mask,
                    return_attention=False,
                )
            x = x + cross_attn_output

        x = x + self.ffn(self.ffn_norm(x))

        if return_attentions:
            return x, {"self_attention": self_attn_weights, "cross_attention": cross_attn_weights}
        return x


def masked_mean_pool(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    masked_x = x * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return masked_x.sum(dim=1) / denom


class TinyTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int,
        num_layers: int,
        pad_id: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_absolute_positions: bool = False,
        max_seq_len: int = 512,
        pretrained_embedding_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if pretrained_embedding_weight is not None:
            if pretrained_embedding_weight.shape != (vocab_size, d_model):
                raise ValueError(
                    "pretrained_embedding_weight must have shape "
                    f"({vocab_size}, {d_model}), got {tuple(pretrained_embedding_weight.shape)}."
                )
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding_weight.detach().clone(),
                freeze=False,
                padding_idx=pad_id,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.use_absolute_positions = use_absolute_positions
        self.max_seq_len = max_seq_len

        if use_absolute_positions:
            pe = sinusoidal_position_encoding(seq_len=max_seq_len, d_model=d_model)
            self.register_buffer("absolute_position_encoding", pe, persistent=False)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                    use_rope=use_rope,
                    enable_cross_attention=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        x = self.embedding(input_ids)

        if self.use_absolute_positions:
            pos = self.absolute_position_encoding[: input_ids.size(1)].to(device=x.device, dtype=x.dtype)
            x = x + pos.unsqueeze(0)

        x = self.dropout(x)
        all_attentions = []

        for block in self.blocks:
            if return_attention:
                x, attention_dict = block(
                    x=x,
                    attention_mask=attention_mask,
                    return_attentions=True,
                )
                all_attentions.append(attention_dict["self_attention"])
            else:
                x = block(x=x, attention_mask=attention_mask)

        x = self.final_norm(x)
        pooled = masked_mean_pool(x, attention_mask)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, all_attentions
        return logits


class TinyBERT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int,
        num_layers: int,
        pad_id: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        use_absolute_positions: bool = True,
        max_seq_len: int = 512,
        pretrained_embedding_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if pretrained_embedding_weight is not None:
            if pretrained_embedding_weight.shape != (vocab_size, d_model):
                raise ValueError(
                    "pretrained_embedding_weight must have shape "
                    f"({vocab_size}, {d_model}), got {tuple(pretrained_embedding_weight.shape)}."
                )
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embedding_weight.detach().clone(),
                freeze=False,
                padding_idx=pad_id,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.use_absolute_positions = use_absolute_positions
        self.max_seq_len = max_seq_len

        if use_absolute_positions:
            pe = sinusoidal_position_encoding(seq_len=max_seq_len, d_model=d_model)
            self.register_buffer("absolute_position_encoding", pe, persistent=False)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout,
                    use_rope=use_rope,
                    enable_cross_attention=False,
                    is_causal=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        x = self.embedding(input_ids)

        if self.use_absolute_positions:
            pos = self.absolute_position_encoding[: input_ids.size(1)].to(device=x.device, dtype=x.dtype)
            x = x + pos.unsqueeze(0)

        x = self.dropout(x)
        all_attentions = []

        for block in self.blocks:
            if return_attention:
                x, attention_dict = block(
                    x=x,
                    attention_mask=attention_mask,
                    return_attentions=True,
                )
                all_attentions.append(attention_dict["self_attention"])
            else:
                x = block(x=x, attention_mask=attention_mask)

        x = self.final_norm(x)

        if return_attention:
            return x, all_attentions
        return x


class TinySBERT(nn.Module):
    def __init__(self, bert_model: nn.Module):
        super().__init__()
        self.bert = bert_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.bert(input_ids, attention_mask)
        if isinstance(token_embeddings, tuple):
            token_embeddings = token_embeddings[0]

        # Mean pooling to get sentence embeddings
        sentence_embeddings = masked_mean_pool(token_embeddings, attention_mask)
        
        # L2 Normalization
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def create_bert(**kwargs) -> TinyBERT:
    """
    Creates a BERT model leveraging the existing TransformerBlock.
    Expected kwargs: vocab_size, d_model, num_heads, mlp_hidden_dim, num_layers, pad_id
    """
    return TinyBERT(**kwargs)


def create_sbert(bert_model: nn.Module) -> TinySBERT:
    """
    Wraps a given BERT model to produce sentence embeddings (SBERT) using mean pooling.
    """
    return TinySBERT(bert_model)


def get_correct_predictions(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    return (preds == labels).float().sum().item()


def evaluate_binary_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    total_size = len(data_loader.dataset)
    if total_size == 0:
        return 0.0, 0.0

    model.eval()
    total_loss = 0.0
    total_correct = 0.0

    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.unsqueeze(-1).to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_correct += get_correct_predictions(logits, labels)

    return total_loss / len(data_loader), total_correct / total_size


def check_early_stop(
    patience: int,
    val_loss: float,
    model: nn.Module,
    best_model_state: Dict,
    lowest_loss: float,
    counter: int,
) -> Tuple[Dict, float, int, bool]:
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


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
) -> None:
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_binary_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 2,
    save_model_path: Optional[Union[str, Path]] = None,
) -> nn.Module:
    model.to(device)
    if save_model_path is not None:
        save_model_path = Path(save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_model_state = deepcopy(model.state_dict())
    lowest_loss = float("inf")
    counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    total_train_size = len(train_loader.dataset)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.unsqueeze(-1).to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_correct_predictions(logits, labels)

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_train_size
        val_loss, val_acc = evaluate_binary_classifier(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        previous_lowest_loss = lowest_loss
        best_model_state, lowest_loss, counter, early_stop = check_early_stop(
            patience=patience,
            val_loss=val_loss,
            model=model,
            best_model_state=best_model_state,
            lowest_loss=lowest_loss,
            counter=counter,
        )
        if save_model_path is not None and lowest_loss < previous_lowest_loss:
            torch.save(best_model_state, save_model_path)
            print(f"Saved best model state to {save_model_path}")

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_state)

    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    return model


def _render_attention_matrix(
    matrix: torch.Tensor,
    query_tokens: List[str],
    key_tokens: List[str],
    title: str,
    cmap: str,
    ax: Optional[plt.Axes] = None,
    show_colorbar: bool = True,
) -> None:
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(key_tokens)))
    ax.set_xticklabels(key_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(query_tokens)))
    ax.set_yticklabels(query_tokens)
    ax.set_xlabel("Key / Value tokens")
    ax.set_ylabel("Query tokens")
    ax.set_title(title)
    if show_colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Attention weight")


def plot_attention_heatmap(
    attention: torch.Tensor,
    query_tokens: List[str],
    key_tokens: Optional[List[str]] = None,
    title: str = "",
    head: int = 0,
    batch_idx: int = 0,
    cmap: str = "magma",
) -> None:
    if attention.dim() != 4:
        raise ValueError("attention must have shape [batch, heads, query_len, key_len].")

    matrix = attention[batch_idx, head].detach().cpu()
    key_tokens = query_tokens if key_tokens is None else key_tokens

    plt.figure(figsize=(max(6, 0.6 * len(key_tokens)), max(4, 0.5 * len(query_tokens))))
    _render_attention_matrix(
        matrix=matrix,
        query_tokens=query_tokens,
        key_tokens=key_tokens,
        title=title or f"Attention heatmap (batch {batch_idx}, head {head})",
        cmap=cmap,
    )
    plt.tight_layout()
    plt.show()


def plot_attention_heads(
    attention: torch.Tensor,
    query_tokens: List[str],
    key_tokens: Optional[List[str]] = None,
    max_heads: int = 4,
    batch_idx: int = 0,
    cmap: str = "magma",
) -> None:
    if attention.dim() != 4:
        raise ValueError("attention must have shape [batch, heads, query_len, key_len].")

    key_tokens = query_tokens if key_tokens is None else key_tokens
    num_heads = min(attention.size(1), max_heads)
    fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, max(4, 0.45 * len(query_tokens))))
    if num_heads == 1:
        axes = [axes]

    for head_idx in range(num_heads):
        matrix = attention[batch_idx, head_idx].detach().cpu()
        _render_attention_matrix(
            matrix=matrix,
            query_tokens=query_tokens,
            key_tokens=key_tokens,
            title=f"Head {head_idx}",
            cmap=cmap,
            ax=axes[head_idx],
        )

    plt.tight_layout()
    plt.show()

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two representation matrices [n, d]."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    gram_xy = (X @ Y.T).norm("fro") ** 2
    gram_xx = (X @ X.T).norm("fro")
    gram_yy = (Y @ Y.T).norm("fro")
    return (gram_xy / (gram_xx * gram_yy + 1e-8)).item()