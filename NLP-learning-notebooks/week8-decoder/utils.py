import gc
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


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


def build_causal_mask(
    query_len: int,
    key_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    key_len = key_len if key_len is not None else query_len
    diagonal_offset = key_len - query_len
    return torch.tril(torch.ones(query_len, key_len, dtype=torch.bool, device=device), diagonal=diagonal_offset).unsqueeze(0).unsqueeze(0)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal_mask: torch.Tensor,
    dropout_p: float = 0.0,
    training: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = query.size(-1) ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)
    weights = weights.masked_fill(~causal_mask, 0.0)

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
        dropout: float = 0.0
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        query = _split_heads(self.q_proj(x), self.num_heads)
        key = _split_heads(self.k_proj(x), self.num_heads)
        value = _split_heads(self.v_proj(x), self.num_heads)

        causal_mask = build_causal_mask(
            query_len=query.size(-2),
            key_len=key.size(-2),
            device=query.device,
        )

        attended, attention_weights = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            causal_mask=causal_mask,
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
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, hidden_dim=mlp_hidden_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        self_attn_input = self.self_attn_norm(x)
        if return_attentions:
            self_attn_output, self_attn_weights = self.self_attn(
                x=self_attn_input,
                return_attention=True,
            )
        else:
            self_attn_output = self.self_attn(
                x=self_attn_input,
                return_attention=False,
            )
            self_attn_weights = None
        x = x + self_attn_output

        x = x + self.ffn(self.ffn_norm(x))

        if return_attentions:
            return x, {"self_attention": self_attn_weights}
        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int,
        num_layers: int,
        pad_id: Optional[int],
        dropout: float = 0.1,
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
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight # Weight Tying optimization

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        B, T = input_ids.shape
        if T > self.max_seq_len:
            raise ValueError(f"The input length should be less than max length {self.max_seq_len}")
        position = torch.arange(0, T, device=input_ids.device)
        x = self.embedding(input_ids) + self.position_embedding(position).unsqueeze(0)
        x = self.dropout(x)
        all_attentions = []

        for block in self.blocks:
            if return_attention:
                x, attention_dict = block(
                    x=x,
                    return_attentions=True,
                )
                all_attentions.append(attention_dict["self_attention"])
            else:
                x = block(x=x)

        hidden_states = self.final_norm(x)
        logits = self.lm_head(hidden_states)

        if return_attention:
            return logits, all_attentions
        return logits


def evaluate_tiny_gpt(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens:int,
    max_seq_len: int,
    temperature:float
) -> torch.Tensor:
    model.eval()
    input_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        idx_condition = input_ids[:, -max_seq_len:]
        logits = model(idx_condition)
        logits = logits[:, -1, :]/max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_idx], dim=1)
    return input_ids
   

def train_tiny_gpt(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    save_model_path: Optional[Union[str, Path]] = None,
) -> nn.Module:
    model.to(device)
    if save_model_path is not None:
        save_model_path = Path(save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_model_state = deepcopy(model.state_dict())
    lowest_loss = float("inf")
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device),  y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if loss.item() < lowest_loss:
                best_model_state = deepcopy(model.state_dict())
                lowest_loss = loss.item()
                if save_model_path is not None:
                    torch.save(best_model_state, save_model_path)
                    print(f"Saved best model state to {save_model_path}")
        if epoch % 10 == 0:
            print(f"epoch {epoch+1} | loss {loss.item():.4f}")
            
    model.load_state_dict(best_model_state)
    return model
