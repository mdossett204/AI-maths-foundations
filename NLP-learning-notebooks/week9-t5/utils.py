from copy import deepcopy
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

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
        attention_mask: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]], Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        context = x if context is None else context

        query = _split_heads(self.q_proj(x), self.num_heads)
        
        # If we are doing cross-attention and have cached encoder K/V
        if context is not x and past_key_value is not None:
            key, value = past_key_value
        else:
            key = _split_heads(self.k_proj(context), self.num_heads)
            value = _split_heads(self.v_proj(context), self.num_heads)
            
            # If we are doing self-attention and have past K/V in cache
            if context is x and past_key_value is not None:
                past_key, past_value = past_key_value
                key = torch.cat([past_key, key], dim=-2)
                value = torch.cat([past_value, value], dim=-2)

        present_key_value = (key, value) if use_cache else None

        attended, attention_weights = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            dropout_p=self.dropout,
            training=self.training,
        )
        output = self.out_proj(_merge_heads(attended))

        if use_cache:
            if return_attention:
                return output, attention_weights, present_key_value
            return output, present_key_value

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


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, hidden_dim=mlp_hidden_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        x = x + self.ffn(self.ffn_norm(x))

        if return_attentions:
            return x, {"self_attention": self_attn_weights}
        return x

class DecoderBlock(nn.Module):
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
            dropout=dropout,
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, hidden_dim=mlp_hidden_dim, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
        past_key_values: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]], Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]:
        self_past = past_key_values.get("self") if past_key_values is not None else None
        cross_past = past_key_values.get("cross") if past_key_values is not None else None

        self_attn_input = self.self_attn_norm(x)
        if use_cache:
            if return_attentions:
                self_attn_output, self_attn_weights, present_self = self.self_attn(
                    x=self_attn_input,
                    attention_mask=tgt_mask,
                    past_key_value=self_past,
                    use_cache=True,
                    return_attention=True,
                )
            else:
                self_attn_output, present_self = self.self_attn(
                    x=self_attn_input,
                    attention_mask=tgt_mask,
                    past_key_value=self_past,
                    use_cache=True,
                    return_attention=False,
                )
                self_attn_weights = None
        else:
            if return_attentions:
                self_attn_output, self_attn_weights = self.self_attn(
                    x=self_attn_input,
                    attention_mask=tgt_mask,
                    return_attention=True,
                )
            else:
                self_attn_output = self.self_attn(
                    x=self_attn_input,
                    attention_mask=tgt_mask,
                    return_attention=False,
                )
                self_attn_weights = None
            present_self = None
            
        x = x + self_attn_output
        cross_attn_input = self.cross_attn_norm(x)
        
        if use_cache:
            if return_attentions:
                cross_attn_output, cross_attn_weights, present_cross = self.cross_attn(
                    x=cross_attn_input,
                    attention_mask=src_mask,
                    context=enc,
                    past_key_value=cross_past,
                    use_cache=True,
                    return_attention=True,
                )
            else:
                cross_attn_output, present_cross = self.cross_attn(
                    x=cross_attn_input,
                    attention_mask=src_mask,
                    context=enc,
                    past_key_value=cross_past,
                    use_cache=True,
                    return_attention=False,
                )
                cross_attn_weights = None
        else:
            if return_attentions:
                cross_attn_output, cross_attn_weights = self.cross_attn(
                    x=cross_attn_input,
                    attention_mask=src_mask,
                    context=enc,
                    return_attention=True,
                )
            else:
                cross_attn_output = self.cross_attn(
                    x=cross_attn_input,
                    attention_mask=src_mask,
                    context=enc,
                    return_attention=False,
                )
                cross_attn_weights = None
            present_cross = None
            
        x = x + cross_attn_output
        x = x + self.ffn(self.ffn_norm(x))

        present_key_values = {"self": present_self, "cross": present_cross} if use_cache else None

        if use_cache:
            if return_attentions:
                return x, {"self_attention": self_attn_weights, "cross_attention": cross_attn_weights}, present_key_values
            return x, present_key_values

        if return_attentions:
            return x, {"self_attention": self_attn_weights, "cross_attention": cross_attn_weights}
        return x

class TinyT5(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        mlp_hidden_dim: int,
        num_layers: int,
        pad_id: int = 0,
        bos_id: int = 1,
        dropout: float = 0.1,
        max_src_len: int = 512,
        max_tgt_len: int = 512,
        pretrained_embedding_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.bos_id = bos_id
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
        self.src_position_embedding = nn.Embedding(max_src_len, d_model)
        self.tgt_position_embedding = nn.Embedding(max_tgt_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
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
    
    def make_src_mask(self, src_ids):
        return (src_ids != self.pad_id).view(src_ids.size(0), 1, 1, src_ids.size(1))
    
    def make_tgt_mask(self, tgt_ids, offset: int = 0):
        batch, seq = tgt_ids.shape
        pad_mask = (tgt_ids != self.pad_id).view(batch, 1, 1, seq)
        if offset > 0:
            q_idx = torch.arange(seq, device=tgt_ids.device).unsqueeze(1)
            k_idx = torch.arange(seq + offset, device=tgt_ids.device).unsqueeze(0)
            causal_mask = k_idx <= (q_idx + offset)
        else:
            causal_mask = torch.tril(torch.ones(seq, seq, device=tgt_ids.device, dtype=torch.bool))
        return pad_mask & causal_mask.view(1, 1, seq, seq + offset)
    
    def shift_right(self, labels):
        bos = torch.full((labels.size(0), 1), self.bos_id, device=labels.device, dtype=labels.dtype)
        return torch.cat([bos, labels[:, :-1]], dim=1)
    
    def encode(self, src_ids):
        _, seq = src_ids.shape
        position = torch.arange(seq, device=src_ids.device).unsqueeze(0)
        x = self.embedding(src_ids) + self.src_position_embedding(position)
        src_mask = self.make_src_mask(src_ids)
        for block in self.encoders:
            x = block(x, src_mask)
        return x, src_mask
    
    def decode(
        self, 
        decoder_ids, 
        enc, 
        src_mask, 
        past_key_values: Optional[List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False
    ):
        _, seq = decoder_ids.shape
        offset = 0
        if past_key_values is not None and past_key_values[0]["self"] is not None:
            offset = past_key_values[0]["self"][0].size(-2)
            
        position = (torch.arange(seq, device=decoder_ids.device) + offset).unsqueeze(0)
        x = self.embedding(decoder_ids) + self.tgt_position_embedding(position)
        
        tgt_mask = self.make_tgt_mask(decoder_ids, offset=offset)
            
        present_key_values = [] if use_cache else None
        for i, block in enumerate(self.decoders):
            block_past = past_key_values[i] if past_key_values is not None else None
            if use_cache:
                x, present_block = block(
                    x, 
                    enc, 
                    tgt_mask, 
                    src_mask, 
                    past_key_values=block_past, 
                    use_cache=True
                )
                present_key_values.append(present_block)
            else:
                x = block(x, enc, tgt_mask, src_mask)
                
        logits = self.lm_head(self.final_norm(x))
        if use_cache:
            return logits, present_key_values
        return logits

    def forward(
        self,
        src_ids: torch.Tensor,
        labels=None,
        past_key_values: Optional[List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]]:
        enc, src_mask = self.encode(src_ids)
        if labels is None:
            raise ValueError("Pass target labels during training so the decoder can be shifted right.")
        decoder_ids = self.shift_right(labels)
        
        if use_cache:
            logits, present_key_values = self.decode(
                decoder_ids, 
                enc, 
                src_mask, 
                past_key_values=past_key_values, 
                use_cache=True
            )
            return logits, present_key_values
            
        logits = self.decode(decoder_ids, enc, src_mask)
        return logits

def make_reverse_batch(batch_size: int, vocab_size: int, src_len: int, EOS: int, device: torch.device):
    src = torch.randint(3, vocab_size, (batch_size, src_len), device=device)
    labels = torch.cat(
        [torch.flip(src, dims=[1]), torch.full((batch_size, 1), EOS, device=device)],
        dim=1,
    )
    return src, labels

def train_tiny_T5(
    model: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size,
    vocab_size,
    src_len,
    EOS,
    pad_id,
    save_model_path: Optional[Union[str, Path]] = None,
) -> nn.Module:
    model.to(device)
    if save_model_path is not None:
        save_model_path = Path(save_model_path)
        save_model_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_model_state = deepcopy(model.state_dict())
    lowest_loss = float("inf")
    for epoch in range(epochs):
        x, y = make_reverse_batch(batch_size, vocab_size, src_len, EOS, device)
        x, y = x.to(device),  y.to(device)
        logits = model(x, labels=y)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=pad_id)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if loss.item() < lowest_loss:
            best_model_state = deepcopy(model.state_dict())
            lowest_loss = loss.item()
            if save_model_path is not None:
                torch.save(best_model_state, save_model_path)
                print(f"Saved best model state to {save_model_path}")
        if epoch % 50 == 0:
            print(f"epoch {epoch+1} | loss {loss.item():.4f}")
            
    model.load_state_dict(best_model_state)
    return model

def evaluate_tiny_T5(
    model: nn.Module,
    batch_size,
    vocab_size,
    src_len,
    EOS,
    device,
) -> Tuple[List[int], List[int], List[int]]:
    model.eval()
    src, labels = make_reverse_batch(batch_size, vocab_size, src_len, EOS, device)
    with torch.no_grad():
        logits = model(src, labels=labels)
        pred = logits.argmax(dim=-1)
    return src[0].tolist(), labels[0].tolist(), pred[0].tolist()
