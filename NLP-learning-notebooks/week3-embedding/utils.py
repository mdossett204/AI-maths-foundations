import re
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union
import torch
from datasets import load_dataset

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
