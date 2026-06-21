import re
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from datasets import load_dataset

def get_imdb_corpus():
    """
    Loads the IMDB dataset and extracts the text column from all splits (train, test, unsupervised)
    to create a large corpus for tokenizer training.
    
    Returns:
        list[str]: A list of all reviews in the dataset.
    """
    # Load the dataset (this will use the cache if already downloaded)
    dataset = load_dataset("imdb")
    
    corpus = []
    # IMDB dataset typically has 'train', 'test', and 'unsupervised' splits
    for split in dataset.keys():
        corpus.extend(dataset[split]['text'])
        
    return corpus

def summarize_one(name, text, tokens, max_show=20):
    print(f"\n[{name}]")
    print(f"text: {text[:90]}{'...' if len(text) > 90 else ''}")
    print(f"num_tokens: {len(tokens)}")
    print(f"tokens[:{max_show}]: {tokens[:max_show]}")

def corpus_token_lengths(texts, tokenize_fn):
    lengths = [len(tokenize_fn(t)) for t in texts]
    return {
        "avg_len": round(sum(lengths) / len(lengths), 2),
        "median_len": statistics.median(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }


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

def byte_level_tokenizer(text: str) -> List[int]:
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
        if len(tokens) <= 1:
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

def train_wordpiece(corpus: List[str], vocab_size: int = 200, min_freq: int = 2) -> List[str]:
    """
    Train a minimal WordPiece-style vocabulary using a greedy pair-merge score.
    Returns a vocab list including [UNK].
    """
    word_freqs = Counter()
    for text in corpus:
        for word in word_tokenizer(text):
            if word.strip():
                word_freqs[word] += 1

    vocab = {"[UNK]"}
    for word, freq in word_freqs.items():
        if freq < min_freq:
            continue
        if not word:
            continue
        vocab.add(word[0])
        for ch in word[1:]:
            vocab.add("##" + ch)

    splits = {}
    for word, freq in word_freqs.items():
        if freq < min_freq or not word:
            continue
        split = [word[0]] + ["##" + ch for ch in word[1:]]
        splits[word] = split


    pair_freqs = defaultdict(int)
    token_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        if word not in splits: 
            continue
        split = splits[word]
        for tok in split: 
            token_freqs[tok] += freq
        for i in range(len(split) - 1): 
            pair_freqs[(split[i], split[i+1])] += freq
    
    def score(pair:Tuple[str,str]) -> float:
        denom = token_freqs[pair[0]] * token_freqs[pair[1]]
        return pair_freqs[pair]/denom if denom > 0 else 0.0
    
    token_to_words = defaultdict(set)
    for word, split in splits.items():
        for tok in split: 
            token_to_words[tok].add(word)

    while len(vocab) < vocab_size:
        if not pair_freqs: 
            break 

        best_pair = max(pair_freqs, key=score)

        if pair_freqs[best_pair] == 0: 
            break

        a, b = best_pair
        merged = (a + b[2:]) if b.startswith("##") else (a + b)
        vocab.add(merged)

        affected_words = token_to_words[a] & token_to_words[b] 

        for word in affected_words:
            split = splits[word]
            freq = word_freqs[word]

            if not any(split[i] == a and split[i+1] == b for i in range(len(split) - 1)):
                continue

            for tok in split: 
                token_freqs[tok] -= freq
                if token_freqs[tok] <= 0:
                    del token_freqs[tok]

            for i in range(len(split) - 1):
                p = (split[i], split[i+1])
                pair_freqs[p] -= freq
                if pair_freqs[p] <= 0:
                    del pair_freqs[p]
            
            new_split = []
            i = 0 
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i+1] == b:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            splits[word] = new_split

            for tok in new_split:
                token_freqs[tok] += freq 
            for i in range(len(new_split) - 1):
                pair_freqs[(new_split[i], new_split[i+1])] += freq 
            
            for tok in split:
                token_to_words[tok].discard(word)
            for tok in new_split:
                token_to_words[tok].add(word)
    return sorted(vocab)

def word_piece_tokenizer(text: str, vocab: List[str], unk_token: str = "[UNK]") -> List[str]:
    """
    Greedy longest-match WordPiece tokenization with ##-prefix for non-initial pieces.
    """
    vocab_set = set(vocab)
    tokens = []
    for word in word_tokenizer(text):
        if word in vocab_set:
            tokens.append(word)
            continue
        if not word:
            continue
        start = 0
        sub_tokens = []
        is_bad = False
        while start < len(word):
            end = len(word)
            cur_sub = None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in vocab_set:
                    cur_sub = piece
                    break
                end -= 1
            if cur_sub is None:
                is_bad = True
                break
            sub_tokens.append(cur_sub)
            start = end
        if is_bad:
            tokens.append(unk_token)
        else:
            tokens.extend(sub_tokens)
    return tokens
