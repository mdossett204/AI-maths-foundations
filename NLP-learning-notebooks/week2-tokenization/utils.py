import statistics
from typing import List
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


def word_tokenizer(text: str, delimiter:str=" ") -> List[str]:
    """
    split the input text by word boundary, assume delimiter is just white space as default value.
    """
    return text.split(sep=delimiter)

def character_tokenizer(text:str) -> List[str]:
    """
    split the input text by character boundary.
    """
    return list(text)

def word_piece_tokenizer(text: str) -> List[str]:
    pass

def bpe_tokenizer(text: str) -> List[str]:
    pass