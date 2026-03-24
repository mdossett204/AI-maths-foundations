# Zero to Hero: NLP Classification & Generation

Welcome to this comprehensive 8-week course designed to guide you from the fundamentals of classical text processing to the cutting edge of modern Large Language Models.

## Course Roadmap

### **Phase 1: The Foundations (Classical NLP)**

#### **Week 1: Classical Machine Learning for Text**

- **Goal:** Establish a strong baseline. Understand how to turn text into numbers and the mechanics of classification (Weights & Gradients).
- **Daily Plan:**
  - **Day 1 (Vectorization):** Loading **IMDB**. Bag of Words (`CountVectorizer`) vs. TF-IDF (`TFIDFVectorizer`). Converting Sparse Matrices to PyTorch Tensors.
  - **Day 2 & 3 (Binary Classification & Training):** The "Black Box" (Sklearn) vs. The "White Box" (PyTorch). Implementing the manual training loop (Forward, Loss, Backward) and comparing Logistic Regression, SVM, and Simple NN.
  - **Day 4-5 (Multi-class Classification):** Loading **20 Newsgroups**. Naive Bayes & Logistic Regression (Sklearn) vs. Softmax Regression (PyTorch).
  - **Day 6-7:** Digest and Blog.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews (Binary Sentiment).
  - **Demo:** **20 Newsgroups** (Multi-class Topic Classification: Sports, Tech, Politics, etc.).
- **Data Source:** Hugging Face `datasets.load_dataset("imdb")`, `sklearn.datasets.fetch_20newsgroups()`.
- **Tech Stack:** `scikit-learn`, `pandas`, `datasets`, `torch`.

#### **Week 2: The Mechanics of Meaning (Tokenization)**

- **Goal:** Open the "black box" of tokenization.
- **Daily Plan:**
  - **Day 1 (Concepts):** Key concepts (Word, Char, Subword, Byte-level). Compare and contrast. Special tokens and multilingual issues.
  - **Day 2 (From Scratch):** Implementation of WordPiece and BPE algorithms **from scratch**.
  - **Day 3 (Modern Usage):** Tying it back to LLMs and modern usage. Integration with Hugging Face `tokenizers`.
- **Key Concepts:**
  - **The OOV Problem:** Why word-level splitting fails on complex languages.
  - **Byte-Level BPE:** How to handle any language (Chinese, Emoji) using bytes.
  - **Implementation:** Writing a BPE algorithm **from scratch**.
- **Datasets:**
  - **Anchor:** IMDB (English).
  - **Demo:** **Multilingual Corpus** (Chinese/Japanese snippets) to demonstrate byte fallback.
- **Data Source:** `datasets.load_dataset("imdb")`, manual snippets.
- **Tech Stack:** Python, Hugging Face `tokenizers`.

---

### **Phase 2: The Deep Learning Bridge**

#### **Week 3: Vector Space Semantics (Embeddings & MLPs)**

- **Goal:** Transition to dense, semantic vectors.
- **Focused 4-Day Plan (single notebook):**
  1. Tokenizer → IDs → padding/masks (use Week 2 tokenizer outputs).
  2. Train a simple embedding model (DAN: `Embedding → mean pool → MLP → classifier`).
  3. Briefly compare with pretrained word embeddings (mention SBERT only, defer depth).
  4. Real‑world usage: similarity search, clustering, retrieval foundations.
- **Key Concepts:**
  - Embeddings: `King - Man + Woman = Queen`.
  - The `nn.Embedding` layer in PyTorch.
  - Deep Averaging Networks (DAN).
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews.
- **Data Source:** `datasets.load_dataset("imdb")`.
- **Tech Stack:** `PyTorch`.

#### **Week 4: Sequence Modeling (RNNs)**

- **Goal:** Treat text as a sequence where order matters.
- **Key Concepts:**
  - Recurrence (RNN) vs. Gating (LSTM/GRU).
  - Bidirectionality.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews.
  - **Demo:** **Sarcasm Detection** (Headlines dataset) - showing where "Bag of Words" fails.
- **Data Source:** `datasets.load_dataset("imdb")`, `datasets.load_dataset("sarcasm")`.
- **Tech Stack:** `PyTorch`.

#### **Week 5: Spatial Features in Text (CNNs)**

- **Goal:** Using Convolutions for text.
- **Key Concepts:**
  - **1D Convolutions:** Acting as N-Gram (trigram/quadgram) detectors.
  - **Feature Extraction:** Detecting "sentiment phrases" regardless of position.
  - Efficiency: CNN speed vs. LSTM slowness.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews (Perfect for n-gram based sentiment).
- **Data Source:** `datasets.load_dataset("imdb")`.
- **Tech Stack:** `PyTorch`.

---

### **Phase 3: The Modern Era (Transformers)**

#### **Week 6: The Attention Revolution**

- **Goal:** Build the Transformer architecture from scratch.
- **Key Concepts:**
  - Self-Attention (Query, Key, Value).
  - The Transformer Block.
  - Positional Encodings.
- **Datasets:**
  - **Anchor:** IMDB (Training a small Transformer encoder from scratch).
- **Data Source:** `datasets.load_dataset("imdb")`.
- **Tech Stack:** `PyTorch`.

#### **Week 7: Transfer Learning (BERTology)**

- **Goal:** Fine-tuning pre-trained encoders.
- **Key Concepts:**
  - **Extractive QA:** Teaching a model to find answers in text.
  - The `Trainer` API.
- **Datasets:**
  - **Anchor:** IMDB (Classification).
  - **Demo:** **SQuAD** (Stanford Question Answering Dataset) - BERT's specialty.
- **Data Source:** `datasets.load_dataset("imdb")`, `datasets.load_dataset("squad")`.
- **Tech Stack:** Hugging Face `transformers`.

#### **Week 8: Generative AI (GPT & T5)**

- **Goal:** Text Generation and Translation.
- **Key Concepts:**
  - **Encoder-Decoder (T5):** Text-to-Text Transfer (Translation).
  - **Decoder-Only (GPT):** Causal Language Modeling (Text Completion).
- **Datasets:**
  - **T5 Demo:** **WMT** (English-to-German Translation).
  - **GPT Demo:** **Shakespeare** or **WikiText** (Creative Generation).
- **Data Source:** `datasets.load_dataset("opus100", "en-de")`, `datasets.load_dataset("wikitext", "wikitext-2-v1")`.
- **Tech Stack:** Hugging Face `transformers`.
