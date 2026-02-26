# Zero to Hero: NLP Classification & Generation

Welcome to this comprehensive 8-week course designed to guide you from the fundamentals of classical text processing to the cutting edge of modern Large Language Models.

## Course Roadmap

### **Phase 1: The Foundations (Classical NLP)**

#### **Week 1: Classical Machine Learning for Text**

- **Goal:** Establish a strong baseline. Understand how to turn text into numbers without neural networks.
- **Key Concepts:**
  - Vectorization: `CountVectorizer` vs. `TfidfVectorizer`.
  - **Binary Classification:** Logistic Regression on sentiment data.
  - **Multi-class Classification:** Handling 20+ distinct topics with Naive Bayes/SVM.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews (Binary Sentiment).
  - **Demo:** **20 Newsgroups** (Multi-class Topic Classification: Sports, Tech, Politics, etc.).
- **Data Source:** Hugging Face `datasets.load_dataset("imdb")`, `sklearn.datasets.fetch_20newsgroups()`.
- **Tech Stack:** `scikit-learn`, `pandas`.

#### **Week 2: The Mechanics of Meaning (Tokenization)**

- **Goal:** Open the "black box" of tokenization.
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
