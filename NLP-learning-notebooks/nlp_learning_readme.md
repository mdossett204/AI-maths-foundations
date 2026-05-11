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
- **Study Plan:**
  - Load the saved Week 3 IMDb tokenized dataset, vocabulary config, and `DAN` weights as the Week 4 baseline setup.
  - Revisit `DAN + linear classifier` as the order-agnostic embedding baseline.
  - Build a basic `RNN` sentiment classifier in PyTorch and use it to introduce hidden states over time.
  - Reuse the same sequence pipeline to implement `GRU` and `LSTM`, then compare gating, hidden states, and memory states.
  - Extend the recurrent setup to `BiLSTM` and compare unidirectional versus bidirectional context.
  - Create a small handwritten benchmark set focused on negation, contrast, sentiment shift, and scope.
  - Compare all models on validation performance, benchmark accuracy, inference speed, parameter count, and failure cases.
- **Key Concepts:**
  - Why sequence order matters beyond bag-of-words style features.
  - Order-agnostic embedding baseline: `DAN + linear classifier`.
  - Recurrence (`RNN`) vs. gating (`LSTM` / `GRU`).
  - Hidden states, memory states, sequence length, and vanishing gradients.
  - Bidirectionality.
  - Behavioral testing with order-sensitive benchmark inputs.
  - Why the training dataset matters: architectural capacity only helps if the training signal rewards order-sensitive reasoning.
  - Why dataset size and parameter count matter: larger recurrent models need enough data and the right supervision to realize their advantage.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews.
  - **Demo:** IMDb-trained models evaluated on a small custom benchmark set covering negation, contrast, sentiment shift, and scope.
- **Data Source:** `datasets.load_dataset("imdb")`, manual challenge examples.
- **Tech Stack:** `PyTorch`.

#### **Week 5: Spatial Features in Text (CNNs)**

- **Goal:** Using Convolutions for text.
- **Core Concepts:**
  - 1D convolutions as learned sliding-window feature detectors over text.
  - Multiple filters/channels at kernel sizes `2`, `3`, and `4` learning different local phrase patterns.
  - Parameter sharing across positions, so the same learned detector can fire anywhere in a review.
  - Global max pooling versus mean pooling, with emphasis on why max pooling often preserves strong sentiment evidence better.
  - `TextCNN`: `Embedding -> Conv1d -> ReLU -> Global Pool -> Linear`, using the saved IMDb tokenization pipeline and the learned `DAN` embedding from Week 3 as initialization.
  - Training on IMDb using the same split setup as Week 4 for consistency.
  - Inspecting learned feature maps and pooling behavior on a small handwritten challenge set targeting negation, intensifiers, and phrase relocation.
  - CNN strengths for local phrase detection and efficient parallel computation, plus a short conceptual contrast with RNNs.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews.
  - **Demo:** IMDb-trained CNN evaluated on a small custom challenge set targeting local phrase phenomena.
- **Data Source:** Saved Week 3/Week 4 IMDb tokenized assets, plus manual challenge examples.
- **Tech Stack:** `PyTorch`.

---

### **Phase 3: The Modern Era (Transformers)**

#### **Week 6: The Attention Revolution**

- **Goal:** Build attention from scratch, then assemble a small Transformer block with modern positional handling.
- **Study Plan:**
  - Reuse the saved IMDb tokenized dataset, vocabulary config, and the same train/validation/test split pattern from Weeks 4 and 5 so the architecture comparison stays grounded.
  - Start with scaled dot-product self-attention and build intuition for `Query`, `Key`, `Value`, attention scores, and attention weights.
  - Extend the single-head version into multi-head attention and explain why different heads can focus on different relationships.
  - Introduce the original Transformer-style positional encoding added after the embedding layer, then contrast it with rotary positional encoding (`RoPE`).
  - Show why `RoPE` is widely used in modern Transformer and LLM architectures: position is injected directly inside each attention block by rotating query/key vectors, which preserves vector geometry better and makes attention scores more naturally sensitive to **relative** position.
  - Build a pre-norm Transformer block with residual connections, self-attention, feed-forward layers, and layer normalization.
  - Add cross-attention as an extension so the block can support both encoder-style self-attention and encoder-decoder style attention.
  - Visualize attention maps in color to inspect what tokens each head is focusing on.
  - Use a tiny handwritten paired-text challenge set, similar in spirit to the Week 4 and Week 5 benchmark probes, to demonstrate cross-attention behavior without introducing a full seq2seq dataset yet.
- **Key Concepts:**
  - Scaled dot-product self-attention: `Q`, `K`, `V`, masking, and attention weights.
  - Multi-head attention.
  - Traditional positional encoding after the embedding layer.
  - Rotary positional encoding (`RoPE`) inside attention blocks.
  - Why matrix rotation is a better modern positional strategy than only adding position vectors once at the input.
  - Residual connections, layer normalization, and feed-forward sublayers inside a Transformer block.
  - Self-attention vs. cross-attention.
  - Attention heatmaps for interpretation.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews.
  - **Demo:** Small handwritten paired-text attention set for cross-attention probes such as review snippet + query.
- **Data Source:** Saved Week 4/Week 5 IMDb tokenized assets, plus manual toy paired-text examples.
- **Tech Stack:** `PyTorch`, `matplotlib`.

#### **Week 7: Transfer Learning (BERT & SBERT)**

- **Goal:** Extend the Transformer architecture to build an encoder, and explore BERT and Sentence-BERT (SBERT) for transfer learning.
- **Daily Plan:**
  - **Day 1 (Building an Encoder):** Extend Week 6's Transformer block into a full Encoder architecture. Understand how multiple layers stack to build deep contextual representations.
  - **Day 2 (BERT Architecture & Pre-training):** Introduce BERT (Bidirectional Encoder Representations from Transformers). Explore its Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) objectives.
  - **Day 3 (Fine-tuning BERT):** Hands-on with fine-tuning a pre-trained BERT model for a classification task using the Hugging Face `Trainer` API.
  - **Day 4 (Sentence-BERT - SBERT):** Introduce SBERT and Siamese networks. Understand why standard BERT is slow for semantic similarity and how SBERT solves this by producing sentence embeddings.
  - **Day 5 (SBERT in Action):** Implement semantic search or retrieval using SBERT embeddings and compare performance with Week 3's baseline.
- **Key Concepts:**
  - Transformer Encoders.
  - BERT Pre-training (MLM, NSP) and Fine-tuning.
  - Sentence Embeddings vs. Token Embeddings.
  - Siamese Network Architecture (SBERT).
  - Semantic Similarity and Retrieval.
- **Datasets:**
  - **Anchor:** IMDB (Classification).
  - **Demo:** STS (Semantic Textual Similarity) dataset for SBERT.
- **Data Source:** Hugging Face `datasets`.
- **Tech Stack:** `PyTorch`, Hugging Face `transformers`, `sentence-transformers`.

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
