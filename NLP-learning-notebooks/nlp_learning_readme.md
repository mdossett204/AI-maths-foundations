# Zero to Hero: NLP Classification & Generation

Welcome to this comprehensive 9-week course designed to guide you from the fundamentals of classical text processing to the cutting edge of modern Large Language Models.

## Course Roadmap

### **Phase 1: The Foundations (Classical NLP)**

#### **Week 1: Classical Machine Learning for Text**

- **Goal:** Establish a strong baseline. Understand how to turn text into numbers and the mechanics of classification (Weights & Gradients).
- **Study Plan:**
  - **Vectorization:** (`classical-ML-vectorization.ipynb`) Loading **IMDB**. Bag of Words (`CountVectorizer`) vs. TF-IDF (`TfidfVectorizer`). Converting Sparse Matrices to PyTorch Tensors.
  - **Binary Classification & Training:** (`binary-classification-pytorch-vs-sklearn.ipynb`) The "Opaque Box" (Sklearn) vs. The "Clear Box" (PyTorch). Implementing the manual training loop (Forward, Loss, Backward) and comparing Logistic Regression, Decision Tree, Random Forest, Gradient Boosting classifier, SVM, and Simple NN.
  - **Multi-class Classification:** (`multiclass-classification-pytorch-vs-sklearn.ipynb`) Loading **20 Newsgroups**. Naive Bayes & Logistic Regression (Sklearn) vs. Softmax Regression in PyTorch with MLP (hidden layer + ReLU + Dropout).
  - **Review:** Digest and Blog.
- **Key Concepts:**
  - **Vectorization:** Bag of Words vs. TF-IDF.
  - **Tensors:** Converting sparse matrices to dense PyTorch tensors.
  - **Classification:** Logistic Regression, SVM, and Naive Bayes vs. Neural Networks.
  - **Training Loop:** Forward pass, loss calculation, and backward pass.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews (Binary Sentiment).
  - **Demo:** **20 Newsgroups** (Multi-class Topic Classification: Sports, Tech, Politics, etc.).
- **Data Source:** Hugging Face `datasets.load_dataset("stanfordnlp/imdb")`, `sklearn.datasets.fetch_20newsgroups()`.
- **Tech Stack:** `scikit-learn`, `pandas`, `datasets`, `torch`.

#### **Week 2: The Mechanics of Meaning (Tokenization)**

- **Goal:** Open the "opaque box" of tokenization.
- **Study Plan:**
  - **Concepts:** Key concepts (Word, Char, Subword, Byte-level). Compare and contrast. Special tokens and multilingual issues.
  - **From Scratch:** Implementation of WordPiece and BPE algorithms **from scratch**.
  - **Modern Usage:** Tying it back to LLMs and modern usage. Integration with Hugging Face `tokenizers`.
- **Key Concepts:**
  - **The OOV Problem:** Why word-level splitting fails on complex languages.
  - **Byte-Level BPE:** How to handle any language (Chinese, Emoji) using bytes.
  - **Implementation:** Writing a WordPiece and BPE algorithm **from scratch**.
- **Datasets:**
  - **Anchor:** IMDB (English).
  - **Demo:** **Multilingual Corpus** (Chinese/Emoji snippets) to demonstrate byte fallback.
- **Data Source:** `datasets.load_dataset("imdb")`, manual snippets.
- **Tech Stack:** Python, Hugging Face `tokenizers`.

---

### **Phase 2: The Deep Learning Bridge**

#### **Week 3: Vector Space Semantics (Embeddings & MLPs)**

- **Goal:** Transition to dense, semantic vectors.
- **Study Plan:**
  - **Tokenizer → IDs → Embedding Inputs:** Reuse Week 2 tokenizer outputs to build vocabulary, IDs, padding, and masks.
  - **Train a Simple Embedding Model (DAN):** Build a Deep Averaging Network (`Embedding → mean pool → MLP → classifier`) for binary sentiment classification on IMDB.
  - **Industry Embeddings comparison:** Compare the custom-trained DAN model with pre-trained word embeddings (GloVe) and sentence embeddings (SBERT) using semantic similarity search and heatmaps.
  - **Real-World Embedding Uses (RAG):** Understand Retrieval-Augmented Generation, explore similarity metrics (Cosine, Euclidean, Jaccard, Dot Product), and perform semantic search.
- **Key Concepts:**
  - **Embeddings:** Dense representations of text.
  - **The `nn.Embedding` layer:** PyTorch implementation of embeddings.
  - **Deep Averaging Networks (DAN):** Mean pooling for sequence representation.
  - **Semantic Similarity Search:** Cosine similarity.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews.
  - **Demo:** Custom sentence snippets for semantic search comparisons.
- **Data Source:** `datasets.load_dataset("imdb")`, pre-trained `glove.6B.300d` vectors.
- **Tech Stack:** `torch`, `sentence-transformers`, `matplotlib`.

#### **Week 4: Sequence Modeling (RNNs)**

- **Goal:** Treat text as a sequence where order matters.
- **Study Plan:**
  - Load the saved Week 3 IMDb tokenized dataset, vocabulary config, and pre-trained `embedding_layer` weights to initialize the sequence models (directly reusing the train, validation, and test dataset splits generated in Week 3).
  - Discuss why order-agnostic baselines (like DAN) can miss sequence information (negation, contrast, sentiment shifts).
  - Build a basic `RNN` sentiment classifier in PyTorch and use it to introduce hidden states over time.
  - Reuse the same sequence pipeline to implement `GRU` and `LSTM`, then compare gating, hidden states, and memory states.
  - Extend the recurrent setup to `BiLSTM` and compare unidirectional versus bidirectional context.
  - Create a small handwritten benchmark set focused on negation, contrast, sentiment shift, and scope.
  - Compare the sequence models on validation performance, benchmark accuracy, inference speed, parameter count, and failure cases.
- **Key Concepts:**
  - Why sequence order matters beyond bag-of-words style features.
  - Transfer learning basics: re-using a pre-trained embedding layer.
  - Recurrence (`RNN`) vs. gating (`LSTM` / `GRU`).
  - Hidden states, memory states, sequence length, and vanishing gradients.
  - Bidirectionality.
  - Behavioral testing with order-sensitive benchmark inputs.
  - Why the training dataset matters: architectural capacity only helps if the training signal rewards order-sensitive reasoning.
  - Why dataset size and parameter count matter: larger recurrent models need enough data and the right supervision to realize their advantage.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews.
  - **Demo:** IMDb-trained models evaluated on a small custom benchmark set covering negation, contrast, sentiment shift, and scope.
- **Data Source:** Saved Week 3 IMDb tokenized dataset splits and vocabulary, plus manual challenge examples.
- **Tech Stack:** `PyTorch`.

#### **Week 5: Spatial Features in Text (CNNs)**

- **Goal:** Using Convolutions for text.
- **Core Concepts:**
  - 1D convolutions as learned sliding-window feature detectors over text.
  - Multiple filters/channels at kernel sizes `6`, `10`, and `16` learning different local phrase patterns.
  - Parameter sharing across positions, so the same learned detector can fire anywhere in a review.
  - Global max pooling versus mean pooling, with emphasis on why max pooling often preserves strong sentiment evidence better.
  - `TextCNN`: `Embedding -> Conv1d -> ReLU -> Global Pool -> Linear`, using the saved IMDb tokenization pipeline and the learned `DAN` embedding from Week 3 as initialization.
  - Training on IMDb directly reusing the train, validation, and test dataset splits generated in Week 3 for consistency.
  - Inspecting learned feature maps and pooling behavior on a small handwritten challenge set targeting negation, intensifiers, and phrase relocation.
  - CNN strengths for local phrase detection and efficient parallel computation, plus a short conceptual contrast with RNNs.
- **Datasets:**
  - **Anchor:** IMDB Movie Reviews.
  - **Demo:** IMDb-trained CNN evaluated on a small custom challenge set targeting local phrase phenomena.
- **Data Source:** Saved Week 3 IMDb tokenized dataset splits and vocabulary, plus manual challenge examples.
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
- **Data Source:** Saved Week 3 IMDb tokenized dataset splits, plus manual toy paired-text examples.
- **Tech Stack:** `PyTorch`, `matplotlib`.

#### **Week 7: Transfer Learning (BERT & SBERT)**

- **Goal:** Extend the Transformer architecture to build an encoder, and explore BERT and Sentence-BERT (SBERT) for transfer learning.
- **Study Plan:**
  - **Building an Encoder:** Extend Week 6's Transformer block into a full Encoder architecture. Understand how multiple layers stack to build deep contextual representations.
  - **BERT Architecture & Pre-training:** Introduce BERT (Bidirectional Encoder Representations from Transformers). Explore its Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) objectives.
  - **Fine-tuning BERT:** Hands-on with fine-tuning a pre-trained BERT model for a classification task using the Hugging Face `Trainer` API.
  - **Sentence-BERT - SBERT:** Introduce SBERT and Siamese networks. Understand why standard BERT is slow for semantic similarity and how SBERT solves this by producing sentence embeddings.
  - **SBERT in Action:** Implement semantic search or retrieval using SBERT embeddings and compare performance with Week 3's baseline.
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

#### **Week 8: Decoder-Only Generation (GPT)**

- **Notebook:** `week8-decoder/Week_8_GPT.ipynb`
- **Goal:** Understand GPT-style causal language modeling, then fine-tune a small pretrained decoder-only model for text completion.
- **Study Plan:**
  - Set up reproducible PyTorch training with CPU, MPS, or CUDA device detection.
  - Build a tiny decoder-only Transformer in pure PyTorch to make the architecture concrete.
  - Implement token embeddings, positional embeddings, stacked Transformer blocks, a causal self-attention mask, and an LM head.
  - Train the tiny model on a self-contained character-level text snippet with next-token prediction loss.
  - Move to Hugging Face `transformers` and fine-tune `distilgpt2` with CPU-friendly defaults.
  - Tokenize and group WikiText into fixed-length causal language modeling blocks.
  - Configure `Trainer` with small batches, gradient accumulation, short sequence length, step-based evaluation, and checkpointing.
  - Run a generation demo from a prompt before or after fine-tuning.
- **Key Concepts:**
  - Decoder-only Transformer architecture.
  - Causal self-attention and future-token masking.
  - Next-token prediction.
  - Language modeling heads and logits over vocabulary.
  - Transfer learning for generation with pretrained GPT models.
- **Datasets:**
  - **Scratch Demo:** Small character-level text snippet.
  - **Fine-tuning Demo:** **WikiText-2** for causal language modeling.
  - **Anchor:** **WikiText-2** for causal language modeling.
  - **Demo:** Small character-level text snippet (Scratch Demo).
- **Data Source:** In-notebook toy text, `datasets.load_dataset("wikitext", "wikitext-2-v1")`.
- **Tech Stack:** `PyTorch`, Hugging Face `datasets`, Hugging Face `transformers`.

#### **Week 9: Encoder-Decoder Generation (T5)**

- **Notebook:** `week9-t5/Week_9_T5.ipynb`
- **Goal:** Learn seq2seq generation through T5's text-to-text framing and fine-tune a small model for English-to-German translation.
- **Study Plan:**
  - Set up reproducible training with the same CPU, MPS, or CUDA device pattern.
  - Compare GPT-style decoder-only generation with T5-style encoder-decoder generation.
  - Build a tiny T5-style encoder-decoder Transformer in pure PyTorch to make the architecture concrete.
  - Review how the T5 encoder reads the full source sequence with bidirectional self-attention.
  - Review how the T5 decoder uses causal self-attention plus cross-attention into encoder states.
  - Train the tiny model on a synthetic reverse-sequence task using shifted-right decoder inputs and teacher forcing.
  - Frame translation as text-to-text generation with the prefix `translate German to English:`.
  - Load `opus_books` German-English examples and create small CPU-friendly train and validation slices.
  - Tokenize source and target text with capped source and target lengths.
  - Fine-tune `t5-small` with `DataCollatorForSeq2Seq`, `Trainer`, gradient accumulation, step-based evaluation, and checkpointing.
  - Run a beam-search translation demo before or after fine-tuning.
- **Key Concepts:**
  - Encoder-decoder Transformer architecture.
  - Text-to-text transfer learning.
  - Bidirectional encoder self-attention.
  - Decoder causal self-attention and encoder-decoder cross-attention.
  - Teacher forcing for seq2seq training.
  - Beam search for translation generation.
- **Datasets:**
  - **Anchor:** **OPUS Books** German-to-English translation.
  - **Demo:** Synthetic reverse-sequence seq2seq task (Scratch Demo).
- **Data Source:** In-notebook synthetic token sequences, `datasets.load_dataset("opus_books", "de-en")`.
- **Tech Stack:** `PyTorch`, Hugging Face `datasets`, Hugging Face `transformers`.
