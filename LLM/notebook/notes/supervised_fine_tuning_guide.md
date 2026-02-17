# Supervised Fine-Tuning (SFT) Guide for LLMs

Supervised Fine-Tuning (SFT) is the process of taking a pre-trained Large Language Model (LLM) and training it further on a specific dataset of instruction-response pairs to make it better at following instructions or performing specific tasks.

Below is a step-by-step guide to the SFT lifecycle.

## 1. Select a Pre-trained Base Model

The foundation of SFT is a strong pre-trained model (PTM). These models have learned general language representations from massive corpora (web text, books, code) but may not yet be good at following specific user instructions.

- **Considerations:**
  - **Size:** Choose a parameter count (e.g., 7B, 13B, 70B) that fits your compute budget and deployment constraints.
  - **Architecture:** Models like Llama 3, Mistral, Gemma, or Falcon are popular choices.
  - **License:** Ensure the model's license (e.g., Apache 2.0, MIT, Creative Commons) permits your intended use case.
  - **Context Window:** Ensure the model supports the input length required for your documents.

## 2. Dataset Creation & Curation

This is arguably the most critical step. "Garbage in, garbage out" applies heavily here. You need a dataset of `(prompt, response)` pairs.

- **Quality over Quantity:** A small, high-quality dataset (e.g., 1,000 diverse examples) often outperforms a large, noisy one (e.g., 50,000 generic examples).
- **Diversity:** Ensure varied instructions: summarization, reasoning, creative writing, classification, extraction, etc.
- **Formatting:** Data usually looks like this:
  ```json
  {
    "instruction": "Explain quantum entanglement to a 5-year-old.",
    "input": "",
    "output": "Imagine you have two magic dice..."
  }
  ```
- **Synthetic Data:** It is common to use stronger models (like GPT-4) to generate or refine training data for smaller models (distillation), provided the license allows it.

## 3. Data Preparation & formatting

Before the model can learn, the data must be formatted into a single string that includes special tokens indicating who is speaking.

- **Prompt Templates:** You must define a strict schema. For example, the **ChatML** format or a custom format:
  - _Example:_ `<s>[INST] {instruction} [/INST] {response} </s>`
- **Tokenization:**
  - Convert text to numbers (tokens).
  - Handle padding (making all sequences the same length) and truncation (cutting off text that exceeds the context window).
  - **Masking:** Crucially, during training, we usually **mask the instruction**. We only calculate loss on the _response_ tokens. We want the model to predict the answer, not the question.

## 4. Training (The Fine-Tuning Process)

We split the data into **Training** and **Validation** sets.

- **Loss Function:** Standard **Cross-Entropy Loss**. The model predicts the next token, and we penalize it based on how far its probability distribution is from the actual next token in your dataset.
- **Techniques:**
  - **Full Fine-Tuning:** Updating all model parameters. Expensive and requires massive GPU memory.
  - **PEFT (Parameter-Efficient Fine-Tuning):** The standard for most practitioners. Techniques like **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)** freeze the main model weights and train only a tiny number of adapter layers. This allows fine-tuning a 70B model on a single consumer GPU.
- **Hyperparameters:**
  - **Learning Rate:** Usually much lower than pre-training (e.g., 2e-4 or 1e-5).
  - **Epochs:** Usually low (1-3 epochs) to prevent overfitting.
  - **Batch Size:** Impacted by GPU memory; often improved with Gradient Accumulation.

## 5. Evaluation & Monitoring during Training

- **Training vs. Validation Loss:** Monitor these curves.
  - _Training loss goes down, Validation loss goes up?_ -> Overfitting. Stop early.
  - _Both high?_ -> Underfitting. Increase learning rate or model complexity.
- **Save Checkpoints:** Save the model weights (or LoRA adapters) at regular intervals (steps or epochs) so you can revert to the best version.

## 6. Post-Training Evaluation & Error Analysis

Loss metrics don't tell the whole story. You need semantic evaluation.

- **Human Evaluation:** The gold standard. Manually review a set of test prompts and rate the outputs on correctness, tone, and safety.
- **LLM-as-a-judge:** Use a stronger model (e.g., GPT-4) to grade your fine-tuned model's responses against a rubric.
- **Benchmarks:** Run standard benchmarks (MMLU, TruthfulQA) to ensure you haven't caused "catastrophic forgetting" (where the model forgets basic facts while learning your new task).
- **Error Analysis:** Look for:
  - **Hallucination:** Making up facts.
  - **Repetition:** Getting stuck in loops.
  - **Refusal:** Over-alignment where the model refuses to answer safe questions.

## Summary Workflow

1.  **Base Model:** Llama-3-8B
2.  **Data:** 500 clean Q&A pairs relevant to your domain.
3.  **Format:** Apply chat template -> Tokenize -> Mask user prompts.
4.  **Train:** QLoRA for 2 epochs, learning rate 2e-4.
5.  **Evaluate:** Check validation loss + manually chat with the model.
6.  **Deploy:** Merge LoRA adapters into base model and serve.
