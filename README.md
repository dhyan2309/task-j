# Generative AI with LSTM – Text Generation

## Project Overview

This project trains a word-level Long Short-Term Memory (LSTM) network on Shakespeare's Complete Works. Given a short seed phrase, the model predicts and samples one word at a time to produce a continuation. The default command downloads the public-domain corpus, preprocesses it, trains and checkpoints the model, saves learning curves, and generates four sample continuations.

## Architecture

```text
Input text → Tokenization → Embedding → LSTM → LSTM → Dense Softmax → Next Word
```

A fixed-length sequence of word IDs is mapped to learned embedding vectors, processed by stacked LSTM layers, then classified with a vocabulary-sized softmax.

## Dataset

Shakespeare's Complete Works, Project Gutenberg:

- Ebook page: [https://www.gutenberg.org/ebooks/100](https://www.gutenberg.org/ebooks/100)
- Plain-text download: [https://www.gutenberg.org/cache/epub/100/pg100.txt](https://www.gutenberg.org/cache/epub/100/pg100.txt)

The application downloads and caches the text as `data/shakespeare.txt` if it is missing or empty.

## Installation

Use Python 3.10+ with a TensorFlow build available for your platform.

```bash
python -m venv venv
```

Windows activation:

```powershell
venv\Scripts\activate
```

macOS/Linux activation:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running

Run the baseline pipeline from the project root:

```bash
python lstm_text_generation.py
```

The model trains for up to 30 epochs and may take substantial time, particularly without a GPU. For a quick low-epoch run, use `python lstm_text_generation.py --epochs 1`.

Run the optional model comparison with:

```bash
python lstm_text_generation.py --run-experiment
```

## Project Structure

```text
lstm-text-generation/
├── data/shakespeare.txt
├── models/best_lstm_model.keras
├── models/tokenizer.json
├── outputs/loss_curve.png
├── outputs/accuracy_curve.png
├── outputs/generated_samples.txt
├── outputs/model_comparison.csv
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── generate.py
├── lstm_text_generation.py
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset, trained checkpoints, tokenizer, and output files are produced locally; the source repository excludes the downloaded dataset and model binaries.

## Model Details

| Setting | Default |
| --- | --- |
| Sequence length | 20 words |
| Vocabulary | Up to 20,000 words; fit only on training text |
| Embedding dimension | 128 |
| LSTM layers | 2, each with 256 units |
| Dropout | 0.2 after each LSTM |
| Output | Dense softmax over the vocabulary |
| Loss | Sparse categorical crossentropy |
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 128 |
| Maximum epochs | 30 |
| Train/validation | Chronological 90% / 10% split |

Integer target labels avoid one-hot encoding every next-word target. EarlyStopping monitors validation loss and restores the best weights; ModelCheckpoint saves the best full Keras model to `models/best_lstm_model.keras`.

## Data Preprocessing

The preprocessing module downloads and caches the Gutenberg text, removes its start/end boilerplate markers when present, lowercases it, replaces punctuation and non-English characters with spaces, normalizes whitespace, and splits into words. A frequency-limited vocabulary is fit on the first 90% of tokens only, with reserved padding and unknown IDs. Sliding windows become inputs and the immediately following word is the target. The final 10% of tokens is held out chronologically for validation.

## Text Generation

`src/generate.py` provides `generate_text(model, tokenizer, seed_text, num_words, temperature)`. It tokenizes the seed, uses the latest model-sized context, predicts a next-word distribution, temperature-scales it, samples a word, and repeats. Low temperature is more predictable; high temperature is more varied and random. The default run generates at least 50 continuation words for each seed:

- `to be or not to be`
- `love is`
- `the king`
- `shall i compare thee`

Actual model-produced samples are saved to `outputs/generated_samples.txt` after training. The exact wording depends on the checkpoint and sampling randomness.

## Results

Training prints training/validation loss and accuracy, best epoch, total epochs, parameter count, and training time. A runtime smoke test using 3,000 corpus tokens, an 8-unit temporary model, and two training batches completed checkpoint saving, plotting, and generation. Its brief excerpt was `to be hue brood kept`; this tiny smoke run is not a baseline quality result. The full command produces the baseline metrics, plots, and four 50-word continuations.

## Bonus Experiment

`python lstm_text_generation.py --run-experiment` trains and compares:

| Model | Sequence length | LSTM layers |
| --- | ---: | ---: |
| A | 20 | 1 |
| B | 50 | 2 |

`outputs/model_comparison.csv` records training time, best epoch, training/validation loss and accuracy, parameter count, context window, and generated text. Quality is subjective, so the file requests human review of the sample rather than claiming an unsupported automatic score. Longer context retains more preceding words; deeper and longer models cost more to train and can overfit.

## Limitations

- The fixed context window cannot preserve arbitrary history.
- LSTMs have difficulty with very long-range dependencies.
- Sampling can repeat phrases or produce locally plausible but incoherent text.
- Large vocabularies increase output-layer cost; rare words may map to the unknown token.
- Training the complete corpus can be expensive, especially on CPU.

## Future Improvements

- Compare Bidirectional LSTM representations (not necessarily causal generation).
- Benchmark GRU and attention-augmented recurrent models.
- Explore Transformer models as a separate architecture comparison.
- Train on larger and more varied public-domain corpora.
- Use subword tokenization to reduce unknown words.

## Interview Questions and Answers

**Why an LSTM instead of a basic RNN?** LSTM gates help retain useful information over longer sequences and mitigate vanishing gradients, though they do not eliminate long-range limitations.

**Why split chronologically?** Randomly mixing neighboring windows can leak nearly identical contexts into validation. Holding out the final segment better simulates predicting later text.

**Why sparse categorical crossentropy?** Each training example has one integer next-word target, so sparse loss avoids a large one-hot target vector.

**What does temperature change?** It rescales next-word probabilities before sampling. Lower values concentrate probability on likely words; higher values flatten the distribution and increase variation.

**How should generated quality be evaluated?** Review coherence, repetition, and seed relevance alongside held-out loss/accuracy. Token accuracy alone does not measure literary quality.
