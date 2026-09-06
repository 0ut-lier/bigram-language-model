Bigram Language Model from Scratch

A simple character-level bigram language model built with Python and PyTorch to understand the mathematical foundations of language modeling.

The model learns the probability of the next character given the current character and uses those probabilities to generate new names.

✨ What This Project Covers
Character-level tokenization
Vocabulary and character encoding
Bigram frequency counting
Conditional probability
Maximum Likelihood Estimation (MLE)
Laplace smoothing
Probability distributions
Random sampling
Log-likelihood
Negative Log-Likelihood (NLL)
Cross-entropy
Basic language-model evaluation
🧠 How It Works

For every word, special boundary tokens are added:

emma → .emma.


This produces bigrams:

.e
em
mm
ma
a.


The model counts how often each character is followed by another character.

These counts are stored in a matrix:

𝑁
𝑖
𝑗
=
count of character 
𝑗
 following character 
𝑖

The counts are converted into probabilities:

𝑃
(
𝑗
∣
𝑖
)
=
𝑁
𝑖
𝑗
∑
𝑘
𝑁
𝑖
𝑘

To avoid zero probabilities, Laplace smoothing is applied:

𝑃
(
𝑗
∣
𝑖
)
=
𝑁
𝑖
𝑗
+
1
∑
𝑘
(
𝑁
𝑖
𝑘
+
1
)

🎲 Text Generation

Starting from the special . token, the model repeatedly samples the next character:

𝑥
𝑡
+
1
∼
𝑃
(
𝑥
𝑡
+
1
∣
𝑥
𝑡
)

Generation stops when the model samples . again.

Example:

. → m → a → r → i → a → .


produces:

maria

📉 Model Evaluation

The probability of a complete sequence is:

𝑃
(
𝑥
1
,
…
,
𝑥
𝑇
)
=
∏
𝑡
𝑃
(
𝑥
𝑡
∣
𝑥
𝑡
−
1
)

Instead of multiplying probabilities, we use log-likelihood:

log
⁡
𝑃
(
𝑥
1
,
…
,
𝑥
𝑇
)
=
∑
𝑡
log
⁡
𝑃
(
𝑥
𝑡
∣
𝑥
𝑡
−
1
)

The model is evaluated using Negative Log-Likelihood:

𝑁
𝐿
𝐿
=
−
1
𝑁
∑
𝑡
log
⁡
𝑃
(
𝑥
𝑡
∣
𝑥
𝑡
−
1
)

A lower NLL means the model assigns higher probability to the observed data.

NLL is closely related to cross-entropy, which becomes the standard loss function when we move to neural language models.

📊 Bigram Model

The model can be visualized as a character transition matrix:

          Next Character
        a   b   c   d   ...
      ┌──────────────────
   a  │
   b  │
   c  │
   d  │
  ... │


Each row represents:

𝑃
(
next character
∣
current character
)

🗂️ Project Structure
bigram-language-model/
│
├── src/
│   └── bigram/
│       ├── __init__.py
│       ├── data.py
│       ├── vocabulary.py
│       ├── model.py
│       ├── sampling.py
│       ├── evaluation.py
│       └── visualization.py
│
├── tests/
│   ├── test_vocabulary.py
│   ├── test_model.py
│   └── test_evaluation.py
│
├── examples/
│   ├── generate_names.py
│   └── visualize_matrix.py
│
├── README.md
├── pyproject.toml
└── .gitignore

🚀 Installation
git clone <your-repository-url>
cd bigram-language-model

python -m venv .venv
source .venv/bin/activate

pip install -e .

▶️ Usage
from bigram import BigramModel, Vocabulary
from bigram.data import download_words
from bigram.sampling import generate_words

words = download_words()

vocab = Vocabulary(words)

model = BigramModel(vocab)
model.fit(words)
model.normalize()

generated = generate_words(
    model,
    num_words=10,
    seed=42,
)

for word in generated:
    print(word)

🧪 Tests

Run the test suite with:

pytest

🛣️ Roadmap
 Character-level bigram model
 Bigram counting
 Probability estimation
 Laplace smoothing
 Random text generation
 Negative log-likelihood
 Bigram visualization
 Neural bigram model
 Embeddings
 MLP language model
 Backpropagation from scratch
 Larger-context language model
 Transformer-based language model
📚 Learning Goal

This project is a step toward understanding modern language models from first principles:

Bigram Model
     ↓
Probability
     ↓
NLL / Cross-Entropy
     ↓
Neural Language Model
     ↓
Embeddings + MLP
     ↓
Backpropagation
     ↓
Attention / Transformers


The emphasis is on understanding why language models work mathematically, not just using existing frameworks.
