import torch

from bigram.model import BigramModel
from bigram.vocabulary import Vocabulary


def test_bigram_counts():
    words = ["aa"]

    vocab = Vocabulary(words)

    model = BigramModel(vocab)
    model.fit(words)

    dot = vocab.encode(".")
    a = vocab.encode("a")

    assert model.counts[dot, a] == 1
    assert model.counts[a, a] == 1
    assert model.counts[a, dot] == 1


def test_probability_rows_sum_to_one():
    words = ["cat", "dog"]

    vocab = Vocabulary(words)

    model = BigramModel(vocab)

    model.fit(words)
    probabilities = model.normalize()

    row_sums = probabilities.sum(dim=1)

    assert torch.allclose(
        row_sums,
        torch.ones(vocab.size),
    )
