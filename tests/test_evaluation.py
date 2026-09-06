from bigram.evaluation import negative_log_likelihood
from bigram.model import BigramModel
from bigram.vocabulary import Vocabulary


def test_nll_is_positive():
    words = ["cat", "dog"]

    vocab = Vocabulary(words)
    model = BigramModel(vocab)

    model.fit(words)
    model.normalize()

    nll = negative_log_likelihood(
        model,
        words,
    )

    assert nll > 0


def test_nll_is_finite():
    import math

    words = ["cat", "dog"]

    vocab = Vocabulary(words)
    model = BigramModel(vocab)

    model.fit(words)
    model.normalize()

    nll = negative_log_likelihood(
        model,
        words,
    )

    assert math.isfinite(nll)


def test_nll_on_same_training_data():
    words = ["aa"]

    vocab = Vocabulary(words)
    model = BigramModel(vocab)

    model.fit(words)
    model.normalize()

    nll = negative_log_likelihood(
        model,
        words,
    )

    assert nll > 0
