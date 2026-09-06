import math

from bigram.model import BigramModel


def negative_log_likelihood(
    model: BigramModel,
    words: list[str],
) -> float:
    """
    Calculate the average negative log-likelihood
    of a collection of words.
    """

    if model.probabilities is None:
        raise RuntimeError(
            "Model must be normalized before evaluation."
        )

    total_log_likelihood = 0.0
    num_bigrams = 0

    vocabulary = model.vocabulary

    for word in words:
        chars = vocabulary.add_boundaries(word)

        for current, next_char in zip(chars, chars[1:]):
            probability = model.probability(
                current,
                next_char,
            )

            total_log_likelihood += math.log(probability)
            num_bigrams += 1

    return -total_log_likelihood / num_bigrams
