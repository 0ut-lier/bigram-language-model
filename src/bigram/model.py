import torch

from bigram.vocabulary import Vocabulary


class BigramModel:
    """
    Character-level bigram language model.

    The model estimates:

        P(c_next | c_current)

    using character transition counts.
    """

    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary

        self.counts = torch.zeros(
            (vocabulary.size, vocabulary.size),
            dtype=torch.int32,
        )

        self.probabilities = None

    def fit(self, words: list[str]) -> None:
        """Build the bigram count matrix from the training data."""

        for word in words:
            chars = self.vocabulary.add_boundaries(word)

            for current, next_char in zip(chars, chars[1:]):
                current_id = self.vocabulary.encode(current)
                next_id = self.vocabulary.encode(next_char)

                self.counts[current_id, next_id] += 1

    def normalize(self, smoothing: int = 1) -> torch.Tensor:
        """
        Convert bigram counts into conditional probabilities.

        Laplace smoothing is applied by default.
        """

        counts = self.counts + smoothing

        self.probabilities = (
            counts.float()
            / counts.sum(dim=1, keepdim=True)
        )

        return self.probabilities

    def probability(
        self,
        current: str,
        next_char: str,
    ) -> float:
        """Return P(next_char | current)."""

        if self.probabilities is None:
            raise RuntimeError(
                "Model must be normalized before querying probabilities."
            )

        current_id = self.vocabulary.encode(current)
        next_id = self.vocabulary.encode(next_char)

        return self.probabilities[current_id, next_id].item()
