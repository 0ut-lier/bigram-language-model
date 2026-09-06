import matplotlib.pyplot as plt
import torch

from bigram.model import BigramModel


def plot_bigram_counts(model: BigramModel) -> None:
    """Display the bigram count matrix."""

    counts = model.counts

    vocabulary = model.vocabulary

    plt.figure(figsize=(16, 16))

    plt.imshow(
        counts,
        cmap="Blues",
    )

    for i in range(vocabulary.size):
        for j in range(vocabulary.size):
            chars = (
                vocabulary.decode(i)
                + vocabulary.decode(j)
            )

            plt.text(
                j,
                i,
                chars,
                ha="center",
                va="bottom",
                color="gray",
            )

            plt.text(
                j,
                i,
                counts[i, j].item(),
                ha="center",
                va="top",
                color="gray",
            )

    plt.axis("off")
    plt.tight_layout()
    plt.show()
