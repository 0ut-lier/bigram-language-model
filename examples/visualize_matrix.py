from bigram import BigramModel, Vocabulary
from bigram.data import download_words
from bigram.visualization import plot_bigram_counts


def main():
    # Load dataset
    words = download_words()

    # Build vocabulary
    vocab = Vocabulary(words)

    # Train model
    model = BigramModel(vocab)
    model.fit(words)

    # Visualize bigram counts
    plot_bigram_counts(model)


if __name__ == "__main__":
    main()
