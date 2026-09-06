from bigram import BigramModel, Vocabulary
from bigram.data import download_words
from bigram.evaluation import negative_log_likelihood


def main():
    words = download_words()

    vocab = Vocabulary(words)

    model = BigramModel(vocab)
    model.fit(words)
    model.normalize()

    nll = negative_log_likelihood(
        model,
        words,
    )

    print(f"Negative Log-Likelihood: {nll:.4f}")


if __name__ == "__main__":
    main()
