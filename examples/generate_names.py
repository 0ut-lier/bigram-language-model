import torch

from bigram import BigramModel, Vocabulary
from bigram.data import download_words
from bigram.sampling import generate_words


def main():
    words = download_words()

    vocabulary = Vocabulary(words)

    model = BigramModel(vocabulary)

    model.fit(words)
    model.normalize()

    generated = generate_words(
        model,
        num_words=20,
        seed=2147483647,
    )

    for word in generated:
        print(word)


if __name__ == "__main__":
    main()
