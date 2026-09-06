from bigram import BigramModel, Vocabulary
from bigram.data import download_words
from bigram.sampling import generate_words


def main():
    # Load dataset
    words = download_words()

    # Build vocabulary
    vocab = Vocabulary(words)

    # Create and train model
    model = BigramModel(vocab)
    model.fit(words)

    # Convert counts into probabilities
    model.normalize()

    # Generate names
    generated_words = generate_words(
        model,
        num_words=20,
        seed=42,
    )

    print("Generated names:")
    print("-" * 30)

    for word in generated_words:
        print(word)


if __name__ == "__main__":
    main()
