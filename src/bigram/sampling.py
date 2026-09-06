import torch

from bigram.model import BigramModel


def generate_word(
    model: BigramModel,
    generator: torch.Generator | None = None,
) -> str:
    """Generate one word from a trained bigram model."""

    if model.probabilities is None:
        raise RuntimeError(
            "Model must be normalized before generating."
        )

    vocabulary = model.vocabulary

    current_id = vocabulary.encode(".")

    output = []

    while True:
        probabilities = model.probabilities[current_id]

        next_id = torch.multinomial(
            probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).item()

        if next_id == vocabulary.encode("."):
            break

        output.append(vocabulary.decode(next_id))
        current_id = next_id

    return "".join(output)


def generate_words(
    model: BigramModel,
    num_words: int,
    seed: int | None = None,
) -> list[str]:

    generator = None

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    return [
        generate_word(model, generator)
        for _ in range(num_words)
    ]
