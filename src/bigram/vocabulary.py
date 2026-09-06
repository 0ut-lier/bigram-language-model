class Vocabulary:
    """Maps characters to integer IDs and integer IDs back to characters."""

    START_END = "."

    def __init__(self, words: list[str]):
        chars = sorted(set("".join(words)))

        self.stoi = {
            char: index + 1
            for index, char in enumerate(chars)
        }

        self.stoi[self.START_END] = 0

        self.itos = {
            index: char
            for char, index in self.stoi.items()
        }

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, char: str) -> int:
        return self.stoi[char]

    def decode(self, index: int) -> str:
        return self.itos[index]

    def add_boundaries(self, word: str) -> list[str]:
        return [
            self.START_END,
            *word,
            self.START_END,
        ]
