from bigram.vocabulary import Vocabulary


def test_vocabulary():
    words = ["cat", "dog"]

    vocab = Vocabulary(words)

    assert vocab.encode(".") == 0
    assert vocab.decode(0) == "."


def test_vocabulary_size():
    words = ["cat"]

    vocab = Vocabulary(words)

    assert vocab.size == 4
