def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Split text into overlapping character n-grams, word by word.

    Words shorter than *n* are kept as-is so they are still indexed.
    """
    text = text.lower().strip()
    tokens: list[str] = []
    for word in text.split():
        if len(word) < n:
            tokens.append(word)
        else:
            tokens.extend(word[i:i + n] for i in range(len(word) - n + 1))
    return tokens
