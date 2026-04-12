"""Text decomposition is intentionally removed in the GME-only refactor."""


class TextDecomposer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This refactor only keeps the GME baseline retrieval pipeline.")
