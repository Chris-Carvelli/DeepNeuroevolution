class SilentTqdm:
    """
    Fake tqdm interface (for consoles that aren't able to render tqdm properly)
    """
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass
