from framework.embedder.EmbedderSetup import EmbedderSetup

class Embedder:
    def __init__(self, setup: EmbedderSetup):
        self.setup = setup
        self.model = None
        pass

    def embedding_texts(self, texts : list[str]):
        pass

    def embedding_items(self, data, data_names):
        pass