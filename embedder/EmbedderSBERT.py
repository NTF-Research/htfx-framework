import numpy as np
from framework.embedder.EmbedderSetup import EmbedderSetup
from framework.embedder.Embedder import Embedder

class EmbedderSBERT(Embedder):
    def __init__(self, setup: EmbedderSetup):
        super().__init__(setup)

    def __load_model__(self):
        if self.model is not None:
            return True
        return False

    def embedding(self, texts : list[str]):
        if not self.__load_model__():
            from sentence_transformers import SentenceTransformer # type: ignore
            self.model = SentenceTransformer(self.setup.pretrained, self.setup.device)
            pass

        if self.model is None:
            return False
        
        vecs = self.model.encode(texts, show_progress_bar=True)
        return vecs
        pass