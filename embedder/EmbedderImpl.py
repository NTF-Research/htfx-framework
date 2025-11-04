import os
import numpy as np # type: ignore
import shutil
from framework.embedder.Embedder import Embedder
from framework.embedder.EmbedderSetup import EmbedderSetup
from framework.embedder.EmbedderSBERT import EmbedderSBERT


class EmbedderImpl:
    def __init__(self):
        self.setup = EmbedderSetup()
        self.instance : Embedder = None
        pass
    
    def __create_instance__(self) -> bool:
        if self.instance is not None:
            return True
        
        if self.setup.method is None or len(self.setup.method.strip()) < 1:
            return False
        
        name = self.setup.method.strip().lower()
        if name == "sbert":
            self.instance = EmbedderSBERT(self.setup)
            return True
        
        return False
        pass

    def finetune(self):
        pass

    def embedding_texts(self, texts : list[str]):
        if not self.__create_instance__():
            return False
        
        return self.instance.embedding(texts)
        pass

    def embedding_items(self, data, data_names):
        if not self.__create_instance__():
            return False

        index_of_item_id = data_names.index("item_id")
        item_ids = []
        texts = []

        for item in data:
            item_ids.append(item[index_of_item_id])
            textformatted = ""
            for index, value in enumerate(item):
                if index == index_of_item_id:
                    continue
                if len(textformatted) > 0 and textformatted.endswith(".") == False:
                    textformatted = textformatted + ". "
                textformatted += value.strip()
            
            texts.append(textformatted)

        vecs = self.instance.embedding(texts)
        return item_ids, vecs
    
    def embedding(self, texts):
        if not self.__create_instance__():
            return None
        
        return self.instance.embedding(texts)