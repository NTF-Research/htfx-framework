import os 
import shutil
import numpy as np # type: ignore
from colorama import init, Fore, Style # type: ignore

from framework.HybridTaxonomyFrameworkSetup import HybridTaxonomyFrameworkSetup
from framework.data.DataImpl import DataImpl
from framework.labeler.LabelerImpl import LabelerImpl
from framework.finetuner.FinetunerImpl import FinetuneImpl
from framework.embedder.EmbedderImpl import EmbedderImpl
from framework.classifier.ClassifierImpl import ClassifierImpl
from framework.searcher.SearcherImpl import SearcherImpl




class HybridTaxonomyFramework:
    def __init__(self):
        self.setup = HybridTaxonomyFrameworkSetup()
        self.data = DataImpl()
        self.labeler = LabelerImpl()
        self.embedder = EmbedderImpl()
        self.classifier = ClassifierImpl()
        self.searcher = SearcherImpl()

        self.label_name_mappings = []
        self.item_label_mappings = []
        pass

    def initialize(self, force: bool = False):
        if self.setup.workspace is None:
            print(f"{Fore.RED}[ HybridTaxonomyFramework -> initialize ] - Workspace is not setup!{Style.RESET_ALL}")
            return False
        
        if force and os.path.exists(self.setup.workspace):
            shutil.rmtree(self.setup.workspace)
        
        os.makedirs(self.setup.workspace, exist_ok=True)
        print(self.setup.workspace)
        pass

    def labelling(self) -> bool:
        data, data_names = self.data.data_for_labelling()
        label_name_mappings, item_label_mappings = self.labeler.labelling(data, data_names)
        self.data.save_label_name_mappings(label_name_mappings)
        self.data.save_item_label_mappings(item_label_mappings)
        pass
    
    def finetune(self):
        return self.embedder.finetune()
        pass
    
    def embedding(self):
        data, data_names = self.data.data_for_embedding()
        item_ids, vecs = self.embedder.embedding_items(data, data_names)
        self.data.save_vecs("embeds", vecs)
        self.data.save_item_ids("embeds", item_ids)
        pass

    def split_data(self):
        return self.data.split()
        pass
    
    def train(self):
        label_ids = self.data.load_label_ids("train")
        vecs = self.data.load_vecs("train")
        return self.classifier.train(label_ids, vecs)
        pass

    def test(self):
        label_ids = self.data.load_label_ids("new")
        vecs = self.data.load_vecs("new")
        label_name_mappings = self.data.load_label_name_mappings()
        return self.classifier.test(label_ids, vecs, label_name_mappings)
        pass

    def proba(self, text):
        vecs = self.embedder.embedding_texts([text])
        return self.classifier.proba(vecs)
    
    def recommend(self, text):
        if not self.searcher.is_ready():
            data_names = ["train", "test", "new"]
            for data_name in data_names:
                is_new = data_name == "new"
                label_ids = self.data.load_label_ids(data_name)
                item_ids = self.data.load_item_ids(data_name)
                vecs = self.data.load_vecs(data_name)
                self.searcher.add_items(item_ids, label_ids, vecs, is_new)
            pass

        vecs = self.embedder.embedding_texts([text])
        probs = self.classifier.proba(vecs)

        items = []
        for label_id, prob in probs.items():
            print(f"{label_id} - {prob}")
            match_items = self.searcher.search(label_id, vecs[0])
            for match_item in match_items:
                item_data, data_names = self.data.data_of_item(match_item["item_id"])
                for data_name in data_names:
                    if data_name == "item_id":
                        continue
                    data_index = data_names.index(data_name)
                    match_item[data_names[data_index]] = item_data[data_index]
                items.append(match_item)
                pass
            
        return items