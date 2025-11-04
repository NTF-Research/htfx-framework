from framework.searcher.SearcherSetup import SearcherSetup
from framework.searcher.SearcherFAISS import SearcherFAISS

class SearcherImpl:
    def __init__(self):
        self.setup = SearcherSetup()
        self.instance = None
        pass
    
    def __create_instance__(self):
        if self.instance is not None:
            return True
        
        if self.setup.method is None or len(self.setup.method.strip().lower()) < 1:
            return False
        
        name =  self.setup.method.strip().lower()
        if name == "faiss":
            self.instance = SearcherFAISS(self.setup)
            return True
        
        return False

    def is_ready(self):
        if not self.__create_instance__():
            return False
        return self.instance.is_ready()
    
    def add_items(self, item_ids, label_ids, vecs, is_new):
        if not self.__create_instance__():
            return False
        return self.instance.add_items(item_ids, label_ids, vecs, is_new)

    def search(self, label_id, vec):
        if not self.__create_instance__():
            return []
        
        return self.instance.search(label_id, vec)