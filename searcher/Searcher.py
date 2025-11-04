from framework.searcher.SearcherSetup import SearcherSetup

class Searcher:
    def __init__(self, setup: SearcherSetup):
        self.setup = setup
        pass


    def is_ready(self) -> bool:
        return False
    
    def add_items(self, item_ids, label_ids, vecs, is_new):
        return False
    
    def search(label_id, vec):
        return []