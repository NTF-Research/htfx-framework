import numpy as np # type: ignore
from framework.searcher.Searcher import Searcher

class SearcherFAISS(Searcher):
    def __init__(self, setup):
        super().__init__(setup)
        self.items = {}
        self.faiss_indexs = {}

    def is_ready(self) -> bool:
        return self.faiss_indexs is not None and len(self.faiss_indexs) > 0
        
    
    def add_items(self, item_ids, label_ids, vecs, is_new):
        import faiss # type: ignore
        import numpy as np # type: ignore

        dim = vecs.shape[1]

        for item_id, label_id, vec in zip(item_ids, label_ids, vecs):
            # Add item
            self.items[item_id] = is_new

            # 
            if label_id not in self.faiss_indexs:
                index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
                self.faiss_indexs[label_id] = index
            
            vec = np.asarray(vec, dtype='float32')
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)  # shape = (1, dim)
            
            ids = np.array([item_id], dtype='int64')
            self.faiss_indexs[label_id].add_with_ids(vec, ids)

        return True
    
    def search(self, label_id, vec):
        items = []
        vec = np.asarray(vec, dtype='float32')
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)  # shape = (1, dim)

        query = np.array(vec, dtype='float32').reshape(1, -1)
        distances, item_ids = self.faiss_indexs[label_id].search(query, self.setup.top_k)
        
        if distances is None or item_ids is None:
            return items
        
        for item_id, distance in zip(item_ids[0], distances[0]):
            if item_id < 0:
                continue
            
            items.append({
                "item_id": int(item_id), 
                "distance":float(distance), 
                "is_new":self.items[item_id]
                })
            
        return items