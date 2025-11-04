import os 
import shutil
import numpy as np # type: ignore
from framework.data.DataSetup import DataSetup
from framework.data.Database import Database
from framework.data.AmazonData import AmazonData

class DataImpl:
    def __init__(self):
        self.setup = DataSetup()
        self.instance: Database = None
        pass
    
    def __create_instance__(self) -> bool:
        if self.instance is not None:
            return True
        
        if self.setup.name is None or len(self.setup.name.strip()) < 1:
            return False
        
        name = self.setup.name.strip().lower()
        if name == "amazon":
            self.instance = AmazonData(self.setup)
            return True
        
        return False
    
    ##########################################################################################
    def data_for_labelling(self) -> list[tuple] | list[str]:
        if not self.__create_instance__():
            return [], []
        
        return self.instance.data_for_labelling()
        pass


    def data_for_embedding(self) -> list[tuple] | list[str]:
        if not self.__create_instance__():
            return [], []
        
        return self.instance.data_for_embedding()
        pass


    def data_for_finetune(self) -> list[tuple] | list[str]:
        if not self.__create_instance__():
            return [], []
        
        pass


    def data_of_item(self, item_id) -> tuple | list[str]:
        if not self.__create_instance__():
            return (), []
        
        return self.instance.data_of_item(item_id)

    ##########################################################################################
    def load_vecs(self, name):
        vecs = []
        file_path = f"{self.setup.workspace}/{name}/vecs.npy"
        if not os.path.exists(file_path):
            return vecs
        
        return np.load(file_path)



    def load_item_ids(self, name):
        ids = []
        file_path = f"{self.setup.workspace}/{name}/item_ids.txt"
        if not os.path.exists(file_path):
            return ids
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(int(line))
        return ids



    def load_label_ids(self, name):
        ids = []
        file_path = f"{self.setup.workspace}/{name}/label_ids.txt"
        if not os.path.exists(file_path):
            return ids
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.append(int(line))
        return ids



    def load_label_name_mappings(self):
        label_name_mappings = {}
        file_path = f"{self.setup.workspace}/label/label_name_mappings.txt"
        
        if not os.path.exists(file_path):
            return label_name_mappings
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    key, value = line.split("->")
                    label_name_mappings[int(key)] = str(value)
        
        return label_name_mappings



    def load_item_label_mappings(self):
        item_label_mappings = {}
        file_path = f"{self.setup.workspace}/label/item_label_mappings.txt"
        
        if not os.path.exists(file_path):
            return item_label_mappings
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    key, value = line.split(",")
                    item_label_mappings[int(key)] = int(value)
        
        return item_label_mappings

    def save_vecs(self, name, vecs):
        dir = f"{self.setup.workspace}/{name}"
        os.makedirs(dir, exist_ok=True)
        file_path = f"{dir}/vecs.npy"
        np.save(file_path, vecs)
        return True

    def save_item_ids(self, name, ids):
        dir = f"{self.setup.workspace}/{name}"
        os.makedirs(dir, exist_ok=True)

        with open(f"{dir}/item_ids.txt", "w", encoding="utf-8") as f:
            first = True
            for id in ids:
                f.write(f"{"" if first else "\n"}{id}")
                first = False
        return True

    def save_label_ids(self, name, ids):
        dir = f"{self.setup.workspace}/{name}"
        os.makedirs(dir, exist_ok=True)

        with open(f"{dir}/label_ids.txt", "w", encoding="utf-8") as f:
            first = True
            for id in ids:
                f.write(f"{"" if first else "\n"}{id}")
                first = False
        return True

    def save_label_name_mappings(self, mappings: dict[int, str]):
        dir = f"{self.setup.workspace}/label"
        os.makedirs(dir, exist_ok=True)
        with open(f"{dir}/label_name_mappings.txt","w", encoding="utf-8") as f:
            first_row = True
            for label_id, label_name in mappings.items():
                f.write(f"{"\n" if first_row == False else ""}{label_id}->{label_name}")
                first_row = False
            pass
        return True

    def save_item_label_mappings(self, mappings: dict[int, int]):
        dir = f"{self.setup.workspace}/label"
        os.makedirs(dir, exist_ok=True)
        with open(f"{dir}/item_label_mappings.txt", "w", encoding="utf-8") as f:
            first_row = True
            for item_id, label_id in mappings.items():
                f.write(f"{"\n" if first_row == False else ""}{item_id},{label_id}")
                first_row = False
        return True
    
    ##########################################################################################
    def split(self):
        vecs = self.load_vecs("embeds")
        item_ids = self.load_item_ids("embeds")
        item_label_mappings = self.load_item_label_mappings()
        label_ids = []
        for item_id in item_ids:
            label_ids.append(item_label_mappings[item_id])

        # Split
        from sklearn.model_selection import train_test_split # type: ignore

        train_item_ids, remain_item_ids, train_label_ids, remain_label_ids, train_vecs, remain_vecs = \
        train_test_split(
            item_ids, 
            label_ids, 
            vecs, 
            test_size=0.4, 
            random_state=42, 
            shuffle=True, 
            stratify=label_ids)
        
        # Train data
        self.save_vecs("train", train_vecs)
        self.save_item_ids("train", train_item_ids)
        self.save_label_ids("train", train_label_ids)
        train_item_ids = None
        train_label_ids = None
        train_vecs = None

        test_item_ids, new_item_ids, test_label_ids, new_label_ids, test_vecs, new_vecs = \
        train_test_split(
            remain_item_ids, 
            remain_label_ids, 
            remain_vecs, 
            test_size=0.5, 
            random_state=42, 
            shuffle=True, 
            stratify=remain_label_ids)
        
        remain_item_ids = None
        remain_label_ids = None
        remain_vecs = None

        self.save_vecs("test", test_vecs)
        self.save_item_ids("test", test_item_ids)
        self.save_label_ids("test", test_label_ids)
        test_item_ids = None
        test_label_ids = None
        test_vecs = None

        self.save_vecs("new", new_vecs)
        self.save_item_ids("new", new_item_ids)
        self.save_label_ids("new", new_label_ids)
        new_item_ids = None
        new_label_ids = None
        new_vecs = None

        return True
        pass
