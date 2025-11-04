import os
from framework.labeler.LabelerSetup import LabelerSetup
from framework.labeler.Labeler import Labeler
from framework.labeler.LabelerLE import LabelerLabelEncoding

class LabelerImpl:
    def __init__(self):
        self.setup = LabelerSetup()
        self.instance : Labeler = None
        self.label_name_mappings = {}
        self.item_label_mappings = {}
        pass
    
    def __create_instance__(self) -> bool:
        if self.instance is not None:
            return True
        
        if self.setup.method is None or len(self.setup.method.strip()) < 1:
            return False
        
        name = self.setup.method.strip().lower()
        if name == "label-encoding":
            self.instance = LabelerLabelEncoding(self.setup)
            return True

        return False
        pass

    def labelling(self, data, data_names) -> dict[int, str] | dict[int, int]:
        if not self.__create_instance__():
            return {}, {}
        
        label_name_mappings, item_label_mappings = self.instance.labelling(data, data_names)
        return label_name_mappings, item_label_mappings
        pass

    def load_label_name_mappings(self) -> bool:
        if self.label_name_mappings is not None and len(self.label_name_mappings) > 0:
            return True
        
        try:
            self.label_name_mappings = {}
            file_path = f"{self.setup.workspace}/label/label_name_mappings.txt"
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    
                    if line:
                        key, value = line.split("->")
                        self.label_name_mappings[int(key)] = str(value)
            
            return True
            pass
        except Exception as e:
            self.label_name_mappings = None
            print(e)
            return False
        

    
    def load_item_label_mappings(self) -> bool:
        if self.item_label_mappings is not None and len(self.item_label_mappings) > 0:
            return True
        
        try:
            self.item_label_mappings = {}
            with open(f"{self.setup.workspace}/label/item_label_mappings.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        key, value = line.split(",")
                        self.item_label_mappings[int(key)] = int(value)
            return True
            pass
        except Exception as e:
            self.item_label_mappings = None
            print(e)
            return False
        return False