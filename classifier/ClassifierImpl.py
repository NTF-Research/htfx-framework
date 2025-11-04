from framework.classifier.ClassifierSetup import ClassifierSetup
from framework.classifier.Classifier import Classifier
from framework.classifier.LogisticRegression import ClsLogisticRegression

class ClassifierImpl:
    def __init__(self):
        self.setup : ClassifierSetup = ClassifierSetup()
        self.instance: Classifier = None
        pass
    
    def __create_instance__(self) -> bool:
        if self.instance is not None:
            return True
        
        if self.setup.method is None or len(self.setup.method.strip()) < 1:
            return False
        
        name = self.setup.method.strip().lower()

        if name == "logistic-regression":
            self.instance = ClsLogisticRegression(self.setup)
            return True
        
        return False
        pass

    def train(self, label_ids, vecs):
        if not self.__create_instance__():
            return False
        
        return self.instance.train(label_ids, vecs)

    def test(self, label_ids, vecs, label_name_mappings):
        if not self.__create_instance__():
            return False
        
        return self.instance.test(label_ids, vecs, label_name_mappings)

    def proba(self, vecs):
        if not self.__create_instance__():
            return False
        
        return self.instance.proba(vecs)
