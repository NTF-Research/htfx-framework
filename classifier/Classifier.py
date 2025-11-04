from framework.classifier.ClassifierSetup import ClassifierSetup

class Classifier:
    def __init__(self, setup: ClassifierSetup):
        self.setup = setup
        self.model = None
        pass

    def train(self, label_ids, vecs):
        pass

    def test(self, label_ids, vecs, label_name_mappings ):
        pass

    def proba(self, vecs):
        pass