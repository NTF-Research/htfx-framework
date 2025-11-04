class ClassifierSetup:
    def __init__(self):
        self.workspace = None
        self.method = "logistic-regression"
        self.top_k = 5
        self.min_proba = 0.1
        pass