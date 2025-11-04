import os
import numpy as np # type: ignore
from framework.classifier.Classifier import Classifier


class ClsLogisticRegression(Classifier):
    def __init__(self, setup):
        super().__init__(setup)

    def __load_model__(self) -> bool:
        if self.model is not None:
            return True
        
        file_path = f"{self.setup.workspace}/classifier/logistic_regression.pkl"
        if not os.path.exists(file_path):
            return False

        import joblib # type: ignore
        self.model = joblib.load(file_path)
        return self.model is not None
    

    def train(self, label_ids, vecs):
        from sklearn.linear_model import LogisticRegression # type: ignore
        from sklearn.preprocessing import normalize # type: ignore
        self.model =  LogisticRegression(
            penalty='l2',
            C=2.0,
            solver='lbfgs',
            max_iter=1000,
            n_jobs=-1,
            class_weight='balanced',
            random_state=42
            )
        
        if self.model is None:
            return False
        
        batch_size = 1000
        for i in range(0, len(vecs), batch_size):
            vecs[i:i+batch_size] = normalize(vecs[i:i+batch_size])
            pass

        self.model.fit(vecs, label_ids)

        import joblib # type: ignore
        output_dir = f"{self.setup.workspace}/classifier"
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.model, f"{output_dir}/logistic_regression.pkl")
        return True

    def test(self, label_ids, vecs, label_name_mappings):
        if not self.__load_model__():
            return False
        
        from sklearn.preprocessing import normalize # type: ignore
        from sklearn.metrics import classification_report, accuracy_score # type: ignore

        batch_size = 1000
        for i in range(0, len(vecs), batch_size):
            vecs[i:i+batch_size] = normalize(vecs[i:i+batch_size])

        predictions = self.model.predict(vecs)

        print("Testing Logistic Regression model")
        print("Classification Logistic Regression Report:")
        if label_name_mappings is None:
            print(classification_report( label_ids,  predictions))
        else:
            label_names = []
            prediction_names = []
            for label_id, label_predict_id in zip(label_ids, predictions):
                label_names.append(label_name_mappings[label_id])
                prediction_names.append(label_name_mappings[label_predict_id])
            print(classification_report( label_names,  prediction_names))

        print(f"Accuracy: {accuracy_score(label_ids, predictions)}")

        return True

    def proba(self, vecs):
        probabilities = {}
        
        if not self.__load_model__():
            return probabilities
        
        probs = self.model.predict_proba(vecs)[0]
        top_indices = np.argsort(probs)[self.setup.top_k:][::-1]
        top_classes = self.model.classes_[top_indices]
        top_probs = probs[top_indices]

        for class_id, prob in zip(top_classes, top_probs):
            if prob < self.setup.min_proba:
                continue
            probabilities[int(class_id)] = float(prob)
        
        return probabilities
        pass