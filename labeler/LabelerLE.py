from framework.labeler.Labeler import Labeler


class LabelerLabelEncoding(Labeler):
    def __init__(self, setup):
        super().__init__(setup)

    def labelling(self, data, data_names) -> dict[int, str] | dict[int, int]:
        index_of_item_id = data_names.index("item_id")
        index_of_category = data_names.index("main_category")

        item_ids = []
        label_names = []
        label_ids = []

        for item in data:
            item_ids.append(item[index_of_item_id])
            label_names.append(item[index_of_category])

        from sklearn.preprocessing import LabelEncoder # type: ignore
        label_encoder = LabelEncoder()
        label_ids = label_encoder.fit_transform(label_names)
        
        item_label_mappings = {}
        label_name_mappings = {}

        for label_id, label_name in enumerate(label_encoder.classes_):
            label_name_mappings[label_id] = label_name
            pass

        for item_id, label_id in zip(item_ids, label_ids):
            item_label_mappings[item_id] = label_id
            pass
        
        return label_name_mappings, item_label_mappings