from framework.data.DataSetup import DataSetup


class Database:
    def __init__(self, setup: DataSetup):
        self.setup = setup
        pass

    def data_for_labelling(self) -> list[tuple] | list[str]:
        pass

    def data_for_embedding(self) -> list[tuple] | list[str]:
        pass

    def data_for_finetune(self) -> list[tuple] | list[str]:
        pass

    def data_of_item(self, item_id) -> tuple | list[str]:
        pass