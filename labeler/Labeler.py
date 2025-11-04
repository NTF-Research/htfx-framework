from framework.labeler.LabelerSetup import LabelerSetup


class Labeler:
    def __init__(self, setup: LabelerSetup):
        self.setup = setup
        pass

    def labelling(self, data, data_names) -> dict[int, str] | dict[int, int]:
        return [], []

