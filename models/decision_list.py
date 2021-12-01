from pac_model import PACModel
from conjunction import Literal


class DecisionList(PACModel):

    def __init__(self, literals: list[Literal], labels: int):
        pass

    def evaluate(self, data_point):
        pass


def decision_list_algorithm(data_train):
    pass


def get_decision_list(data_train):
    return decision_list_algorithm(data_train)
