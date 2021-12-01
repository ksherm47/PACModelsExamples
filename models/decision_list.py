from pac_model import PACModel
from conjunction import Literal
import numpy as np


class DecisionList(PACModel):

    def __init__(self, literals: list[Literal], labels: list[int], default: int):
        if len(literals) != len(labels):
            raise ValueError(f'Size of literal list {len(literals)} != size of label list {len(labels)}')
        if default not in (1, 0):
            raise ValueError(f'Default value must be either 1 or 0 (received {default})')
        for label in labels:
            if label not in (1, 0):
                raise ValueError(f'Label list must only include values equal to 1 or 0  \
                (Label list includes value {label})')

        self.__literals = literals
        self.__labels = labels
        self.__default = default
        self.__max_index = list(literals).sort(key=lambda x: x.index)[-1]

    def evaluate(self, data_point):
        if len(data_point) < self.__max_index + 1:
            raise ValueError(f'Data point includes literals not included in decision list. \
            (Data point has length {len(data_point)} but decision list contains literal with index {self.__max_index}')

        for (lit, label) in zip(self.__literals, self.__labels):
            if bool(data_point[lit.index]) == lit.negation:
                return label

        return self.__default


def decision_list_algorithm(data_train: np.array, data_train_labels: np.array):
    pass



def get_decision_list(data_train):
    return decision_list_algorithm(data_train)
