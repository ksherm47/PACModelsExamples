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
    decision_list_literals = []
    decision_list_labels = []

    s = list(enumerate(list(data_train_labels)))
    all_literals = [Literal(idx, neg) for idx in range(data_train.shape[1]) for neg in (True, False)]
    s_l_lookup = {lit: [] for lit in all_literals}

    # Construct S_l for each literal
    for lit in all_literals:
        for data_index, label in s:
            if data_train[data_index][lit.index] == 1:
                s_l_lookup[lit].append((data_index, label))

    def all_data_has_same_label(examples):
        s_labels = [b for _, b in examples]
        return s_labels.count(1) == len(s_labels) or s_labels.count(0) == len(s_labels)

    while not all_data_has_same_label(s):

        # Find useful literals
        for lit in all_literals:
            lit_data_labels = [label for _, label in s_l_lookup[lit]]
            lit_data_indices = [idx for idx, _ in s_l_lookup[lit]]
            for b in (1, 0):
                if lit_data_labels.count(b) == len(lit_data_labels):
                    decision_list_literals.append(lit)
                    decision_list_labels.append(b)

                    # Remove data points from s
                    s = [(data_idx, label) for data_idx, label in s if data_idx not in lit_data_indices]

    decision_list_default = s[0][1] # They all have same common label
    return DecisionList(decision_list_literals, decision_list_labels, decision_list_default)


def get_decision_list(data_train: np.array, data_train_labels: np.array):
    return decision_list_algorithm(data_train, data_train_labels)
