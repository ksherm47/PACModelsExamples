from .pac_model import PACModel
from .literal import Literal
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
        self.__max_index = sorted(literals, key=lambda x: x.index)[-1].index if literals else -1

    def __str__(self):
        chain_str = ' OR '.join([str(lit) + ' -> ' + str(label) for lit, label in zip(self.__literals, self.__labels)])
        return '(' + chain_str + f') -> {self.__default}'

    def size(self):
        return len(self.__literals)

    def evaluate(self, data_point):
        if len(data_point) < self.__max_index + 1:
            raise ValueError(f'Decision list includes literals not included in data point. \
            (Data point has length {len(data_point)} but decision list contains literal with index {self.__max_index}')

        for (lit, label) in zip(self.__literals, self.__labels):
            if bool(data_point[lit.index]) == lit.negation:
                return label

        return self.__default


def __decision_list_algorithm(data_train: np.array, data_train_labels: np.array):
    decision_list_literals = []
    decision_list_labels = []

    s = list(enumerate(list(data_train_labels)))
    all_literals = [Literal(idx, neg) for idx in range(data_train.shape[1]) for neg in (True, False)]

    # Constructs S_l for each literal
    def define_s_l(examples, literals):
        s_l_lookup = {lit: [] for lit in literals}
        for lit in literals:
            for data_index, label in examples:
                if data_train[data_index][lit.index] != int(lit.negation):
                    s_l_lookup[lit].append((data_index, label))
        return s_l_lookup

    # Checks for loop termination criteria
    def all_data_has_same_label(examples):
        labels = [b for _, b in examples]
        return labels.count(1) == len(labels) or labels.count(0) == len(labels)

    s_l_lookup = define_s_l(s, all_literals)
    while not all_data_has_same_label(s):

        # Find useful literals
        useful_literal_found = False
        for lit in all_literals:
            lit_data_labels = [label for _, label in s_l_lookup[lit]]
            lit_data_indices = [idx for idx, _ in s_l_lookup[lit]]
            for b in (1, 0):
                if lit_data_labels and lit_data_labels.count(b) == len(lit_data_labels):
                    decision_list_literals.append(lit)
                    decision_list_labels.append(b)

                    # Remove data points from s
                    s = [(data_idx, label) for data_idx, label in s if data_idx not in lit_data_indices]
                    s_l_lookup = define_s_l(s, all_literals)
                    useful_literal_found = True

        # If no useful literal found, find "most useful" literal
        if not useful_literal_found:
            literal_usefulnesses = []
            for lit in all_literals:
                lit_data_labels = [label for _, label in s_l_lookup[lit]]
                lit_data_indices = [idx for idx, _ in s_l_lookup[lit]]
                if lit_data_labels:
                    for b in (1, 0):
                        usefulness = lit_data_labels.count(b) / len(lit_data_labels)
                        literal_usefulnesses.append((lit, usefulness, b, lit_data_indices))

            most_useful = sorted(literal_usefulnesses, key=lambda x: x[1])[-1]
            decision_list_literals.append(most_useful[0])
            decision_list_labels.append(most_useful[2])

            s = [(data_idx, label) for data_idx, label in s if data_idx not in most_useful[3]]
            all_literals = [l for l in all_literals if l.index != most_useful[0].index]
            s_l_lookup = define_s_l(s, all_literals)

    decision_list_default = s[0][1] if s else 0  # They all have same common label. If s is empty, arbitrarily set to 0
    return DecisionList(decision_list_literals, decision_list_labels, decision_list_default)


def get_approx_sample_size(epsilon, delta, n, c=1):
    return c * int((1 / epsilon) * (n * np.log(n) + np.log(1 / delta)))


def get_decision_list(data_train: np.array, data_train_labels: np.array):
    return __decision_list_algorithm(data_train, data_train_labels)
