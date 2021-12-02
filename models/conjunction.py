from pac_model import PACModel
import numpy as np


class Literal:

    def __init__(self, index: int, negation: bool):
        if index <= 0:
            raise ValueError('Literal must have non-negative index')
        self.index = index
        self.negation = negation


class Conjunction(PACModel):

    def __init__(self, literals: list[Literal]):
        self.__max_index = list(literals).sort(key=lambda x: x.index)[-1]
        self.__literals = literals

    def evaluate(self, data_point):
        if len(data_point) < self.__max_index + 1:
            raise ValueError(f'Data point includes literals not included in conjunction. \
            (Data point has length {len(data_point)} but conjunction contains literal with index {self.__max_index}')

        for lit in self.__literals:
            if bool(data_point[lit.index]) == lit.negation:
                return 0

        return 1


def elimination_algorithm(data_train: np.array, data_train_labels: np.array):
    lits = [Literal(idx, neg) for idx in range(data_train.shape[1]) for neg in (True, False)]

    for (data_point, label) in zip(data_train, data_train_labels):
        if label == 1:
            for idx in range(data_train.shape[1]):
                if data_point[idx] == 1:

                    # Remove NOT x_idx from lits
                    for lit in lits:
                        if lit.index == idx and lit.negation:
                            lits.remove(lit)
                            break

                else:  # data_point[idx] == 0

                    # Remove x_idx from lits
                    for lit in lits:
                        if lit.index == idx and not lit.negation:
                            lits.remove(lit)
                            break

    return Conjunction(lits)


def get_approx_sample_size(epsilon, delta, n, improved=True):
    if improved:
        return (1 / epsilon) * np.log(1 / delta) + n / epsilon
    return (2 * n / epsilon) * (np.log(2 * n) + np.log(1 / delta))


def get_conjunction(data_train: np.array, data_train_labels: np.array):
    return elimination_algorithm(data_train, data_train_labels)
