from .pac_model import PACModel
from .literal import Literal
import numpy as np


class Conjunction(PACModel):

    def __init__(self, literals: list[Literal]):
        self.__max_index = sorted(literals, key=lambda x: x.index)[-1].index if literals else -1
        self.__literals = literals

    def __str__(self):
        return ' AND '.join([str(lit) for lit in self.__literals]) if self.__literals else 'Empty Conjunction'

    def size(self) -> int:
        return len(self.__literals)

    def get_literals(self) -> list[Literal]:
        return self.__literals

    def evaluate(self, data_point) -> int:
        if len(data_point) < self.__max_index + 1:
            raise ValueError(f'Conjunction includes literals not included in data point. \
            (Data point has length {len(data_point)} but conjunction contains literal with index {self.__max_index}')

        for lit in self.__literals:
            if bool(data_point[lit.index]) == lit.negation:
                return 0

        return 1


def __elimination_algorithm(data_train: np.array, data_train_labels: np.array,
                            literal_name='x', verbose=True, tolerant=False, mu=0.8) -> Conjunction:
    lit_dict = {(idx, neg): True for idx in range(data_train.shape[1]) for neg in (True, False)}

    if not tolerant:
        for i, (data_point, label) in enumerate(zip(data_train, data_train_labels)):
            if label == 1:
                for idx in range(data_train.shape[1]):
                    if data_point[idx] == 1:
                        lit_dict[(idx, True)] = False
                    else:  # data_point[idx] == 0
                        lit_dict[(idx, False)] = False
            if verbose:
                print(f'\rElimination algorithm iterated over {i + 1}/{data_train.shape[0]} data points', end='')
    else:
        for idx in range(data_train.shape[1]):
            pos_lit_tolerance = 0
            neg_lit_tolerance = 0
            for i, (data_point, label) in enumerate(zip(data_train, data_train_labels)):
                pos_lit_tolerance += 1 if label == 1 and data_point[idx] == 0 else 0
                neg_lit_tolerance += 1 if label == 1 and data_point[idx] == 1 else 0

            if pos_lit_tolerance > int(mu * data_train.shape[1]):
                lit_dict[(idx, False)] = False
            if neg_lit_tolerance > int(mu * data_train.shape[1]):
                lit_dict[(idx, True)] = False
            if verbose:
                print(f'\rElimination algorithm iterated over {idx + 1}/{data_train.shape[1]} literals', end='')
    if verbose:
        print('\n', end='')

    conj_literals = [Literal(idx, neg, name=literal_name) for idx, neg in lit_dict.keys() if lit_dict[(idx, neg)]]
    return Conjunction(conj_literals)


def get_approx_sample_size(epsilon, delta, n, c=1, improved=False, tolerant=False, mu=0.8) -> int:
    if improved:
        return c * int((1 / epsilon) * np.log(1 / delta) + n / epsilon)
    if tolerant:
        return c * int((4 / mu) * np.log(1 / delta))
    return int((c * 2 * n / epsilon) * (np.log(2 * n) + np.log(1 / delta)))


def get_conjunction(data_train: np.array, data_train_labels: np.array, tolerant=False, mu=0.8,
                    literal_name: str = 'x', verbose=True) -> Conjunction:
    return __elimination_algorithm(data_train, data_train_labels,
                                   literal_name=literal_name, verbose=verbose, tolerant=tolerant, mu=mu)
