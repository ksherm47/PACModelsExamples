from .pac_model import PACModel
from .disjunction import Disjunction
from .literal import Literal
from .conjunction import get_conjunction
from data import project_data
import itertools
import numpy as np


class ThreeCNF(PACModel):

    def __init__(self, terms: list[Disjunction]):
        for disj in terms:
            if disj.size() != 3:
                raise ValueError('Disjunction term is not of size 3')
        self.__terms = terms

    def size(self):
        return sum([term.size() for term in self.__terms])

    def evaluate(self, data_point):
        for disj in self.__terms:
            if disj.evaluate(data_point) == 0:
                return 0
        return 1

    def __str__(self):
        return ' AND '.join([f'({term})' for term in self.__terms])


def __three_cnf_algorithm(data_train: np.array, data_train_labels: np.array):
    n = data_train.shape[1]
    all_literals = [Literal(idx, False) for idx in range(n)]
    all_literals.extend([Literal(idx, True) for idx in range(n)])
    literal_combos = list(itertools.combinations(all_literals, 3))

    disjunction_lookup = {}
    clauses = []
    for trans_idx, combo in enumerate(literal_combos):
        disj = Disjunction(list(combo))
        clauses.append(disj)
        disjunction_lookup[trans_idx] = disj

    if not project_data.data_obj_exists('three_cnf_transformed') and not project_data.data_obj_exists('3CNF_hypothesis'):
        transformed_data = np.empty(shape=(data_train.shape[0], len(clauses)))
        for i, data_point in enumerate(data_train):
            transformed_row = [clause.evaluate(data_point) for clause in clauses]
            transformed_data[i] = transformed_row

        project_data.save_data_obj(transformed_data, 'three_cnf_transformed')

    if not project_data.data_obj_exists('3CNF_hypothesis'):
        print('Loading transformed data...')
        transformed_data = project_data.get_data_obj('three_cnf_transformed')
        print('Beginning elimination algorithm on transformed data...')
        transformed_conj = get_conjunction(transformed_data, data_train_labels, literal_name='z')
        three_cnf_terms = [disjunction_lookup[lit.index] for lit in transformed_conj.get_literals()]
        print('Saving 3CNF hypothesis...')
        three_cnf_hypothesis = ThreeCNF(three_cnf_terms)
        project_data.save_data_obj(three_cnf_hypothesis, '3CNF_hypothesis')
    else:
        print('Loading 3CNF hypothesis...')
        three_cnf_hypothesis = project_data.get_data_obj('3CNF_hypothesis')

    return three_cnf_hypothesis


def get_three_cnf(data_train: np.array, data_train_labels: np.array):
    return __three_cnf_algorithm(data_train, data_train_labels)
