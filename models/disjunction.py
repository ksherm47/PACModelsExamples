from .pac_model import PACModel
from .literal import Literal


class Disjunction(PACModel):

    def __init__(self, literals: list[Literal]):
        self.__max_index = sorted(literals, key=lambda x: x.index)[-1].index
        self.__literals = literals

    def __str__(self):
        return ' OR '.join([str(lit) for lit in self.__literals]) if self.__literals else 'Empty Disjunction'

    def size(self) -> int:
        return len(self.__literals)

    def get_literals(self) -> list[Literal]:
        return self.__literals

    def evaluate(self, data_point) -> int:
        if len(data_point) < self.__max_index + 1:
            raise ValueError(f'Disjunction includes literals not included in data point. \
            (Data point has length {len(data_point)} but disjunction contains literal with index {self.__max_index}')

        for lit in self.__literals:
            if bool(data_point[lit.index]) != lit.negation:
                return 1

        return 0
