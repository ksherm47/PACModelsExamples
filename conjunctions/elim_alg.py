from enum import Enum

class Literal:

    def __init__(self,idx, neg=False):
        self.Index = idx
        self.Not = neg


class LiteralPresence(Enum):
    POS = 0
    NEG = 1
    ABSENT = 2


class Conjunction:

    def __init__(self, presence_list):
        self.__literals = []
        for i in range(0, len(presence_list)):
            if presence_list[i] != LiteralPresence.ABSENT:
                negation = True if presence_list[i] == LiteralPresence.NEG else False
                self.__literals.append(Literal(i + 1, negation))
                self.__max_index = i + 1


    def evaluate(self, bool_list):
        if len(bool_list) >= self.__max_index:
            raise ValueError(f'Size mismatch between boolean list {len(bool_list)} and conjunction ({self.__max_index}).')
        label = True
        for lit in self.__literals:
            value = bool(bool_list[lit.Index - 1])
            value = not value if lit.Not else value
            label = label and value
        return label





