
class Literal:

    def __init__(self, index: int, negation: bool, name: str = 'x'):
        if index < 0:
            raise ValueError('Literal must have non-negative index')
        self.index = index
        self.negation = negation
        self.__name = name

    def __str__(self):
        return ('NOT ' if self.negation else '') + f'{self.__name}_{self.index}'

    def __eq__(self, other):
        return self.index == other.index and self.negation == other.negation

    def __ne__(self, other):
        return self.index != other.index or self.negation != other.negation

    def __hash__(self):
        return hash((self.index, self.negation))
