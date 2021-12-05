from .literal import Literal
from .conjunction import Conjunction
from .disjunction import Disjunction


def negate_conjunction(conj: Conjunction) -> Disjunction:
    return Disjunction([Literal(lit.index, not lit.negation) for lit in conj.get_literals()])


def negate_disjunction(disj: Disjunction) -> Conjunction:
    return Conjunction([Literal(lit.index, not lit.negation) for lit in disj.get_literals()])
