from abc import ABC, abstractmethod


class PACModel(ABC):

    @abstractmethod
    def evaluate(self, data_point) -> int:
        pass

    @abstractmethod
    def size(self) -> int:
        pass
