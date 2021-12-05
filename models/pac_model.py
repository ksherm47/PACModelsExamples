from abc import ABC, abstractmethod


class PACModel(ABC):

    @abstractmethod
    def evaluate(self, data_point):
        pass

    @abstractmethod
    def size(self):
        pass
