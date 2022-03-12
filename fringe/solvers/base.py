from abc import abstractmethod


class Solver:
    @abstractmethod
    def solve(self, input_, *args, **kwargs):
        raise NotImplementedError
