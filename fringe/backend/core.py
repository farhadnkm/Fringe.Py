from abc import abstractmethod


class CoreFunctions:
    @staticmethod
    @abstractmethod
    def convert(input_):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def zeros_like(input_):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def expand_dim(input_, axis):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def repeat(input_, repeats, axis):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def reduce_sum(input_, axis):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def exp(input_):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def multiply(x, y):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def divide(x, y):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def sqrt(input_):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def complex(real, imag):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def meshgrid(x, y, indexing):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def linspace(start, stop, num):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def where(condition, x, y):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def logical_and(*args):
        raise NotImplementedError


    @staticmethod
    @abstractmethod
    def pad(input_, padding, fill_value):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def unpad(input_, padding):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def abs(input_):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def angle(input_):
        raise NotImplementedError


    @staticmethod
    @abstractmethod
    def fft(input_):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def ifft(input_):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fftshift(input_):
        raise NotImplementedError

    @staticmethod
    def zeros(shape, dtype):
        raise NotImplementedError

    @staticmethod
    def zeros_like(a, dtype):
        raise NotImplementedError
