from .relu import ReLU, LeakyReLU
from .gelu import GELU
from .sigmoid import Sigmoid
from .softmax import Softmax
from .tanh import Tanh
from .activation_function import ActivationFunction

act_functions:dict[str, ActivationFunction] = {
    'ReLU'     :ReLU,
    'LeakyRelU':LeakyReLU,
    'GELU'     :GELU,
    'sigmoid'  :Sigmoid,
    'softmax'  :Softmax,
    'tanh'     :Tanh,
}

__all__ = [
    'ReLU',
    'LeakyReLU',
    'GELU',
    'Sigmoid',
    'Softmax',
    'Tanh',
    'ActivationFunction',
    'act_functions',
]