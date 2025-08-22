from .relu import ReLU, LeakyReLU
from .gelu import GELU
from .sigmoid import Sigmoid
from .softmax import Softmax
from .tanh import Tanh

act_functions = {
    'ReLU'     :ReLU,
    'LeakyRelU':LeakyReLU,
    'GELU'     :GELU,
    'sigmoid'  :Sigmoid,
    'softmax'  :Softmax,
    'tanh'     :Tanh,
}