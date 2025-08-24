from typing import Union, TextIO, BinaryIO, TypeAlias
from torch import Tensor
from numpy import ndarray
from os import PathLike
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.modules.loss_functions.loss_function import LossFunction
    from src.training.Optimizers.base_optimizer import BaseOptimizer
    from assets.configs.base_config import BaseConfig
    from src.modules.activation_functions.activation_function import ActivationFunction
    from src.nlp.tokenizer import koTokenizer


Path: TypeAlias = str
FileLike: TypeAlias = TextIO|BinaryIO
ListLike: TypeAlias = list|tuple|Tensor|ndarray