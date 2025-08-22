from typing import Union, TextIO, BinaryIO, TypeAlias
from torch import Tensor
from numpy import ndarray
from os import PathLike


Path: TypeAlias = str
FileLike: TypeAlias = TextIO|BinaryIO
ListLike: TypeAlias = list|tuple|Tensor|ndarray