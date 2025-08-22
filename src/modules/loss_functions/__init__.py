from .cross_entropy_loss import CrossEntropyLoss
from .mean_squared_error import MeanSquaredError

loss_functions = {
    'CE':CrossEntropyLoss,
    'MSE':MeanSquaredError
}