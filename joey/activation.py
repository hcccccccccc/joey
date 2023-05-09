from abc import ABC, abstractmethod
from sympy import Max, sign, Min, exp
from devito import Eq


class Activation(ABC):
    """
    An abstract class representing an activation function.

    When you create a subclass of Activation, you must implement
    the backprop_eqs() method.

    Parameters
    ----------
    function : function
        A function to apply to data. Usually, it will be a one-argument
        function f(x) where x is the raw output of a layer.
    """

    def __init__(self, function):
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    @abstractmethod
    def backprop_eqs(self, layer):
        """
        Returns a list of Devito equations describing how backpropagation
        should proceed when encountering this activation function.

        Parameters
        ----------
        layer : Layer
            The next layer in a backpropagation chain.
        """
        pass


class ReLU(Activation):
    """An Activation subclass corresponding to ReLU."""

    def __init__(self):
        super().__init__(self.relu)

    def backprop_eqs(self, layer):
        dims = layer.result_gradients.dimensions
        return [Eq(layer.result_gradients[dims],
                   layer.result_gradients[dims]
                   * Max(0, sign(layer.result[dims])))]

    def relu(self, x):
        return Max(0, x)


class LeakyReLU(Activation):
    """An Activation subclass corresponding to ReLU."""
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope
        
        super().__init__(self.leakyrelu)

    def backprop_eqs(self, layer):
        dims = layer.result_gradients.dimensions
        return [Eq(layer.result_gradients[dims],
                   layer.result_gradients[dims]
                   * (Max(0, sign(layer.result[dims])
                      + Min(0, sign(layer.result[dims]
                                    * self.negative_slope)))))]
    
    def leakyrelu(self, x):
        return Max(0,x) + Min(0, x*self.negative_slope)


class Sigmoid(Activation):
    """An Activation subclass corresponding to ReLU."""
    def __init__(self):
        super().__init__(self.sigmoid)

    def backprop_eqs(self, layer):
        dims = layer.result_gradients.dimensions
        return [Eq(layer.result_gradients[dims],
                   layer.result_gradients[dims]*(1-layer.result[dims])*layer.result[dims])]

    def sigmoid(self, x):
        return 1/(1+exp(x))

class Dummy(Activation):
    """An Activation subclass corresponding to f(x) = x."""

    def __init__(self):
        super().__init__(self.dummy)

    def backprop_eqs(self, layer):
        return []

    def dummy(self, x):
        return x
