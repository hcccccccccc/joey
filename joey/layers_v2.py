from abc import abstractmethod
from devito import Function, Eq, Inc, \
    Constant, Operator, SubDimension, \
    Max, ConditionalDimension, SpaceDimension, sqrt
from sympy import And
import numpy as np
import sympy as sp
from itertools import product
from joey import Layer
from joey import default_name_allocator as alloc
from joey import default_dim_allocator as dim_alloc


class ConvV2(Layer):
    """
    A Layer subclass corresponding to convolution layer (mathematically,
    it performs a cross-correlation operation).

    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    dimensions: int
        The dimension of conv operation i.e.
        if the convolution is 1d, 2d or so on
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size, dimensions,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        self._ndims = dimensions

        if (type(padding) is int):
            padding = tuple([padding] * self._ndims)
        if (type(stride) is int):
            stride = tuple([stride] * self._ndims)
        self._error_check(kernel_size, input_size, stride, padding,
                          strict_stride_check)

        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).
        self._kernel_size = (kernel_size[0], input_size[1], *kernel_size[1:])
        self._stride = stride
        self._padding = padding

        super().__init__(self._kernel_size, input_size, activation,
                         alloc, self.get_name,
                         generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding,
                     strict_stride_check):
        if input_size is None or (len(input_size) != self._ndims+2):
            raise Exception("Input size is incorrect")

        if kernel_size is None or (len(kernel_size) != self._ndims+1):
            raise Exception("Kernel size is incorrect")

        if stride is None or (len(stride) != self._ndims):
            raise Exception("Stride is incorrect")

        if padding is None or (len(padding) != self._ndims):
            raise Exception("Padding is incorrect")

        for i in range(0, self._ndims):

            if (type(stride[i]) is not int):
                raise Exception("Stride must be an integer")

            if (type(padding[i]) is not int):
                raise Exception("Padding must be an integer")

            if stride[i] < 1:
                raise Exception("Stride cannot be less than 1")

            if padding[i] < 0:
                raise Exception("Padding cannot be negative")

        if strict_stride_check:
            input_d = input_size[-self._ndims+i] + 2 * padding[i]
            if (input_d - kernel_size[-self._ndims+i]) % stride[i] != 0:
                raise Exception("Stride " + str(stride) + " is not "
                                "compatible with feature map, kernel and "
                                "padding sizes. If you want to proceed "
                                "anyway, set strict_stride_check=False "
                                "when instantiating this object")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  get_name):

        no_of_kernels = kernel_size[0]
        self.dim_dict = dim_dict = {3: 'depth', 2: 'height', 1: 'width'}

        dimensions = ['dbatch', 'dchannel']
        result_shape = []
        input_size = list(input_size)
        # generating  in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
        for i in range(0, self._ndims):
            result_d = (input_size[(-self._ndims+i)] -
                        kernel_size[(-self._ndims+i)] +
                        2 * self._padding[i])//self._stride[i] + 1
            input_size[(-self._ndims+i)] += 2 * self._padding[i]

            result_shape.append(result_d)
            dimensions.append('d_'+dim_dict.get(self._ndims-i, self._ndims-i))

        result_shape = (input_size[0], no_of_kernels, *result_shape)

        # input data function
        input_dimensions = [SpaceDimension(
            get_name("Input_"+x)) for x in dimensions]

        input_func = Function(name=get_name("Input_F"), shape=(input_size),
                              dimensions=input_dimensions, space_order=0,
                              dtype=np.float64)

        # function for kernel
        kernel_dims = [SpaceDimension(get_name("kernel_"+x))
                       for x in dimensions]

        kernel_func = Function(name=get_name("Kernel_F"), shape=(kernel_size),
                               dimensions=(kernel_dims), space_order=0,
                               dtype=np.float64)

        # Result for convolution
        result_dimensions = [SpaceDimension(
            get_name("Result_"+x)) for x in dimensions]

        result_func = Function(name=get_name("Result_F"), shape=result_shape,
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)

        bias_dimensions = [SpaceDimension(get_name("bias_"+x)) for x in ['d']]

        bias = Function(name=get_name("bias_F"), shape=(
            kernel_size[0],), dimensions=bias_dimensions, space_order=0,
            dtype=np.float64)

        kernel_grad_dimensions = [SpaceDimension(
            get_name("kernel_grad"+x)) for x in dimensions]

        kernel_grad = Function(name=get_name("kgrad_"), shape=(
            kernel_size), dimensions=kernel_grad_dimensions, space_order=0,
            dtype=np.float64)

        output_grad_dimensions = [SpaceDimension(
            get_name("output_grad"+x)) for x in dimensions]
        self._output_grad_padded_dimensions = [SpaceDimension(
            get_name("output_grad_padded"+x)) for x in dimensions]
        self._output_grad_dilated_dims = [SpaceDimension(
            get_name("output_dil"+x)) for x in dimensions]

        output_grad = Function(name=get_name("outgrad_"),
                               shape=result_shape,
                               dimensions=output_grad_dimensions,
                               space_order=0, dtype=np.float64)

        bias_grad_dimensions = [SpaceDimension(
            get_name("bias_"+x)) for x in ['d']]

        bias_grad = Function(name=get_name("bgrad_"), shape=(
            kernel_size[0],), dimensions=bias_grad_dimensions, space_order=0,
            dtype=np.float64)

        return (kernel_func, input_func, result_func, bias, kernel_grad,
                output_grad, bias_grad)

    def execute(self, input_data, bias, kernel_data=None) -> np.array:
        if kernel_data is not None:
            self._K.data[:] = kernel_data

        self._bias.data[:] = bias
        self._R.data[:] = 0
        indices = [slice(0, self._I.shape[0], 1),
                   slice(0, self._I.shape[1], 1)]
        for i in range(self._ndims):
            indices.append(slice(self._padding[i],
                                 self._I.data.shape[2+i]-self._padding[i], 1))

        self._I.data[tuple(indices)] = input_data
        return super().execute()

    def equations(self):

        result_dimensions = self._R.dimensions
        kernel_dims = self._K.dimensions
        eqs = []
        input_dims = [result_dimensions[0], kernel_dims[1]]
        for i in range(0, self._ndims):
            input_dims.append(
                result_dimensions[-self._ndims + i]*self._stride[i] +
                kernel_dims[-self._ndims + i])

        eqs += [Inc(self._R[result_dimensions],
                    self._K[(result_dimensions[1],
                             *kernel_dims[1:])]*self._I[input_dims])]

        eqs.append(Inc(self._R[result_dimensions],
                   self._bias[result_dimensions[1]]))
        if self._activation is not None:
            eqs.append(Eq(self._R[result_dimensions],
                          self._activation(self._R[result_dimensions])))
        return (eqs, [])

    def backprop_equations(self, prev_layer, next_layer):
        layer = self

        kernel_dims = layer.kernel_gradients.dimensions
        bias_dims = layer.bias_gradients.dimensions
        result_grad_dims = layer.result_gradients.dimensions
        result_grad_shape = layer.result_gradients.shape

        input_dims = [result_grad_dims[0], kernel_dims[1]]
        eqs = []
        padded_shape = [0] * self._ndims
        for i in range(0, self._ndims):
            padded_shape[-self._ndims+i] = result_grad_shape[-self._ndims+i] \
                + (self._stride[0]-1)*(result_grad_shape[-self._ndims+i]-1)

        res_grad_dilated = Function(name=self.get_name("outgrad_dilated"),
                                    shape=(*result_grad_shape[0:2],
                                           *padded_shape),
                                    dimensions=self._output_grad_dilated_dims,
                                    space_order=(0),
                                    dtype=np.float64)
        dims = list(result_grad_dims)
        for i in range(0, layer._ndims):
            dims[2+i] = dims[2+i] + (dims[2+i] * (self._stride[i]-1))
        eqs += [Eq(res_grad_dilated[(dims)], layer.result_gradients)]

        for i in range(0, self._ndims):
            input_dims.append(
                kernel_dims[-self._ndims + i] +
                self._output_grad_dilated_dims[-self._ndims + i])

        eqs += [Inc(layer.kernel_gradients[kernel_dims],

                    res_grad_dilated[(result_grad_dims[0], kernel_dims[0],
                                      *self._output_grad_dilated_dims[2:]
                                      )]*self._I[input_dims])]

        eqs += [Inc(layer.bias_gradients[bias_dims],
                    layer.result_gradients[(result_grad_dims[0], bias_dims[0],
                                            *result_grad_dims[2:])])]

        if next_layer is not None:
            next_layer_dims = next_layer.result_gradients.dimensions
            padded_shape = [0] * self._ndims
            for i in range(0, self._ndims):
                padded_shape[-self._ndims+i] = result_grad_shape[
                    -self._ndims+i]\
                    + (2*(self._padding[0]+self._kernel_size[-1]-1)) + \
                    (self._stride[0]-1)*(result_grad_shape[-self._ndims+i]-1)
            self.op = output_grad_padded = Function(
                name=self.get_name("outgrad_padded"),
                shape=(*result_grad_shape[0:2], *padded_shape),
                dimensions=self._output_grad_padded_dimensions,
                space_order=(0), dtype=np.float64)
            dims = list(result_grad_dims)
            for i in range(0, layer._ndims):
                dims[2+i] = dims[2+i] - layer._padding[i] + \
                    (self._kernel_size[-1]-1) + dims[2+i] * (self._stride[0]-1)
            eqs += [Eq(output_grad_padded[(dims)], layer.result_gradients)]

            input_dims = [next_layer_dims[0], kernel_dims[0]]
            reversed_k_dims = [kernel_dims[0],
                               next_layer_dims[1], *kernel_dims[2:]]
            for i in range(0, self._ndims):
                input_dims.append(
                    next_layer_dims[-self._ndims + i] +
                    kernel_dims[-self._ndims + i])
                reversed_k_dims[-self._ndims+i] = \
                    reversed_k_dims[-self._ndims
                                    + i].symbolic_max - \
                    reversed_k_dims[-self._ndims+i]
            eqs += [Inc(next_layer.result_gradients[next_layer_dims],
                        self._K[(reversed_k_dims)]
                        * output_grad_padded[input_dims], implicit_dims=(
                            next_layer_dims[0], kernel_dims[0]))]

            eqs += next_layer.activation.backprop_eqs(next_layer)

        return (eqs, [])


class Conv3D(ConvV2):
    """
    A Layer subclass corresponding to a 3D convolution layer (mathematically,
    it performs a cross-correlation operation).

    Parameters
    ----------
    kernel_size : (int, int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,
                 stride=(1, 1, 1), padding=(0, 0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 3
        super().__init__(kernel_size, input_size, dimensions,
                         stride, padding, activation, generate_code,
                         strict_stride_check)


class Conv2DV2(ConvV2):
    """
    A Layer subclass corresponding to a 2D convolution layer (mathematically,
    it performs a cross-correlation operation).

    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 2
        super().__init__(kernel_size, input_size, dimensions,
                         stride, padding, activation, generate_code,
                         strict_stride_check)


class Pooling(Layer):
    """
    A Layer abstract subclass corresponding to a generic pooling layer.
    When you create a subclass of Pooling, you have to implement
    the following methods: equations(), backprop_equations().

    Parameters
    ----------
    kernel_size : (int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,  dimensions,
                 name_allocator_func=alloc, dim_allocator_func=dim_alloc,
                 stride=(1, 1), padding=(0, 0), activation=None,
                 generate_code=False, strict_stride_check=True):
        # Kernel size is expressed as (rows, columns).
        # Input size is expressed as (batch size, channels, rows, columns).
        self._ndims = dimensions
        self._dim_dict = {3: 'depth', 2: 'height', 1: 'width'}

        if (type(padding) is int):
            padding = tuple([padding] * self._ndims)
        if (type(stride) is int):
            stride = tuple([stride] * self._ndims)
        self._error_check(kernel_size, input_size, stride, padding,
                          strict_stride_check)

        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        super().__init__(self._kernel_size, input_size, activation,
                         alloc, dim_alloc,
                         generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding,
                     strict_stride_check):
        if input_size is None or (len(input_size) != self._ndims+2):
            raise Exception("Input size is incorrect")

        if kernel_size is None or (len(kernel_size) != self._ndims):
            raise Exception("Kernel size is incorrect")

        if stride is None or (len(stride) != self._ndims):
            raise Exception("Stride is incorrect")

        if padding is None or (len(padding) != self._ndims):
            raise Exception("Padding is incorrect")

        for i in range(0, self._ndims):

            if stride[i] < 1:
                raise Exception("Stride cannot be less than 1")

            if stride[i] is None or type(stride[i]) is not int:
                raise Exception("Stride must be an integer")

            if padding[i] < 0:
                raise Exception("Padding cannot be negative")

            if padding[i] is None or type(padding[i]) is not int:
                raise Exception("Padding must be an integer")

        if strict_stride_check:
            input_d = input_size[-self._ndims+i] + 2 * padding[i]
            if (input_d - kernel_size[-self._ndims+i]) % stride[i] != 0:
                raise Exception("Stride " + str(stride) + " is not "
                                "compatible with feature map, kernel and "
                                "padding sizes. If you want to proceed "
                                "anyway, set strict_stride_check=False "
                                "when instantiating this object")

    def _set_padding_result_values(self, input_func, result_func, value=0):
        devito_func_dims = input_func.dimensions[2:]
        eqs = []
        for i, dim in enumerate(devito_func_dims):
            dim_l = SubDimension.left(name='sub%s_l' % dim.name, parent=dim,
                                      thickness=self._padding[i])
            eqs.append(Inc(input_func.subs({dim: dim_l}), value))
            dim_r = SubDimension.right(name='sub%s_r' % dim.name, parent=dim,
                                       thickness=self._padding[i])
            eqs.append(Inc(input_func.subs({dim: dim_r}), value))
        eqs.append(Eq(result_func, value))
        op = Operator(eqs)
        op.apply()

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):

        dimensions = ['dbatch', 'dchannel']
        result_shape = []
        input_size = list(input_size)
        # generating  in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
        for i in range(0, self._ndims):
            result_d = (input_size[(-self._ndims+i)] -
                        kernel_size[(-self._ndims+i)] +
                        2 * self._padding[i])//self._stride[i] + 1
            result_shape.append(result_d)
            input_size[(-self._ndims+i)] += 2 * self._padding[i]
            dimensions.append(
                'd_'+self._dim_dict.get(self._ndims-i, self._ndims-i))

        result_shape = (*input_size[0:2], *result_shape)
        # input data function
        input_dimensions = [SpaceDimension("Input_"+x) for x in dimensions]

        input_func = Function(name="Input_F", shape=(input_size),
                              dimensions=input_dimensions,
                              space_order=0, dtype=np.float64)

        # Result for pool layer
        result_dimensions = [SpaceDimension("Result_"+x) for x in dimensions]

        result_func = Function(name="Result_F", shape=result_shape,
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)

        output_grad_dimensions = [SpaceDimension(
            "output_grad"+x) for x in dimensions]

        output_grad = Function(name="outgrad_%s" % name_allocator_func(
        ), shape=result_shape, dimensions=output_grad_dimensions,
            space_order=0, dtype=np.float64)

        self._set_padding_result_values(
            input_func, result_func, value=np.finfo(np.float32).min)

        return (None, input_func, result_func, None, None, output_grad, None)

    @property
    def stride(self):
        """Stride of the layer."""
        return self._stride

    @property
    def kernel_size(self):
        """The kernel size of the layer."""
        return self._kernel_size

    def execute(self, input_data):
        '''execute implementation'''
        indices = [slice(0, self._I.shape[0], 1),
                   slice(0, self._I.shape[1], 1)]

        for i in range(self._ndims):
            indices.append(slice(self._padding[i],
                                 self._I.data.shape[2+i]-self._padding[i], 1))
        self._I.data[tuple(indices)] = input_data

        return super().execute()

    @abstractmethod
    def equations(self):
        pass

    @abstractmethod
    def backprop_equations(self, prev_layer, next_layer):
        pass


class MaxPoolingV2(Pooling):
    """
    A Layer/Pooling subclass corresponding to a max pooling layer

    Parameters
    ----------
    See Pooling.__doc__.
    """

    def __init__(self, kernel_size, input_size, dims,
                 stride, padding, activation, generate_code,
                 strict_stride_check):
        self._indices = None
        self._forward_tmp_constants = None
        self._backward_tmp_constants = None
        super().__init__(kernel_size, input_size, dims, alloc, dim_alloc,
                         stride, padding, activation, generate_code,
                         strict_stride_check)

    def equations(self):
        result_dimensions = self._R.dimensions

        input_dims = [result_dimensions[0], result_dimensions[1]]
        args = []
        for i in range(0, self._ndims):
            new_dim = SpaceDimension(
                name='d_kernel'+self._dim_dict.get(
                    self._ndims-i, self._ndims-i))
            input_dims.append(
                result_dimensions[-self._ndims + i]*self._stride[i]+new_dim)
            args.append((new_dim.name + '_M', self._kernel_size[i] - 1))

        eqs = [Eq(self._R[result_dimensions], Max(
            self._R[result_dimensions], self._I[input_dims], evaluate=False))]

        return (eqs, args)

    def backprop_equations(self, prev_layer, next_layer):
        if next_layer is None:
            return ([], [])
        # TODO: Is the line below referencing alloc as a global?
        if self._backward_tmp_constants is None:
            self._backward_tmp_constants = \
                [Constant(name="bk_tmp_c_%s" % alloc(), dtype=np.int32),
                 Constant(name="bk_tmp_c_%s" % alloc(), dtype=np.int32)]

        dims = self._R.dimensions
        stride_rows, stride_cols = self.stride

        index = self._indices[dims[0], dims[1], dims[2], dims[3]]
        a = self._backward_tmp_constants[0]
        b = self._backward_tmp_constants[1]

        return ([Eq(a, index // 2),
                 Eq(b, index % 2),
                 Inc(next_layer.result_gradients[dims[0],
                                                 dims[1],
                                                 stride_rows * dims[2] + a,
                                                 stride_cols * dims[3] + b],
                     self.result_gradients[dims[0],
                                           dims[1], dims[2], dims[3]])] +
                next_layer.activation.backprop_eqs(next_layer), [])


class MaxPooling2D(MaxPoolingV2):

    """
    A Layer subclass corresponding to a 2D Max pool layer,
    which calls the generic pooling class.
    Parameters
    ----------
    kernel_size : (int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 2
        super().__init__(kernel_size, input_size, dimensions,
                         stride, padding, activation, generate_code,
                         strict_stride_check)


class MaxPooling3D(MaxPoolingV2):

    """
    A Layer subclass corresponding to a 2D Max pool layer,
    which calls the generic pooling class.
    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (depth, rows, columns).
    input_size : (int, int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, depth, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 3
        super().__init__(kernel_size, input_size, dimensions,
                         stride, padding, activation, generate_code,
                         strict_stride_check)


class UpSample(Layer):
    """
    A Layer subclass corresponding to upsampling layer
    Parameters
    ----------
    scale_factor : int or tuple of (input_dims)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (depth, rows, columns) or (rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    dimensions: int
        The dimension of conv operation i.e.
        if the convolution is 1d, 2d or so on
    """

    def __init__(self, scale_factor, input_size,
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).

        self._dims = len(input_size)-2
        if type(scale_factor) is int:
            scale_factor = tuple([scale_factor]*self._dims)

        self._error_check(scale_factor, input_size)

        self._scale_factor = scale_factor
        super().__init__(None, input_size, activation,
                         alloc, dim_alloc,
                         generate_code)

    def _error_check(self, scale_factor, input_size):
        if input_size is None or (len(input_size) < 3):
            raise Exception("Input size is incorrect")

        if scale_factor is None or (len(scale_factor) != len(input_size)-2):
            raise Exception("Kernel size is incorrect")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):

        self._dim_dict = dim_dict = {3: 'depth', 2: 'height', 1: 'width'}
        self._dims = len(input_size)-2
        dimensions = ['dbatch', 'dchannel']
        result_shape = []
        # generating  in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
        for i in range(0, self._dims):
            result_d = (input_size[(-self._dims+i)] * self._scale_factor[i])
            result_shape.append(result_d)
            dimensions.append('d_'+dim_dict.get(self._dims-i, self._dims-i))

        result_shape = (*input_size[0:2], *result_shape)

        # input data function
        input_dimensions = [SpaceDimension("Input_"+x) for x in dimensions]

        input_func = Function(name="Input_F",
                              shape=(input_size), space_order=0,
                              dimensions=input_dimensions, dtype=np.float64)

        result_dimensions = [SpaceDimension("Result_"+x) for x in dimensions]

        result_func = Function(name="Result_F", shape=result_shape,
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)
        
        result_grad = Function(name="Result_G",
                               shape=result_shape,
                               dimensions=result_dimensions,
                               space_order=0, dtype=np.float64)
        return (None, input_func, result_func, None, None,
                result_grad, None)

    def execute(self, input_data) -> np.array:

        self._I.data[:] = input_data
        self._R.data[:] = 0
        return super().execute()

    def equations(self):

        input_dimensions = self._I.dimensions
        res_dims = [input_dimensions[0], input_dimensions[1]]
        args = []
        for i in range(0, self._dims):
            new_dim = SpaceDimension(
                name='d_upsample'+self._dim_dict.get
                (self._dims-i, self._dims-i))
            res_dims.append(
                input_dimensions
                [-self._dims + i]*self._scale_factor[i]+new_dim)
            args.append((new_dim.name + '_M', self._scale_factor[i]-1))

        eqs = [Eq(self._R[res_dims], self._I[input_dimensions])]

        return (eqs, args)

    def backprop_equations(self, prev_layer, next_layer):
        # if next_layer is None:
        #     return ([], [])

        dims = self._R.dimensions

        if next_layer is not None:
            eqs = [Eq(next_layer._RG[dims], self.result_gradients[dims])]
            return (eqs, [])



class InstanceNorm(Layer):
    """
    A Layer subclass corresponding to convolution layer (mathematically,
    it performs a cross-correlation operation).
    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    dimensions: int
        The dimension of conv operation i.e.
        if the convolution is 1d, 2d or so on
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).
        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.
        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, input_size, dimensions,
                 activation, generate_code,
                 strict_stride_check):
        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).
        self._dims = dimensions
        self.mean = None
        self.var = None
        self.epsilon = 0.00001
        self._error_check(input_size)

        super().__init__(None, input_size, activation,
                         alloc, dim_alloc,
                         generate_code)

    def _error_check(self, input_size):
        if input_size is None or (len(input_size) != self._dims+2):
            raise Exception("Input size is incorrect")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):

        dim_dict = {3: 'depth', 2: 'height', 1: 'width'}

        dimensions = ['dbatch', 'dchannel']
        result_shape = []
        # generating  in the order depth, height, width ,
        # hence arr[-3], arr[-2] and so on
        for i in range(0, self._dims):
            result_d = (input_size[(-self._dims+i)])
            result_shape.append(result_d)
            dimensions.append('d_'+dim_dict.get(self._dims-i, self._dims-i))

        result_shape = (input_size[0], input_size[1], *result_shape)

        # input data function
        input_dimensions = [SpaceDimension("Input_"+x) for x in dimensions]

        input_func = Function(name="Input_F", shape=(input_size),
                              dimensions=input_dimensions, space_order=0, dtype=np.float64)

        # Result for convolution
        result_dimensions = [SpaceDimension("Result_"+x) for x in dimensions]

        result_func = Function(name="Result_F", shape=result_shape,
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)

        bias_dimensions = [SpaceDimension("bias_"+x) for x in ['d']]

        bias = Function(name="bias_F", shape=(
            input_size[1],), dimensions=bias_dimensions, space_order=0,
            dtype=np.float64)

        output_grad_dimensions = [SpaceDimension(
            "output_grad"+x) for x in dimensions]

        output_grad = Function(name="outgrad_%s" % name_allocator_func(
        ), shape=result_shape, dimensions=output_grad_dimensions,
            space_order=0, dtype=np.float64)

        bias_grad_dimensions = [SpaceDimension("bias_"+x) for x in ['d']]

        bias_grad = Function(name="bgrad_%s" % name_allocator_func(), shape=(
            input_size[1],), dimensions=bias_grad_dimensions, space_order=0,
            dtype=np.float64)

        return (None, input_func, result_func, bias, None,
                output_grad, bias_grad)

    def execute(self, input_data) -> np.array:

        self._I.data[:] = input_data
        self._R.data[:] = 0

        return super().execute()

    def equations(self):

        result_dimensions = self._R.dimensions
        result_shape = self._R.shape
        bias = self._bias.dimensions
        k_dims_offsets = []
        for i in range(0, self._dims):
            k_dims_offsets.append(
                list(range(0, result_shape[-self._dims + i])))


        # indices of kernel matrix for convolution
        k_indices = product(* k_dims_offsets)

        # sum = Function(name="mean_sum",
        #                space_order=0, shape=result_shape,
        #                dimensions=result_dimensions)
        # mean = [sum[(*result_dimensions[:2], *x)] for x in k_indices]
        
        temp_func = Function(name="Ones_Filter", shape=result_shape,
                             dimensions=result_dimensions, space_order=0,
                             dtype=np.float64)

        k_indices = product(* k_dims_offsets)
        # indices of input based on resullt matrix for convolution
        r_indicies = product(*k_dims_offsets)

        temp_func.data[:] = 1
        weight_matrix = sp.Matrix(
            [temp_func[(*result_dimensions[:2], *x)] for x in k_indices])

        r_indices_matrix = sp.Matrix(
            [self._I[(*result_dimensions[:2], *x)] for x in r_indicies])
        N = np.prod(result_shape[2:])

        # stencil operation corresponding to the convolution with kernel of input_shape with value to simulate sum of input_mat
        sum_input_sten = weight_matrix.dot(r_indices_matrix)

        self.mean = (sum_input_sten/N)
        '''
        .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} 
        '''
        
        # deviation from mean
        eqs = [Eq(self._R[result_dimensions], self._I[result_dimensions] - self.mean)]
        r_indicies = product(*k_dims_offsets)
        r_indices_matrix = sp.Matrix(
            [self._R[(*result_dimensions[0:2], *x)]**2 for x in r_indicies])

        sum_var_stencil = r_indices_matrix.dot(weight_matrix)
        self.var = sp.sqrt(sum_var_stencil/N+self.epsilon)
        eqs += [Eq(self._I[result_dimensions], sum_var_stencil/N)]
        epsilon = 0.00001
        eqs += [Eq(self._R[result_dimensions], self._R[result_dimensions] /
                   sp.sqrt(self._I[result_dimensions]+epsilon))]

        eqs.append(Inc(self._R[result_dimensions], self._bias[bias]))

        if self._activation is not None:
            eqs.append(Eq(self._R[result_dimensions],
                          self._activation(self._R[result_dimensions])))
        return (eqs, [])

    def backprop_equations(self, prev_layer, next_layer):

        result_dimensions = self._R.dimensions
        result_shape = self._R.shape
        var_g = None
        mean_g = None
        k_dims_offsets = []
        for i in range(0, self._dims):
            k_dims_offsets.append(
                list(range(0, result_shape[-self._dims + i])))


        # indices of kernel matrix for convolution
        k_indices = product(* k_dims_offsets)

        # sum = Function(name="mean_sum",
        #                space_order=0, shape=result_shape,
        #                dimensions=result_dimensions)
        # mean = [sum[(*result_dimensions[:2], *x)] for x in k_indices]
        
        temp_func = Function(name="Ones_Filter", shape=result_shape,
                             dimensions=result_dimensions, space_order=0,
                             dtype=np.float64)

        k_indices = product(* k_dims_offsets)
        # indices of input based on resullt matrix for convolution
        r_indicies = product(*k_dims_offsets)

        temp_func.data[:] = 1
        weight_matrix = sp.Matrix(
            [temp_func[(*result_dimensions[:2], *x)] for x in k_indices])

        r_indices_matrix = sp.Matrix(
            [self._I[(*result_dimensions[:2], *x)] for x in r_indicies])
        N = np.prod(result_shape[2:])
        sum_input_sten = weight_matrix.dot(r_indices_matrix)

        mean_g = (sum_input_sten/(N*self.var))
        double = 2
        if next_layer is not None:
            eqs = [Eq(self.result_gradients[result_dimensions], self.result_gradients[result_dimensions]/self.var)]
            eqs.append([Eq(var_g, self.result_gradients[result_dimensions]*(self._R[result_dimensions-self.mean])/(self.var*sp.sqrt(self.var)*double))])
            eqs.append([Eq(var_g[result_dimensions], var_g[result_dimensions]*(2*(self._R[result_dimensions]-self.mean)/N))])
            eqs.append([Inc(self.result_gradients[result_dimensions], var_g[result_dimensions])])
            eqs.append([Inc(mean_g[result_dimensions], var_g[result_dimensions]*(-2*self.mean())/N)])
            eqs.append([Inc(self.result_gradients[result_dimensions], mean_g[result_dimensions]/N)])
            eqs.append([Eq(next_layer.result_gradients[result_dimensions], self.result_gradients[result_dimensions])])
            return (eqs, [])


class InstanceNorm3D(InstanceNorm):
    """
    A Layer subclass corresponding to a 3D convolution layer (mathematically,
    it performs a cross-correlation operation).
    Parameters
    ----------
    kernel_size : (int, int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0, 0).
        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.
        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, input_size,
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 3
        super().__init__(input_size, dimensions,
                         activation, generate_code,
                         strict_stride_check)


class InstanceNorm2D(InstanceNorm):
    """
    A Layer subclass corresponding to a 2D convolution layer (mathematically,
    it performs a cross-correlation operation).
    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).
        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.
        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, input_size,
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        dimensions = 2
        super().__init__(input_size, dimensions,
                         activation, generate_code,
                         strict_stride_check)


class add(Layer):
    def __init__(self, input_size,  layer=None, activation=None,
                 name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc,
                 generate_code=False):
        self.layer = layer
        super().__init__(None, input_size, activation,
                         name_allocator_func, dim_allocator_func,
                         generate_code)
    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):


        self.dim_dict = dim_dict = {3: 'depth', 2: 'height', 1: 'width'}

        dimensions = ['dbatch', 'dchannel', 'depth', 'height', 'width']
        result_shape = []
        input_size = list(input_size)


        result_shape = input_size

        # input data function
        input_dimensions = [SpaceDimension("Input_"+x) for x in dimensions]

        input_func = Function(name="Input_F", shape=(input_size),
                              dimensions=input_dimensions, space_order=0,
                              dtype=np.float64)


        # Result function
        result_dimensions = [SpaceDimension(
            ("Result_"+x)) for x in dimensions]

        result_func = Function(name="Result_F", shape=(result_shape),
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)

        output_grad_dimensions = [SpaceDimension(("output_grad"+x)) for x in dimensions]

        output_grad = Function(name=("out_grad_"),
                               shape=result_shape,
                               dimensions=output_grad_dimensions,
                               space_order=0, dtype=np.float64)



        return (None, input_func, result_func, None, None,
                output_grad, None)
    
    def execute(self, input_data) -> np.array:
        self._I.data[:] = input_data

        return super().execute()

    def equations(self):
        dims = self._I.dimensions
        # eqs = [Inc(self._I[dims], self._R[dims])]
        self._R = self.layer._R
        # eqs = [Eq(self._R[dims], self.layer.result[dims])]
        eqs = [Inc(self._R[dims], self._I[dims])]

        return (eqs, [])
    
    def backprop_equations(self, prev_layer, next_layer):
        if next_layer is not None:
            eqs = [Eq(next_layer.result_gradients, self.result_gradients)]
            return (eqs, [])

class cat(Layer):
    def __init__(self, input_size,  layer=None, activation=None,
                 name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc,
                 generate_code=False):
        self.layer = layer
        super().__init__(None, input_size, activation,
                         name_allocator_func, dim_allocator_func,
                         generate_code)
    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):


        self.dim_dict = dim_dict = {3: 'depth', 2: 'height', 1: 'width'}

        dimensions = ['dbatch', 'dchannel', 'depth', 'height', 'width']
        result_shape = []
        input_size = list(input_size)

        result_shape = [input_size[0], input_size[1]*2, input_size[2], input_size[3], input_size[4]]

        # input data function
        input_dimensions = [SpaceDimension("Input_"+x) for x in dimensions]

        input_func = Function(name="Input_F", shape=(input_size),
                              dimensions=input_dimensions, space_order=0,
                              dtype=np.float64)


        # Result function
        result_dimensions = [SpaceDimension(
            ("Result_"+x)) for x in dimensions]

        result_func = Function(name="Result_F", shape=(result_shape),
                               dimensions=result_dimensions, space_order=0,
                               dtype=np.float64)

        output_grad_dimensions = [SpaceDimension(("output_grad"+x)) for x in dimensions]

        output_grad = Function(name=("out_grad_"),
                               shape=result_shape,
                               dimensions=output_grad_dimensions,
                               space_order=0, dtype=np.float64)



        return (None, input_func, result_func, None, None,
                output_grad, None)
    
    def execute(self, input_data) -> np.array:
        self._I.data[:] = input_data
        self._R.data[:] = 0
        return super().execute()

    def equations(self):
        dims = self._I.dimensions

        cat_dims = self.layer._R.dimensions
        self._R = self.layer._R

        eqs = [Inc(self._R[(cat_dims[0],dims[1]+cat_dims[1],*cat_dims[2:])],self._I[dims])]

        return (eqs, [])
    
    def backprop_equations(self, prev_layer, next_layer):
        if next_layer is not None:
            eqs = [Eq(next_layer.result_gradients, self.result_gradients)]
            return (eqs, [])