from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

from operator import mul
from functools import reduce

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.params['W1'] = np.random.normal(size=(num_filters,
                                                   input_dim[0],
                                                   filter_size,
                                                   filter_size),
                                             scale=weight_scale)
        self.params['W2'] = np.random.normal(size=(num_filters * \
                                                   input_dim[1] // 2 *\
                                                   input_dim[2] // 2,
                                                   hidden_dim),
                                             scale=weight_scale)
        self.params['W3'] = np.random.normal(size=(hidden_dim,
                                                   num_classes),
                                             scale=weight_scale)

        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # conv - relu - 2x2 max pool - affine - relu - affine - softmax


        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}


        h1, c1 = conv_forward_im2col(X, W1, b1, conv_param) #
        h1, r1 = relu_forward(h1)
        h1, p1 = max_pool_forward_fast(h1, pool_param) #
        max_pool_shape = h1.shape
        h1 = h1.reshape(X.shape[0], -1)
        h2, c2 = affine_relu_forward(h1, W2, b2)
        scores, c3 = affine_forward(h2, W3, b3)

        if y is None:
            return scores

        loss, dx = softmax_loss(scores, y)

        loss += self.reg / 2 * (self.params['W1']**2).sum()
        loss += self.reg / 2 * (self.params['W2']**2).sum()
        loss += self.reg / 2 * (self.params['W3']**2).sum()

        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        grads = {}
        dx, grads['W3'], grads['b3'] = affine_backward(dx, c3)
        grads['W3'] += self.reg * self.params['W3']
        dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, c2)
        dx = dx.reshape(max_pool_shape)
        dx = max_pool_backward_fast(dx, p1)
        dx = relu_backward(dx, r1)
        dx, grads['W1'], grads['b1'] = conv_backward_im2col(dx, c1)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
