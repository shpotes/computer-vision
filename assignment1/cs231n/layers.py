from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    N, *d = x.shape
    D, M = w.shape
    x = x.reshape(N, D)

    out = x @ w + b 
    cache = (x.reshape(N, *d), w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N, *d = x.shape
    D, M = w.shape

    dx = (dout @ w.T).reshape((N, *d))
    dw = x.reshape(N, D).T @ dout
    db = dout.sum(axis=0)
    
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(x, 0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = dout * (cache > 0), cache
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var ????

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features 

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'train':
        batch_mean = x.mean(axis=0)
        var = x.var(axis=0)
        batch_std = np.sqrt(var + eps)

        out = gamma * (x - batch_mean) / batch_std + beta
        
        cache = x, gamma, beta, batch_mean, var, eps

        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var = momentum * running_var + (1 - momentum) * batch_std

        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
        
        cache = x, gamma, beta, batch_mean, var, eps, bn_param


    elif mode == 'test':
        out = gamma * (x - running_mean)/running_var
        out += beta

        cache = x, gamma, beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    
    See https://arxiv.org/pdf/1502.03167.pdf page 4
    """
    N, D = dout.shape
    x, gamma, beta, mu, var, eps, bn_param = cache

    dhat = dout * gamma
    dsigma = np.sum(dhat * (x - mu), axis=0) * -1/2 * (var + eps) ** (- 3 / 2)
    dmu = np.sum(dhat, axis=0) * (- 1 / np.sqrt(var + eps))
    dmu += dsigma * np.sum(-2 * (x - mu), axis=0) / N
    dx = dhat * 1/np.sqrt(var + eps) + dsigma * 2 *  (x - mu)/N + dmu / N 
    dgamma = np.sum(dout * (x - mu)/np.sqrt(var + eps), axis=0)
    dbeta = np.sum(dout, axis=0)
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    eps = ln_param.get('eps', 1e-5)
    
    ft_mean = x.mean(axis=1)
    ft_var = x.var(axis=1)
    ft_std = np.sqrt(ft_var + eps)

    out = ((gamma * (x.T - ft_mean).T).T / ft_std).T + beta

    cache = x, gamma, beta, ft_mean, ft_var, eps

    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    
    N, D = dout.shape
    x, gamma, beta, mu, var, eps = cache

    dhat = dout * gamma # 4, 5
    dsigma = np.sum(dhat * (x.T - mu).T, axis=1) * -1/2 * \
        (var + eps) ** (- 3 / 2) 
    dmu = np.sum(dhat, axis=1) * (- 1 / np.sqrt(var + eps)) + dsigma * \
        np.sum(-2 * (x.T - mu).T, axis=1) / D
    dx = dhat.T* 1/np.sqrt(var + eps) + dsigma * 2 *  (x.T - mu)/D + dmu / D
    dgamma = np.sum(dout * ((x.T - mu)/np.sqrt(var + eps)).T, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx.T, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = mask * dout
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    pad, stride = conv_param['pad'], conv_param['stride']
    x_old = x.copy()
    x = np.pad(x, pad, 'constant')[pad:-pad,pad:-pad,:,:]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_p = 1 + (H - HH) // stride
    W_p = 1 + (W - WW) // stride
    B = b
    out = np.zeros((N, F, H_p, W_p))

    for f in range(F):
        a = 0; c = HH; 
        for i in range(H_p):
            b = 0; d = WW;
            for j in range(W_p):
                tmp = x[:,:,a:c,b:d] * w[f,:,:,:].reshape(1, C, HH, WW) 
                out[:,f,i,j] = tmp.sum(axis=(1,2,3)) + B[f]
                b += stride; d += stride
            a += stride; c += stride
            
                

    cache = (x_old, w, B, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """

    x, w, B, conv_param = cache
    pad, stride = conv_param['pad'], conv_param['stride']
    x = np.pad(x, pad, 'constant')[pad:-pad,pad:-pad,:,:]
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_p = 1 + (H - HH) // stride
    W_p = 1 + (W - WW) // stride

    db = dout.sum(axis=(0,2,3))
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    #print(w.shape)

    #print(x.shape)
    for f in range(F):
        a = 0; c = HH; 
        for i in range(H_p):
            b = 0; d = WW;
            for j in range(W_p):
                dx[:,:,a:c,b:d] += (w[f,:,:,:].reshape(1, *w[f,:,:,:].shape).T * dout[:,f,i,j]).T
                dw[f,:,:,:] += (x[:,:,a:c,b:d].T * dout[:,f,i,j]).T.sum(axis=0)
                b += stride; d += stride
            a += stride; c += stride

    return dx[:,:,pad:-pad,pad:-pad], dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """

    WW, HH, stride = pool_param['pool_width'], pool_param['pool_height'], pool_param['stride']
    N, C, H, W = x.shape
    H_p = 1 + (H - HH) // stride
    W_p = 1 + (W - WW) // stride

    out = np.zeros((N, C, H_p, W_p))

    a = 0; c = HH; 
    for i in range(H_p):
        b = 0; d = WW;
        for j in range(W_p):
            out[:,:,i,j] = x[:,:,a:c,b:d].max(axis=(2,3))
            b += stride; d += stride
        a += stride; c += stride

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    
    WW, HH, stride = pool_param['pool_width'], \
        pool_param['pool_height'], pool_param['stride']
    N, C, H, W = x.shape
    H_p = 1 + (H - HH) // stride
    W_p = 1 + (W - WW) // stride

    dx = np.zeros(x.shape)
    a = 0; c = HH; 
    for i in range(H_p):
        b = 0; d = WW;
        for j in range(W_p):
            #out[:,:,i,j] = x[:,:,a:c,b:d].max(axis=(2,3))
            maxi = x[:,:,a:c,b:d].max(axis=(2,3))
            mask = np.equal(x[:,:,a:c,b:d].T, maxi.T)
            dx[:,:,a:c,b:d][mask.T] += dout[:,:,i,j].flatten()
            b += stride; d += stride
        a += stride; c += stride
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    x = x.reshape(N * H * W, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
  

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    N, C, H, W = dout.shape
    dout = dout.reshape(N * H * W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    dx = dx.reshape(N, C, H, W)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner 
    identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape
    eps = gn_param.get('eps',1e-5)
    x = x.reshape(N, G, C // G, H, W)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    gn_mean = x.mean(axis=2)
    gn_var = x.var(axis=2)
    gn_std = np.sqrt(gn_var + eps)
    
    out = (x - gn_mean.reshape(N, G, 1, H, W)) / gn_std.reshape(N, G, 1, H, W)
    out = gamma * out.reshape(N, C, H, W) + beta

    cache = x, gamma, beta, gn_mean, gn_var, gn_std, eps, G

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape
    x, gamma, beta, mu, var, std, eps, G = cache
    D = C // G
    
    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    dhat = dout * gamma
    dhat = dhat.reshape(N, G, D, H, W)
    mu = mu.reshape(N, G, 1, H, W)
    dsigma = np.sum(dhat * (x - mu), axis=2) \
                    * -1/2 * (var + eps) ** (-3/2)

    dmu = (np.sum(dhat, axis=2) * (- 1 / np.sqrt(var + eps)) + dsigma * \
        np.sum(-2 * (x - mu), axis=2) / D).reshape(N, G, 1, H, W)

    std = std.reshape(N, G, 1, H, W)
    dsigma = dsigma.reshape(N, G, 1, H, W)
    dx = (dhat * 1/std + dsigma * 2 * (x - mu)/D + dmu / D).reshape(N, C, H, W)

    x = x.reshape(N, C, H, W)

    dgamma = np.sum(dout * (x - mu)/std, axis=(0,1,3,4)).reshape(1, C, 1, 1)
    dbeta = np.sum(dout.reshape(N, C, H, W), axis=(0,2,3)).reshape(1, C, 1, 1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
