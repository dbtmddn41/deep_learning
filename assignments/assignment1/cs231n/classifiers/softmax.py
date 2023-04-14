from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W
    dscores = np.zeros_like(scores)
    num_train = scores.shape[0]
    num_classes = scores.shape[1]
    for i in range(num_train):
      scores[i] -= np.max(scores[i])
      sum_exp_score = 0.0
      sfmx_yi = np.zeros(num_classes)
      for j in range(num_classes):
        exp_score = np.exp(scores[i, j])
        sum_exp_score += exp_score
        sfmx_yi[j] = exp_score
        dscores[i,j] = exp_score
      sfmx_yi /= sum_exp_score
      loss += -np.log(sfmx_yi[y[i]])

      dscores[i] /= sum_exp_score
      dscores[i, y[i]] -= 1
    
    loss /= num_train

    dW = X.T @ dscores
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W
    num_train = scores.shape[0]
    num_classes = scores.shape[1]
    scores -= np.max(scores, axis=1).reshape(-1,1)
    exp_scores = np.exp(scores)
    sfmx_y = exp_scores / np.sum(exp_scores, axis=1).reshape(-1,1)
    loss = -np.log(sfmx_y[np.arange(num_train), y])
    loss = np.mean(loss)

    dscores = np.copy(sfmx_y)
    dscores[np.arange(num_train), y] -= 1

    dW = X.T @ dscores
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg*2*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
