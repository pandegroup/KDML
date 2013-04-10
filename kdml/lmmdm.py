"""
Large-Margin Mahalanobis Metric

Implements the algorithm from
Shen, C; Jim, J, Wang, Lei, Scalable Large-Margin Mahalanobis Distance Metric Learning,
IEEE Transactions on Neural Networks, 21 1524 (2010)

Also contains the ability to learn a diagonal weighted euclidean distance metric
using the same objective function with a simpler optimizer.
"""

import sys, os
import numpy as np
import warnings
from scipy.optimize import fmin_tnc, fmin_bfgs, fmin_l_bfgs_b
from scipy.optimize.linesearch import line_search_wolfe1
import scipy.sparse
from scipy.sparse.linalg import eigs as sparse_eigs
from scipy.sparse.linalg import eigsh as sparse_eigsh
from memoize import memoized
from msmbuilder import metrics
import scipy.weave
import scipy.linalg.blas


def first_eigen(matrix):
    """Get the first (largest) eigenvalue/eigenvector of a symmetric real matrix

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric, real, square matrix

    Returns
    -------
    eigenvalue : float
        The eigenvalue
    eigenvector : ndarray
        The eigenvector is returned as a row 1D row vector
    """
    n_dims = matrix.shape[0]
    value, vector = scipy.linalg.eigh(matrix, eigvals=(n_dims-1, n_dims-1))

    return np.float(value), vector.flatten()


def _diagonal_margin(omegas, a_to_b, a_to_c):
    """Compute the difference in the distances from a to b and a to c under the
    metric parameterized by the omegas, over all triplets

    Parameters
    ----------
    a_to_b : np.ndarray, dtype=float shape=[n_triplets, n_features]
        The difference from the "a" to "b" conformations along each feature.
        The "a" and "b" conformations, for each triplet, are the pair that
        are supposed to be close together.
    a_to_b : np.ndarray, dtype=float shape=[n_triplets, n_features]
        The difference from the "a" to "c" conformations along each feature.
        The "a" and "c" conformations, for each triplet, are the pair that
        are supposed to be far apart.
    omegas : np.ndarray, dtype=float, shape=[n_features]

    Returns
    -------
    margin: np.ndarray, dtype=float, shape=[n_triplets]
        d(a, c) - d(a, b) for each of the triplets.

    Examples
    --------
    >>> a_to_b = np.array([[1,1,1]])
    >>> a_to_c = np.array([[2,2,2]])
    >>> omegas = np.array([1,2,3])

    >>> sq_triplets_b2c = np.square(np.matrix(a_to_c.T)) - np.square(np.matrix(a_to_b.T))
    >>> margin = (np.diag(omegas)*sq_triplets_b2c).sum(axis=0)
    >>> float(margin)
    18.0
    >>> float(_diagonal_margin(omegas, a_to_b, a_to_c))
    18.0
    """
    assert a_to_b.shape == a_to_c.shape
    n_triplets, n_features = a_to_b.shape

    margin = np.zeros(n_triplets)
    scipy.weave.inline("""
    int i, j;
    double d1, d2;

    for (i = 0; i < n_triplets; i++) {
        for (j = 0; j < n_features; j++) {
            d1 = (a_to_b[i*n_features + j]) * (a_to_b[i*n_features + j]);
            d2 = (a_to_c[i*n_features + j]) * (a_to_c[i*n_features + j]);
            margin[i] += omegas[j] * (d2 - d1);
        }
    }
    """, ['a_to_b', 'a_to_c', 'omegas', 'n_triplets', 'n_features', 'margin'])
    return margin


def _diagonal_margin_deriv(omegas, a_to_b, a_to_c):
    return (np.square(a_to_c) - np.square(a_to_b)).T


def square_loss(vector):
    """Squared hinge loss function

    0      if v >= 0
    v**2   else

    Parameters
    ----------
    vector : array_like
        Input vector or float

    Returns
    -------
    loss : array_like
        Output vector or float, of same shape as input. Elementwise operation

    """
    return np.square(np.minimum(vector, 0))

def square_loss_deriv(vector):
    """Derivative of the square loss function, elementwise

    Examples
    --------
    >>> x = np.linspace(-1.5, 1.5)
    >>> h = 1e-7
    >>> numerical = (square_loss(x + h) - square_loss(x)) / h
    >>> analytical = square_loss_deriv(x)
    >>> np.all(np.abs(analytical - numerical) < 1e-6)
    True
    """
    return 2 * np.minimum(vector, 0)


def huber_loss(vector, h=0.5):
    """Huber loss function. A smoothed l1 hinge loss

    0                   if v >= h
    (h - v)**2 / (4h)   if -h < v < h
    -v                  if v <= -h

    Parameters
    ----------
    vector : array_like
        Input vector or float

    Returns
    -------
    loss : array_like
        Output vector or float, of same shape as input. Elementwise operation
    """
    term1 = np.square(h - vector) / (4*h) * ((vector < h) & ( vector > -h))
    term2 = -vector * (vector <= -h)
    huber = term1 + term2
    return huber


def huber_loss_deriv(vector, h=0.5):
    """ Derivative of the huber loss function, elementwise

    Examples
    --------
    >>> x = np.linspace(-1.5, 1.5)
    >>> h = 1e-7
    >>> numerical = (huber_loss(x + h) - huber_loss(x)) / h
    >>> analytical = huber_loss_deriv(x)
    >>> np.all(np.abs(analytical - numerical) < 1e-6)
    True
    """
    term1 = (vector - h) / (2*h) * ((vector < h) & ( vector > -h))
    term2 = -1 * (vector <= -h)
    huber_deriv = term1 + term2
    return huber_deriv


def _diagonal_objgrad(rho_and_omegas, a_to_b, a_to_c, alpha, loss='huber'):
    """Diagonal objective function, and its gradient, in raw coordinates

    Parameters
    ----------
    rho_and_omegas : np.ndarray, shape=[n_features + 1]
        The first entry is rho, the target margin. The rest of the entries
        are the weights, omega_i which run over all of the features and are
        the diagonal entries of the Mahalanobis matrix.

    Returns
    -------
    objective : float
        The objective function
    grad : np.ndarry, shape=[n_features + 1]
        The first entry is the derivative of the objective function
        w.r.t \rho, and the rest of the entries are the derivative with respect
        to each element of \omega.

    Notes
    -----
    This is not in transformed coordinates
    """

    assert a_to_b.shape == a_to_c.shape
    n_triplets, n_features = a_to_b.shape

    # extract the variables
    rho = rho_and_omegas[0]
    omegas = rho_and_omegas[1:len(rho_and_omegas)]
    # sanity check
    assert len(omegas) == n_features
    if loss == 'huber':
        f_loss = huber_loss; g_loss = huber_loss_deriv
    elif loss == 'square':
        f_loss = square_loss; g_loss = square_loss_deriv
    else:
        raise NotImplementedError('sorry')

    margin = _diagonal_margin(omegas, a_to_b, a_to_c)
    print 'Classification accuracy: %d/%d=%.5f' % (np.count_nonzero(margin > 0),
        n_triplets, np.count_nonzero(margin > 0) / float(n_triplets))

    loss_value = f_loss(margin - rho)
    loss_deriv = g_loss(margin - rho)

    #print alpha, rho, loss_value, n_triplets
    objective = alpha * rho - np.sum(loss_value) / float(n_triplets)

    #print 'objective', objective

    # compute the gradient
    grad_rho = alpha + np.sum(loss_deriv) / float(n_triplets)
    grad_omegas = - np.dot(_diagonal_margin_deriv(omegas, a_to_b, a_to_c), loss_deriv) / float(n_triplets)

    # pack the gradients into a single return value
    grad = np.empty(n_features + 1)
    grad[0] = grad_rho
    grad[1:] = grad_omegas

    return objective, grad


def _diagonal_objgrad_transformed(R_and_Ws, a_to_b, a_to_c, alpha, loss='huber'):
    """Diagonal objective function, and its gradient, in transformed coordinates

    Parameters
    ----------
    R_and_Ws : np.ndarray, shape=[n_features + 1]
        The first entry is R, the square root of the target margin.
        The rest of the entries are the transformed weights, W_i which run over
        all of the features and are the diagonal entries of the Mahalanobis
        matrix. The transformed weights W are related to the real weights omega
        by omega_i = \frac{W_i**2}{\sum_j{W_j**2}}. This keeps the weights positive
        and summing to one.

    Returns
    -------
    objective : float
        The objective function
    grad : np.ndarry, shape=[n_features + 1]
        The first entry is the derivative of the objective function
        w.r.t R, and the rest of the entries are the derivative with respect
        to each element of W.
    """

    rho = R_and_Ws[0]**2
    W = R_and_Ws[1:]
    W2 = np.square(W)
    omegas = W2 / np.sum(W2)

    rho_and_omegas = np.zeros(1 + len(omegas))
    rho_and_omegas[0] = rho
    rho_and_omegas[1:] = omegas

    objective, grad_rho_omegas = _diagonal_objgrad(rho_and_omegas, a_to_b, a_to_c, alpha, loss)

    partial_R = 2*R_and_Ws[0]*grad_rho_omegas[0]
    grad_omegas = grad_rho_omegas[1:]

    jacobian = -2 * np.matrix(W).T * np.matrix(W2) / np.square(np.sum(W2))
    jacobian += np.diag(2*W / np.sum(W2))
    grad_W = np.array(jacobian * np.matrix(grad_omegas).T)[:,0]

    return objective, np.hstack(([partial_R], grad_W))


def _test_diagonal_objgrad():
    rho_and_omegas = np.random.randn(1+10)

    h = 1e-7
    a_to_b = np.random.randn(5,10)
    a_to_c = 1 + np.random.randn(5,10)
    alpha = 1

    def test(loss):
        for i in range(len(rho_and_omegas)):
            rho_and_omegas2 = np.copy(rho_and_omegas)
            rho_and_omegas2[i] += h
            obj1, grad1 = _diagonal_objgrad(rho_and_omegas,  a_to_b, a_to_c, alpha, loss=loss)
            obj2, grad2 = _diagonal_objgrad(rho_and_omegas2, a_to_b, a_to_c, alpha, loss=loss)

            assert ((obj2 - obj1) / h - grad1[i]) < 1e-6

    test('square')
    test('huber')
    print 'passed'

def _test_diagonal_objgrad_transformed():
    R_and_Ws = np.random.randn(1+10)

    h = 1e-7
    a_to_b = np.random.randn(5,10)
    a_to_c = 1 + np.random.randn(5,10)
    alpha = 1

    def test(loss):
        for i in range(len(R_and_Ws)):
            R_and_Ws2 = np.copy(R_and_Ws)
            R_and_Ws2[i] += h
            obj1, grad1 = _diagonal_objgrad_transformed(R_and_Ws,  a_to_b, a_to_c, alpha, loss=loss)
            obj2, grad2 = _diagonal_objgrad_transformed(R_and_Ws2, a_to_b, a_to_c, alpha, loss=loss)

            assert ((obj2 - obj1) / h - grad1[i]) < 1e-6

    test('square')
    test('huber')
    print 'passed'


def optimize_diagonal(a_to_b, a_to_c, alpha, initial_weights='uniform', loss='huber'):
    """Optimize a diagonal mahanalonis distance metric in the large-margin framework

    Parameters
    ----------
    a_to_b : np.ndarray, dtype=float shape=[n_triplets, n_features]
        The difference from the "a" to "b" conformations along each feature.
        The "a" and "b" conformations, for each triplet, are the pair that
        are supposed to be close together.
    a_to_b : np.ndarray, dtype=float shape=[n_triplets, n_features]
        The difference from the "a" to "c" conformations along each feature.
        The "a" and "c" conformations, for each triplet, are the pair that
        are supposed to be far apart.
    alpha : float
        The weight given to maximizing the margin in the objective function.
        This trades off the desire to minimize the loss on triplets that are
        classified with a margin less than the target margin rho.
    initial_weights : np.ndarray
    loss_function : huber, square
        The type of loss function you want

    Returns
    -------
    rho : float
    omegas : np.ndarray, dtype=float, shape=[n_features]
    """

    if not loss in ['huber', 'square']:
        raise ValueError('Not supported')
    if not a_to_b.shape == a_to_c.shape:
        raise ValueError('Shape mismatch')
    n_triplets, n_features = a_to_b.shape
    if initial_weights == 'uniform':
        omegas = np.ones(n_features)
    else:
        omegas = initial_weights
        if len(initial_weights) != n_features:
            raise ValueError("initial weights must be length n_features")

    obj = lambda R_and_Ws: - _diagonal_objgrad_transformed(R_and_Ws, a_to_b, a_to_c, alpha, loss)[0]
    grad = lambda R_and_Ws: - _diagonal_objgrad_transformed(R_and_Ws, a_to_b, a_to_c, alpha, loss)[1]
    R_and_Ws = np.ones(1 + n_features)
    R_and_Ws[1:] = np.sqrt(omegas)

    R_and_Ws = fmin_bfgs(f=obj, x0=R_and_Ws, fprime=grad, disp=False)
    rho = R_and_Ws[0]**2; W = R_and_Ws[1:]
    W2 = np.square(R_and_Ws[1:]);
    omegas = W2 / np.sum(W2)

    return rho, omegas


################################################################################
#
#
################################################################################


def _dense_objgrad(rho, X, a_to_b, a_to_c, alpha, loss='square'):
    """Get the objective function and its gradeitn for a dense Mahalanobis
    metric.
    
    Parameters
    ----------
    rho : float
        The target margin
    X : np.ndarray, shape=[n_features, n_features]
        The mahalanobis matrix
    a_to_b : np.ndarray, dtype=float shape=[n_triplets, n_features]
        The difference from the "a" to "b" conformations along each feature.
        The "a" and "b" conformations, for each triplet, are the pair that
        are supposed to be close together.
    a_to_b : np.ndarray, dtype=float shape=[n_triplets, n_features]
        The difference from the "a" to "c" conformations along each feature.
        The "a" and "c" conformations, for each triplet, are the pair that
        are supposed to be far apart.
    alpha : float
        The weight given to maximizing the margin in the objective function.
        This trades off the desire to minimize the loss on triplets that are
        classified with a margin less than the target margin rho.
    loss : {'square', 'huber'}
        The type of loss function you like

    Returns
    -------
    objective : float
    grad_rho : float
        The gradient of objective respect to rho
    grad_X : elementwise gradient of objective with respect to each entry
        in X
    """
    if not np.isscalar(rho):
        rho = rho[0]

    # distance_AtoB should be a n_triplets length vector
    # where the distance_AtoB[i] is calculated as the vector
    # AtoB[i].T * X * AtoB[i]. This can be calculated as just the
    # diagonal entries of AtoB.T * X * AtoB, but doing it that way involves
    # calculating all the cross terms and throwing them away
    assert a_to_b.shape == a_to_c.shape
    n_triplets, n_features = a_to_b.shape
    assert X.shape == (n_features, n_features)

    distance_AtoB = np.sum(a_to_b * np.dot(a_to_b, X), axis=1)
    distance_AtoC = np.sum(a_to_c * np.dot(a_to_c, X), axis=1)
    margin = distance_AtoC - distance_AtoB
    

    if loss == 'huber':
        f_loss = huber_loss; g_loss = huber_loss_deriv
    elif loss == 'square':
        f_loss = square_loss; g_loss = square_loss_deriv
    else:
        raise NotImplementedError('sorry')

    loss_value = f_loss(margin - rho)
    loss_deriv = g_loss(margin - rho)

    objective = alpha * rho - np.mean(loss_value)
    grad_rho = alpha + np.mean(loss_deriv)

    grad_X = np.zeros((n_features, n_features), dtype=np.double)
    for i in xrange(n_triplets):
        vec1 = a_to_c[i, :]
        vec2 = a_to_b[i, :]
        # grad_X = loss_deriv[i] + alpha * np.outer(vec1, vec1)
        # general rank 1 update
        #grad_X -= (np.outer(a_to_c[i], a_to_c[i]) - np.outer(a_to_b[i], a_to_b[i])) * loss_deriv[i]
        grad_X = scipy.linalg.blas.cblas.dger(-loss_deriv[i], vec1, vec1, 1, 1, grad_X)
        grad_X = scipy.linalg.blas.cblas.dger(loss_deriv[i], vec2, vec2, 1, 1, grad_X)
    grad_X /= n_triplets

    return objective, grad_rho, grad_X


def _test_dense_objgrad():
    h = 1e-7
    a_to_b = np.random.randn(5,10)
    a_to_c = 1 + np.random.randn(5,10)
    alpha = 1
    rho = 1
    X = np.random.randn(10, 10)

    def test(loss):

        o1, gr1, gx1 = _dense_objgrad(rho, X, a_to_b, a_to_c, alpha, loss=loss)
        o2, gr2, gx2 = _dense_objgrad(rho+h, X, a_to_b, a_to_c, alpha, loss=loss)
        assert (o2 - o1) / h - gr1 < 1e-6

        for i in range(10):
            for j in range(10):
                X2 = np.copy(X)
                X2[i,j] += h
                o2, gr2, gx2 = _dense_objgrad(rho, X2, a_to_b, a_to_c, alpha, loss=loss)

                assert (o2 - o1) / h - gx1[i, j] < 1e-6

    test('square')
    test('huber')
    print 'passed'


def optimize_dense(a_to_b, a_to_c, alpha, initial_rho, initial_X, loss='square',
    epsilon=1e-5, max_outer_iterations=100, max_inner_iterations=10):
    """ Optimize a dense mahalanobis metric
    
    
    """
    assert a_to_b.shape == a_to_c.shape
    n_triplets, n_features = a_to_b.shape

    as_vector = lambda X: np.reshape(X, n_features * n_features)
    as_matrix = lambda V: np.reshape(V, (n_features, n_features))
    # wrappers used for getting rho
    def minus_f_of_rho(rho, X):
        return -_dense_objgrad(rho, X, a_to_b, a_to_c, alpha, loss=loss)[0]
    def minus_g_of_rho(rho, X):
        return [-_dense_objgrad(rho, X, a_to_b, a_to_c, alpha, loss=loss)[1]]
    # wrappers used for getting X
    def minus_f_X(vectorX, rho):
        value = -_dense_objgrad(rho, as_matrix(vectorX), a_to_b, a_to_c, alpha, loss=loss)[0]
        return np.array([value])
    def minus_g_X(vectorX, rho):
        value = -_dense_objgrad(rho, as_matrix(vectorX), a_to_b, a_to_c, alpha, loss=loss)[2]
        return as_vector(value)


    rho = initial_rho
    X = initial_X / np.trace(initial_X)
    objective = -np.inf

    #_dense_objgrad(rho, X, a_to_b, a_to_c, alpha, loss='square'):
    obj, gr, gx = _dense_objgrad(rho, X, a_to_b, a_to_c, alpha, loss=loss)

    print 'INITIAL OBJECTIVE', obj

    for k in range(max_outer_iterations):
        old_X = X.copy()
        old_rho = rho
        print 'Outer iteration %s' % k

        rho_optimizatrion_result = scipy.optimize.fmin_tnc(minus_f_of_rho, [rho],
            fprime=minus_g_of_rho, bounds=[(0, None)], args=(X,), disp=0)
        rho = rho_optimizatrion_result[0][0]
        print 'rho[k=%s] = %s' % (k, rho)

        # now find new X
        for i in range(max_inner_iterations):
            #print 'Inner iteration {}'.format(i)

            # The algorithm needs the gradient of the objective function,
            # but it wants the gradient of the objective function to be MAXIMIZED
            # and our objective function is the one to minimize, so we take the negative
            gradX = _dense_objgrad(rho, X, a_to_b, a_to_c, alpha, loss=loss)[2]
            eigenvalue, eigenvector = first_eigen(gradX)

            #print np.linalg.eigvalsh(gradX)
            if eigenvalue < epsilon:
                print 'Converged inner loop: eigenvalue= %s' % eigenvalue
                break

            # if you don't put a search multipler (or set it to 1), it tries
            # evaluating the objective function with X=0, and then you get NaNs
            search_multiplier = 0.50
            search_direction = search_multiplier * (np.outer(eigenvector, eigenvector) - X)

            line_search_result = scipy.optimize.line_search(f=minus_f_X,
                myfprime=minus_g_X, xk=as_vector(X), pk=as_vector(search_direction),
                args=(rho,), gfk=as_vector(minus_g_X(X, rho)))
            forward_increment = np.float(line_search_result[0])

            X = X + (forward_increment * search_direction)

        old_objective = objective
        objective = _dense_objgrad(rho, X, a_to_b, a_to_c, alpha, loss=loss)[0]
        convergence_criteria = np.linalg.norm(old_X - X)

        print 'X[k={}] eigenvalues: {}'.format(k, scipy.linalg.eigvalsh(X))
        print '||X[k={}]- X[k={}]|| = {}'.format(k, k-1, convergence_criteria)
        print 'objective function = {}'.format(objective)

        if convergence_criteria < epsilon:
            print('X changed less than %s, breaking' % epsilon)
            break
        # if objective < old_objective:
        #     print('Objective went in wrong direction. Breaking')
        #     return old_X, old_rho


    return rho, X


def optimize_rank1(a_to_b, a_to_c, alpha, loss='square'):
    assert a_to_b.shape == a_to_c.shape
    n_triplets, n_features = a_to_b.shape

    def objective_and_grad(rho, v):
        """Get the objective function, and its gradient using its natural
        variables rho and v
        """
        margin = np.dot(a_to_c, v)**2 - np.dot(a_to_b, v)**2
        
        print 'Classification accuracy %s/%s=%s' % (np.count_nonzero(margin > 0),
            len(margin), np.count_nonzero(margin > 0)/float(len(margin)))

        if loss == 'huber':
            loss_value = huber_loss(margin - rho)
            loss_deriv = huber_loss_deriv(margin - rho)
        elif loss == 'square':
            loss_value = square_loss(margin - rho)
            loss_deriv = square_loss_deriv(margin - rho)
        else:
            raise NotImplementedError('sorry')

        objective = alpha*rho - np.mean(loss_value)

        grad_rho = alpha + np.mean(loss_deriv)
        grad_v = 2.0/n_triplets * (-np.dot(np.multiply(loss_deriv, np.dot(a_to_c, v)), a_to_c)
            + np.dot(np.multiply(loss_deriv, np.dot(a_to_b, v)), a_to_b))

        return objective, grad_rho, grad_v


    def minus_transformed(s_and_w):
        """Get the negative of the objective function and its gradient in
        the transformed coordinates s and w, where rho=s**2 and 
        v = w / np.sqrt(np.sum(np.square(w))).
        
        s and w are packed into the same vector
        """
        s, w = s_and_w[0], s_and_w[1:]
        rho = s**2
        v = w / np.sqrt(np.sum(np.square(w)))

        objective, grad_rho, grad_v = objective_and_grad(rho, v)

        partial_R = 2*s*grad_rho        
        # dV_i / dw_i = \frac{1}{\sqrt{\sum_j{w_j^2}}} - ||w||^{-3}*w_i
        # dV_i / dw_j = ||w||^{-3}*w_j (where j != i)
        jacobian = - np.outer(w, w) * np.sum(np.square(w))**(-3.0/2)
        jacobian += np.diag(np.ones(len(w)) * np.sum(np.square(w)) ** (-1.0 / 2.0))

        grad_w = np.dot(jacobian, grad_v)

        return -objective, -np.hstack(([partial_R], grad_w))

    x0 = np.ones(1 + n_features)
    s_and_w = fmin_bfgs(f=lambda x: minus_transformed(x)[0], x0=x0,
        fprime=lambda x: minus_transformed(x)[1], disp=False)
    s, w = s_and_w[0], s_and_w[1:]
    rho = s**2; v = w / np.sqrt(np.sum(np.square(w)))

    return rho, np.outer(v, v)


if __name__ == '__main__':
    #_test_diagonal_objgrad()
    #_test_diagonal_objgrad_transformed()
    #_test_dense_objgrad()
    a_to_b = np.random.randn(5,10)
    a_to_c = np.random.randn(5,10)
    alpha = 1
    print optimize_rank1(a_to_b, a_to_c, alpha, loss='square')


################################################################################
#
# OLD CODE
#
################################################################################

class DiagonalMetricCalculator(object):
    @classmethod
    def optimize_metric(cls, a_to_b, a_to_c, alpha, verbose=True, loss_function='square'):
        m = cls(a_to_b, a_to_c, alpha, verbose, loss_function)
        return m.X

    def test1(self):
        def test_deriv(i, dx=1e-6):
            o1, g1 = self.minus_objective_and_grad(rho_and_omegas)
            new = np.copy(rho_and_omegas)
            new[i] += dx
            o2 =  self.minus_objective_and_grad(new)[0]

            print 'numeric' , (o2-o1)/dx
            print 'analytic', g1[i]
            assert np.abs((o2 - o1)/dx - g1[i])/g1[i] < 1e-5

        rho_and_omegas = np.empty(self.dim + 1)
        rho_and_omegas[0] = np.random.randn()
        rho_and_omegas[1:] = np.random.randn(self.dim)

        for i in range(len(rho_and_omegas)):
            test_deriv(i)

        print 'Passed Test 1\n\n'

    def test3(self):
        def test_deriv(i, dx=1e-6):
            o1, g1 = self.minus_huber_objective_and_grad(rho_and_omegas)
            new = np.copy(rho_and_omegas)
            new[i] += dx
            o2 =  self.minus_huber_objective_and_grad(new)[0]

            print 'numeric' , (o2-o1)/dx
            print 'analytic', g1[i]
            assert np.abs(((o2 - o1)/dx - g1[i])/(max(g1[i], (o2-o1)/dx))) < 1e-5

        rho_and_omegas = np.empty(self.dim + 1)
        rho_and_omegas[0] = np.random.randn()
        rho_and_omegas[1:] = np.random.randn(self.dim)

        for i in range(len(rho_and_omegas)):
            test_deriv(i)

        print 'Passed Test 3'

    def test2(self):
        def test_deriv(i, dx=1e-5):
            o1, g1 = self.transformed_minus_objective_and_grad(R_and_Ws)
            new = np.copy(R_and_Ws)
            new[i] += dx
            o2 = self.transformed_minus_objective_and_grad(new)[0]

            numeric = (o2 - o1) / dx
            analytic = g1[i]
            print 'analytic', analytic
            print 'numeric ', numeric
            assert np.abs((numeric - analytic)/max(numeric, analytic)) < 1e-4

        R_and_Ws = np.random.randn(1 + self.dim/2)

        for i in range(len(R_and_Ws)):
            test_deriv(i)

        print 'Passed Test 2'


    def __init__(self, a_to_b, a_to_c, alpha, verbose, loss_function):
        """Initialize the DiagonalMetricCalculator

        Parameters
        ----------
        a_to_b : np.ndarray, dtype=float shape=[n_triplets, n_features]
            The difference from the "a" to "b" conformations along each feature.
            The "a" and "b" conformations, for each triplet, are the pair that
            are supposed to be close together.
        a_to_b : np.ndarray, dtype=float shape=[n_triplets, n_features]
            The difference from the "a" to "c" conformations along each feature.
            The "a" and "c" conformations, for each triplet, are the pair that
            are supposed to be far apart.
        alpha : float
            The weight given to maximizing the margin in the objective function.
            This trades off the desire to minimize the loss on triplets that are
            classified with a margin less than the target margin rho.
        verbose : boolean
        """

        if not loss_function in ['huber', 'square']:
            raise ValueError('loss_function must be "huber" or "square". you supplied %s' % loss_function)

        self.num_triplets, self.dim = a_to_b.shape
        self.num_triplets = float(self.num_triplets)
        assert a_to_b.shape == a_to_c.shape


        self.sq_triplets_b2c = np.square(np.matrix(a_to_c.T)) - np.square(np.matrix(a_to_b.T))

        # we're going to use K as the 'alpha' parameter
        self.K = alpha
        self.printer = sys.stdout if verbose else open('/dev/null', 'w')

        if loss_function == 'huber' and alpha >= 1:
            raise ValueError(('With the huber loss function, a value of alpha greater '
                'than or equal to one will get you in serious trouble.'))

        # We do the optimization in a transformed set of coordinates, R and W
        # where \rho = R^2 and W_i = \omega_i / \sum_j{\omega_j}
        R_and_Ws = np.ones(self.dim+1)
        R_and_Ws[0] = 1

        obj = lambda X: self.transformed_minus_objective_and_grad(X, loss_function)[0]
        grad = lambda X: self.transformed_minus_objective_and_grad(X, loss_function)[1]

        R_and_Ws = fmin_bfgs(f=obj, x0=R_and_Ws, fprime=grad, disp=False)

        W2 = np.square(R_and_Ws[1:])

        self.X = np.matrix(np.diag(W2/ np.sum(W2)))
        self.rho = R_and_Ws[0]**2


    def minus_square_objective_and_grad(self, rho_and_omegas):
        """Negative of the objective function, and its gradient using the squared
        hinge loss function

        Parameters
        ----------
        rho_and_omegas : np.ndarray, shape=[n_features + 1]
            The first entry is rho, the target margin. The rest of the entries
            are the weights, omega_i which run over all of the features and are
            the diagonal entries of the Mahalanobis matrix.

        Returns
        -------
        -objective : float
            The negative objective function
        -grad : np.ndarry, shape=[n_features + 1]
            The first entry is the negative of the derivative of the objective function
            w.r.t \rho, and the rest of the entries are the derivative with respect
            to each element of \omega.

        Notes
        -----
        This is not in transformed coordinates
        """

        # extract the variables
        rho = rho_and_omegas[0]
        omegas = rho_and_omegas[1:len(rho_and_omegas)]

        # sanity check
        assert len(omegas) == self.dim

        # compute d(a,c) - d(a,b), the margin
        margin = (np.diag(omegas)*self.sq_triplets_b2c).sum(axis=0)

        # look for where the margin is less than the target margin rho -- this
        # is what we need to penalize
        mmargin = np.ma.masked_greater(margin - rho, 0)
        mmargin = mmargin.filled(0)

        print 'Classification accuracy: {0}/{1}={2:5f}'.format(np.count_nonzero(margin > 0), self.num_triplets, np.count_nonzero(margin > 0)/float(self.num_triplets))

        # compute the squared hinge loss, and then normalize by the number of triplets
        loss = np.square(mmargin).sum() / self.num_triplets

        # the objective function trades off the target margin against the loss
        objective = self.K * rho - loss

        print 'objective', objective

        # compute the gradient
        grad = np.empty(self.dim + 1)
        grad[0] = self.K  +  2 * mmargin.sum() / self.num_triplets
        grad_omegas = -2*np.matrix(mmargin) * self.sq_triplets_b2c.T / self.num_triplets
        grad[1:self.dim+1] = grad_omegas

        return -objective, -grad


    def minus_huber_objective_and_grad(self, rho_and_omegas, h=0.5):
        """Negative of the objective function, and its gradient using the huber
        loss function

        Parameters
        ----------
        rho_and_omegas : np.ndarray, shape=[n_features + 1]
            The first entry is rho, the target margin. The rest of the entries
            are the weights, omega_i which run over all of the features and are
            the diagonal entries of the Mahalanobis matrix.

        Returns
        -------
        -objective : float
            The negative objective function
        -grad : np.ndarry, shape=[n_features + 1]
            The first entry is the negative of the derivative of the objective function
            w.r.t \rho, and the rest of the entries are the derivative with respect
            to each element of \omega.

        Notes
        -----
        This is not in transformed coordinates
        """

        # extract the actual variables
        rho = rho_and_omegas[0]
        omegas = rho_and_omegas[1:len(rho_and_omegas)]
        # sanity check
        assert len(omegas) == self.dim

        # d(a,c) - d(a,b) for all triplets
        margin = (np.diag(omegas)*self.sq_triplets_b2c).sum(axis=0)
        margin = np.ravel(np.array(margin))

        margin_minus_rho = margin - rho
        greater_h = np.where(margin_minus_rho >= h)[0]
        between = np.where(np.logical_and(-h <  np.array(margin_minus_rho), margin_minus_rho < h))[0]
        less_minus_h = np.where(margin_minus_rho <= -h)[0]

        t1 = np.sum(np.square((h - margin_minus_rho[between]))) / (4*h*self.num_triplets)
        t2 = -np.sum(margin_minus_rho[less_minus_h]) / self.num_triplets
        loss = t1 + t2
        print 'Classification accuracy: {0}/{1} = {2:5f}'.format(np.count_nonzero(margin > 0), self.num_triplets, np.count_nonzero(margin > 0)/self.num_triplets)

        objective = self.K * rho - loss

        print 'objective', objective

        grad = np.empty(self.dim + 1)
        grad_rho_t1 = -(2 * h * self.num_triplets)**(-1) * np.sum(h-margin_minus_rho[between])
        grad_rho_t2 = -len(less_minus_h) / self.num_triplets
        grad[0] = self.K + grad_rho_t1 + grad_rho_t2

        grad_omega_t1 = (2 * h * self.num_triplets)**(-1) * np.matrix(h - margin_minus_rho[between,:]) * self.sq_triplets_b2c.T[between, :]
        grad_omega_t2 = self.sq_triplets_b2c.T[less_minus_h,:].sum(axis=0) / self.num_triplets
        grad[1:] = grad_omega_t1 + grad_omega_t2

        return -objective, -grad

    @memoized
    def transformed_minus_objective_and_grad(self, R_and_Ws, loss_function):
        rho = R_and_Ws[0]**2
        W = R_and_Ws[1:]
        W2 = np.square(W)
        omegas = W2 / np.sum(W2)

        if loss_function == 'huber':
            objective, grad_rho_omegas = self.minus_huber_objective_and_grad(np.hstack(([rho], omegas)))
        else:
            objective, grad_rho_omegas = self.minus_square_objective_and_grad(np.hstack(([rho], omegas)))


        partial_R = 2*R_and_Ws[0]*grad_rho_omegas[0]
        grad_omegas = grad_rho_omegas[1:]

        jacobian = -2 * np.matrix(W).T * np.matrix(W2) / np.square(np.sum(W2))
        jacobian += np.diag(2*W / np.sum(W2))
        grad_W = np.array(jacobian * np.matrix(grad_omegas).T)[:,0]

        #print 'Computuing: Omegas', omegas
        #print 'gradient', np.hstack(([partial_R], grad_W))
        return objective, np.hstack(([partial_R], grad_W))



class MetricCalculator(object):
    def test_huber(self):
        X0 = np.matrix(np.random.randn(self.dim, self.dim))
        X1 = np.copy(X0)
        dx = 0.001
        X1[0,0] += dx
        rho = np.random.randn()
        self.R = np.random.randn()

        o0, gr0, gx0 = self.minus_huber_objective_and_grad(rho, X0)
        o1, gr1, gx1 = self.minus_huber_objective_and_grad(rho, X1)

        print (o1 - o0) / dx
        print gx0

        sys.exit(1)


    @classmethod
    def optimize_metric(cls, a_to_b, a_to_c, num_outer=10, num_inner=10, alpha=1, beta=0, epsilon=1e-5, verbose=True, loss_function='huber'):
        """"Optimize a Large-Margin Mahalanobis Distance Metric with triplet training examples.

        triplets: 3 element tuple (a,b,c). Each of a,b,c should be 2D arrays of
            length equal to the number of training examples, and width the number
            of dimensions. Each training example is the statement that a[i,:] is closer
            to b[i,:] than it is to c[i,:]

        num_outer: number of outer iterations of the algorithm (int)
        num_inner: number of inner iterations of the algorithm (int)
        alpha: Algorithmic parameter which trades off between loss on the training
           examples and the margin. When alpha goes to infinity, the objective function
           contains only the margin. When alpha goes to zero, the loss function is
           only penalty terms.
        beta: Regularization strength on the frobenius norm of the metric matrix.
        epsilon: Convergence cutoff. Interpreted as the percent change."""

        m = cls(a_to_b, a_to_c, num_outer, num_inner, alpha, beta, epsilon, verbose, loss_function)
        return m.X

    def __init__(self, a_to_b, a_to_c, num_outer, num_inner, alpha, beta, epsilon, verbose, loss_function):
        self.num_triplets, self.dim = a_to_b.shape
        assert a_to_b.shape == a_to_c.shape
        self.tripletsa2b = np.matrix(a_to_b.T)
        self.tripletsa2c = np.matrix(a_to_c.T)

        self.loss_function = loss_function
        assert self.loss_function in ['square', 'huber']

        # gxm is a num_triplets length array. Each element is a matrix giving
        # the elementwise partial derivative with respect to X of the margin
        # on that training example.
        # the whole thing is num_triplets * (dimensionality)^2
        self.gxm = np.empty(self.num_triplets, dtype='object')
        for i in xrange(self.num_triplets):
            self.gxm[i] = self.tripletsa2c[:, i] * self.tripletsa2c[:, i].T - \
                        self.tripletsa2b[:, i] * self.tripletsa2b[:, i].T

        self.num_outer = num_outer
        self.num_inner = num_inner
        self.epsilon = epsilon
        self.K = alpha
        self.R = beta
        self.printer = sys.stdout if verbose else open('/dev/null', 'w')

        # use DiagonalMetricCalculator as initial value
        dmc = DiagonalMetricCalculator(a_to_b, a_to_c, alpha, verbose=True,
                            loss_function=loss_function)
        initial_X = dmc.X
        #initial_X = np.eye(self.dim) / float(self.dim)

        self.X = self.outer_loop(initial_X)

    def outer_loop(self, initial_X):
        """Main loop that constructs the X matrix, and iterates maximizing rho
        and the maximizing X"""


        X = initial_X
        prev_obj = -np.inf
        for i in range(self.num_outer):
            rho = self.find_rho(X)

            obj = -self.minus_square_objective_and_grad(rho, X)[0]
            print >> self.printer, 'Iteration: {0}'.format(i)
            print >> self.printer, 'rho:       {0:5f}'.format(rho)
            print >> self.printer, 'X\n', X
            print >> self.printer

            X, finished = self.find_X(rho, X)
            if finished or np.abs((obj - prev_obj) / obj) < self.epsilon:
                break

            prev_obj = obj


        print >> self.printer, '\nFinal Metric:'
        print >> self.printer, 'eigvals', np.real(np.linalg.eigvals(X))
        print >> self.printer, 'Contributions to the objective function'
        m, l, r = self._objective_function_contributions
        print >> self.printer, '(alpha)*rho:    {0:5f}'.format(m)
        print >> self.printer, 'Hinge Loss:     {0:5f}'.format(l)
        print >> self.printer, 'Regularization: {0:5f}'.format(r)
        return X

    def find_X(self, rho, X1):
        """Find X that maximizes the objective function at fixed rho"""

        objective = lambda X: self.minus_square_objective_and_grad(rho, as_matrix(X))[0]
        gradient = lambda X: as_vector(self.minus_square_objective_and_grad(rho, as_matrix(X))[2])
        as_vector = lambda X: np.reshape(np.array(X), self.dim * self.dim)
        as_matrix = lambda V: np.matrix(np.reshape(V, (self.dim, self.dim)))

        #print >> self.printer, 'alg2 starting', objective(X1)
        for i in range(self.num_inner):
            current_grad = as_matrix(gradient(X1))
            try:
                u, v = sparse_eigsh(-current_grad, k=1, which='LA')
            except Exception as e:
                print >> self.printer, 'Warning: Sparse solver failed'
                u, v = np.linalg.eigh(-current_grad)
            u = np.real(u)[0]

            v = np.matrix(np.real(v))
            p = (v * v.T - X1)

            if u < 0:
                #print >> self.printer, 'u < 0', u
                break

            try:
                u2, v2 = sparse_eigsh(X1, k=1, which='LM', sigma=1)
                #u2 = np.linalg.eigvals(X1)
            except NotImplementedError:
                warnings.warn("Warning: Your sparse eigensolver does not support shift-invert mode")
                u2, v2 = sparse_eigsh(X1, k=1, which='LM')
            except Exception as e:
                print >> self.printer, 'Warning: Sparse solver failed'
                u2 = np.linalg.eigvals(X1)


            u2 = np.real(u2)[0]
            if u2 > 1 + self.epsilon:
                print >> self.printer, 'u2 > 1', u2
                break

            stp, f_count, g_count, f_val, old_fval, gval = line_search_wolfe1(f=objective, fprime=gradient, xk=as_vector(X1), pk=as_vector(p))
            #print 'stp', stp
            if stp == None:
                #print >> self.printer, 'breaking for stp=None'
                break

            if np.abs((f_val - old_fval) / old_fval) < self.epsilon:
                #print >> self.printer, 'breaking for insufficient gain'
                break

            X1 = X1 + stp * p

        #print >> self.printer, 'j: {0}'.format(i)
        return X1, i == 0

    def find_rho(self, X):
        """Find rho that maximizes the objective function at fixed X"""
        # argmax with respect to p of f(X, p)
        # f(X,p) = p - C * sum_r loss <A,X> - p
        obj_and_grad = lambda rho: (self.minus_objective_and_grad(rho, X)[0],
            (self.minus_objective_and_grad(rho, X)[1],))
        result = fmin_tnc(obj_and_grad, (0,), disp=0, bounds=[(0,None)])
        return result[0][0]

    def minus_objective_and_grad(self, rho, X):
        if self.loss_function == 'square':
            return self.minus_square_objective_and_grad(rho, X)
        else:
            return self.minus_huber_objective_and_grad(rho, X)

    @memoized
    def minus_square_objective_and_grad(self, rho, X):
        """Computes the objective function, the partial derivative of the objective function with
        respect to rho, and the matrix of partial derivatives with respect to each of the
        elements of X"""
        # sometimes rho is a 1element vector (when called from fmin_tnc)
        if not np.isscalar(rho):
            rho = rho[0]

        m1 = np.array(np.multiply(self.tripletsa2b, X*self.tripletsa2b)).sum(axis=0)
        m2 = np.array(np.multiply(self.tripletsa2c, X*self.tripletsa2c)).sum(axis=0)
        m = np.ma.masked_greater((m2 - m1) - rho, 0)
        #print 'minus_objective_and_grad  rho=', rho
        print 'Classification accuracy: {0}/{1} = {2:5f}'.format(np.count_nonzero(m2 - m1 > 0), self.num_triplets, np.count_nonzero(m2 - m1 > 0)/float(self.num_triplets))

        if np.count_nonzero(m.mask) == self.num_triplets:
            avg_loss = 0
            grad_avg_loss = 0
            avg_grad_triplets = 0
        else:
            avg_loss = np.square(m).sum() / self.num_triplets
            grad_avg_loss = -2 * np.sum(m) / self.num_triplets
            avg_grad_triplets = (2 * m * np.ma.array(self.gxm, mask=m.mask)).sum()
            avg_grad_triplets /= self.num_triplets

        regularization = 0.5 * self.R * np.sum(np.square(X)) if self.R != 0 else 0
        regularization_grad = self.R * X

        # record the contributions to the objective function
        self._objective_function_contributions = (self.K * rho, -avg_loss, -regularization)
        #print >> self.printer, self._objective_function_contributions

        objective = self.K * rho - avg_loss - regularization
        grad_rho = self.K - grad_avg_loss

        return -objective, -grad_rho, avg_grad_triplets - regularization_grad

    @memoized
    def minus_huber_objective_and_grad(self, rho, X, h=0.5):

        m1 = np.array(np.multiply(self.tripletsa2b, X*self.tripletsa2b)).sum(axis=0)
        m2 = np.array(np.multiply(self.tripletsa2c, X*self.tripletsa2c)).sum(axis=0)
        margin = m2 - m1
        print 'Classification accuracy: {0}/{1} = {2:5f}'.format(np.count_nonzero(margin > 0), self.num_triplets, np.count_nonzero(margin > 0)/float(self.num_triplets))

        margin_minus_rho = margin - rho
        greater_h = np.where(margin_minus_rho >= h)[0]
        between = np.where(np.logical_and(-h <  np.array(margin_minus_rho), margin_minus_rho < h))[0]
        less_minus_h = np.where(margin_minus_rho <= -h)[0]

        t1 = np.sum(np.square((h - margin_minus_rho[between]))) / (4.0*h*self.num_triplets)
        t2 = -np.sum(margin_minus_rho[less_minus_h]) / self.num_triplets
        loss = t1 + t2

        regularization = 0.5 * self.R * np.sum(np.square(X)) if self.R != 0 else 0
        regularization_grad = self.R * X

        grad_rho_t1 = -(2.0 * h * self.num_triplets)**(-1) * np.sum(h-margin_minus_rho[between])
        grad_rho_t2 = -len(less_minus_h) / float(self.num_triplets)
        grad_rho = self.K + grad_rho_t1 + grad_rho_t2

        grad_X_t1 = -(2.0 * h * self.num_triplets)**(-1) * sum([(h - margin_minus_rho[i]) * self.gxm[i] for i in between])

        if len(less_minus_h) > 0:
            grad_X_t2 = -np.sum(self.gxm[less_minus_h, :]) / float(self.num_triplets)
        else:
            grad_X_t2 = np.matrix(np.zeros((self.dim, self.dim)))

        objective = self.K * rho - loss + regularization
        return -objective, -grad_rho, grad_X_t1 + grad_X_t2 - regularization_grad

optimize_metric = MetricCalculator.optimize_metric
optimize_dmetric = DiagonalMetricCalculator.optimize_metric
