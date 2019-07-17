"""
@author: erathorn
@date: July 2019
@version: 1.0
"""

import numpy as np
import numpy.linalg as np_la
import scipy.stats as spst


def initial_simplex(vector, size, scale):
    # type: (np.ndarray, int, float) -> np.ndarray
    """
    This function creates the initial simplex for simplex slice sampling.
    This sampling scheme is described in Cowles, Yan Smith 2009: DOI: 10.1198/jcgs.2009.08012

    :param vector: vector to sample
    :type vector: np.ndarray
    :param size: size of vector
    :type size: int
    :param scale: scale of the simplex
    :type scale: float
    :return: vertices of the simplex
    :rtype: np.ndarray
    """

    vertices = np.eye(size, dtype=np.double, order="C")
    vertices[:, 1:] += (1.0 - scale) * (vertices[:, 0][:, np.newaxis] - vertices[:, 1:])
    return np.transpose(vertices + vector - np.dot(vertices, spst.dirichlet.rvs(np.ones_like(vector))[0])[:, None])


def slice_sample_simplex(vertices, vector):
    # type: (np.ndarray, np.ndarray) -> tuple
    """
    This sampling scheme is described in Cowles, Yan Smith 2009: DOI: 10.1198/jcgs.2009.08012

    :param vertices: vertices of the simplex to sample from
    :type vertices: np.ndarray
    :param vector: original starting values
    :type vector: np.ndarray
    :return: first part of sampling, edges of the simplex
    :rtype: tuple
    """
    vb = np_la.solve(vertices, vector)
    xb = spst.dirichlet.rvs(np.ones_like(vector))[0]
    return xb, vb


def shrink_simplex(vertices, vector, bc, cc, vb):
    # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    """
    This function rescales and shrinks the initial simplex, if the initial sample was unsuccessful.
    This sampling scheme is described in Cowles, Yan Smith 2009: DOI: 10.1198/jcgs.2009.08012

    :param vertices: vertices of the initial simplex
    :type vertices: np.ndarray
    :param vector: original starting values
    :type vector: np.ndarray
    :param bc: boundaries of the simplex
    :type bc: np.ndarray
    :param cc: proposed sample
    :type cc: np.ndarray
    :param vb: simplex
    :type vb: np.ndarray
    :return: new vertices
    :rtype: np.ndarray
    """
    size = len(vector)
    for i in np.where(bc < vb)[0]:
        inds = np.arange(size) != i
        vertices[:, inds] += bc[i] * (vertices[:, i].reshape(-1, 1) - vertices[:, inds])
        bc = np_la.solve(vertices, cc)
    return vertices


def slice_sample(vec, width, lower_bound=0.0):
    # type: (np.ndarray, float, float) -> tuple
    """
    Slice sample multivariate vector

    See: Neal 2001 doi:10.1214/aos/1056562461

    :param vec: initial vector
    :type vec: np.ndarray
    :param width: sampling width
    :type width: fkoat
    :param lower_bound: lower bound for sampling
    :type lower_bound: float
    :return: sampled vector, lower and upper bound vector
    :rtype: tuple
    """
    size = len(vec)
    lower = vec - (np.random.uniform(0, 1, size) * width)
    upper = lower + width

    lower[np.where(lower < lower_bound)] = 0.0
    x = width * np.random.uniform(0, 1, size) + lower

    return x, lower, upper


def slice_shrink(vec, lower, upper, original):
    # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    """
    Shrink the slice window.

    See: Neal 2001 doi:10.1214/aos/1056562461

    :param vec: sampled vector to be shrunk
    :type vec: np.ndarray
    :param lower: lower bound vector
    :type lower: np.ndarray
    :param upper: upper bound vector
    :type upper: np.ndarray
    :param original: original vector
    :type original: np.ndarray
    :return: new sampled vector
    :rtype: np.ndarray
    """
    for ind in range(len(vec)):
        if vec[ind] < original[ind]:
            lower[ind] = vec[ind]
        else:
            upper[ind] = vec[ind]
        vec[ind] = np.random.uniform(lower[ind], upper[ind])
    return vec
