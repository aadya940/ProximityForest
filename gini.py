import numpy as np
import random
from typing import List

from aeon.distances import (
    euclidean_distance,
    ddtw_distance,
    dtw_distance,
    wddtw_distance,
    wdtw_distance,
    lcss_distance,
    erp_distance,
    twe_distance,
    msm_distance,
)

def func_with_params(func: callable, X: np.ndarray, exemplar: np.ndarray):
    """
    Select appropriate parameters for different distances.

    Parameters:
    -----------
    func : callable
        Distance function.
    X : np.ndarray
        Data.
    exemplar : np.ndarray
        Exemplar data.

    Returns:
    --------
    float
        Calculated distance.
    """
    _name = func.__name__
    if (
        _name == "euclidean_distance"
        or _name == "dtw_distance"
        or _name == "ddtw_distance"
    ):
        return func(X, exemplar)
    elif _name == "wddtw_distance" or _name == "wdtw_distance":
        return func(X, exemplar, g=random.random())
    elif _name == "erp_distance":
        return func(X, exemplar)
    elif _name == "lcss_distance":
        return func(X, exemplar, random.uniform(0, (X.shape[0] + 1) // 4))
    elif _name == "twe_distance":
        return func(
            X,
            exemplar,
            nu=np.random.choice([10 ** (-5 * i) for i in range(6)]),
            lmbda=np.random.choice(np.arange(10) / 9),
        )
    elif _name == "msm_distance":
        return msm_distance(
            X, exemplar, c=np.random.uniform(low=10**-2, high=10**2, size=1)[0]
        )
    else:
        raise ValueError("Not a Valid Distance Measure")

def _gini_impurity(y):
    """
    Calculate Gini impurity.

    Parameters:
    -----------
    y : np.ndarray
        Target values.

    Returns:
    --------
    float
        Gini impurity.
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities**2)
    return gini

def _weighted_gini_impurity(y_branches: List[np.ndarray]):
    """
    Calculate weighted Gini impurity.

    Parameters:
    -----------
    y_branches : List[np.ndarray]
        List of target values for branches.

    Returns:
    --------
    float
        Weighted Gini impurity.
    """
    total_samples = sum(len(y_branch) for y_branch in y_branches)
    weights = [len(y_branch) / total_samples for y_branch in y_branches]
    gini_impurities = [_gini_impurity(y_branch) for y_branch in y_branches]
    weighted_gini = sum(weight * gini for weight, gini in zip(weights, gini_impurities))
    return weighted_gini

def gini_difference(y_parent, y_branches: List[np.ndarray]):
    """
    Calculate Gini difference.

    Parameters:
    -----------
    y_parent : np.ndarray
        Parent target values.
    y_branches : List[np.ndarray]
        List of target values for branches.

    Returns:
    --------
    float
        Gini difference.
    """
    return _gini_impurity(y_parent) - _weighted_gini_impurity(y_branches)
