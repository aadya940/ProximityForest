import numpy as np
import random
from typing import List
from numba import njit
import pandas as pd

from aeon.classification.base import BaseClassifier
from aeon.utils.validation import check_n_jobs
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

DISTANCE_MEASURES = [
    euclidean_distance,
    ddtw_distance,
    dtw_distance,
    wddtw_distance,
    wdtw_distance,
    lcss_distance,
    erp_distance,
    twe_distance,
    msm_distance,
]


# Select's Appropriate Parameters for Different Distances as described in the Paper
def func_with_params(func: callable, X: np.ndarray, exemplar: np.ndarray):
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
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities**2)
    return gini


def _weighted_gini_impurity(y_branches: List[np.ndarray]):
    total_samples = sum(len(y_branch) for y_branch in y_branches)
    weights = [len(y_branch) / total_samples for y_branch in y_branches]
    gini_impurities = [_gini_impurity(y_branch) for y_branch in y_branches]
    weighted_gini = sum(weight * gini for weight, gini in zip(weights, gini_impurities))
    return weighted_gini


def gini_difference(y_parent, y_branches: List[np.ndarray]):
    return _gini_impurity(y_parent) - _weighted_gini_impurity(y_branches)


class ProximityTreeNode:
    def __init__(
        self,
        depth,
    ):
        self.depth = depth
        self.incoming_classes = None
        self.leaf_node = False
        self.next_nodes = None
        self.is_fit = False

    def _fit(
        self,
        X,
        y,
        X_incoming,  # Incoming X to the Tree Node
        y_incoming,  # Incoming y to the Tree Node
        incoming_classes: List,  # Use class_dict instead present in Aeon Infrastrucutre
        num_candidates_for_selection=5,
    ):
        self.incoming_classes = incoming_classes

        if len(incoming_classes) == 1:
            self.leaf_node = True
            self.num_next_nodes = 0

        if (
            not self.leaf_node
            and not self.is_fit
            and len(X_incoming) != 0
            and len(y_incoming) != 0
        ):
            _distance_measures, _exemplars = self._generate_candidate_splitters(
                X, y, num_candidates=num_candidates_for_selection
            )

            self.distance_measures, self.exemplars = self._generate_best_splitters(
                X_incoming, y_incoming, _exemplars, _distance_measures
            )
            self.num_next_nodes = len(np.unique(y_incoming))

            self._generate_next_nodes()

        self.is_fit = True

    def _generate_candidate_splitters(self, X, y, num_candidates=5):
        # Complete X of the Training Data
        # Complete Y of the Training Data
        _measures = []
        _exemplars = [[] for _ in range(num_candidates)]
        for j in range(num_candidates):
            _measures.append(random.choice(DISTANCE_MEASURES))
            for i in self.incoming_classes:
                _val = random.choice(X[y == i])
                _exemplars[j].append(_val)
        _exemplars = np.array(_exemplars, dtype=object)

        # _measures: List[distance measures]
        # Exemplars: ndarray[num_candidates, num_incoming_classs]
        return _measures, _exemplars

    def _generate_best_splitters(
        self,
        X_incoming_train,  # Array of X values (ndarray)
        y_incoming_train,  # Array of y values (ndarray)
        exemplars,  # List of exemplars, each exemplar contains a series for each class
        distance_measures: List,  # List of distance measures
    ):
        scores = np.zeros(len(exemplars))
        y_branch = [[] for _ in range(len(np.unique(y_incoming_train)))]
        for i, exemplar in enumerate(exemplars):
            # For a Particular distance measure and `c` exemplars
            dist = distance_measures[i]
            _distance = np.zeros((len(X_incoming_train), len(exemplars[0])))
            for idxs, series in enumerate(exemplar):
                for idxx, x_values in enumerate(X_incoming_train):
                    _distance[idxx, idxs] = func_with_params(
                        dist, np.array(x_values), np.array(series)
                    )

            indices = np.argmin(_distance, axis=1)
            for yvals in y_incoming_train:
                for _idx in indices:
                    y_branch[_idx].append(yvals)

            scores[i] = gini_difference(y_incoming_train, y_branch)

        idx = np.argmax(scores)
        return distance_measures[idx], exemplars[idx]

    def _generate_branched_data(
        self,
        X_incoming,
        y_incoming: np.ndarray = None,  # Only useful while fitting the Model
    ):
        if not self.is_fit:
            raise ValueError("Fit the Node to get branched data.")

        dist = self.distance_measures
        X_branch = [[] for _ in range(self.num_next_nodes)]
        _distance = np.zeros((X_incoming.shape[0], len(self.exemplars)))
        for idxs, series in enumerate(self.exemplars):
            for idxx, x_values in enumerate(X_incoming):
                _distance[idxx, idxs] = func_with_params(dist, x_values, series)

        indices = np.argmin(_distance, axis=1)
        for xvals in X_incoming:
            for _idx in indices:
                X_branch[_idx].append(xvals)

        if y_incoming is not None:
            y_branch = [[] for _ in range(self.num_next_nodes)]
            for yvals in y_incoming:
                for _idx in indices:
                    y_branch[_idx].append(yvals)
            return np.array(X_branch, dtype=object), np.array(y_branch, dtype=object)

        return np.array(X_branch, dtype=object)

    def _generate_next_nodes(self):
        self.next_nodes = []
        for _ in range(self.num_next_nodes):
            self.next_nodes.append(ProximityTreeNode(depth=self.depth + 1))


class ProximityTreeClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        self.root_node = None

    def _fit(
        self, X, y, num_candidates_for_selection=5, max_depth=5, min_samples_split=1
    ):
        self.root_node = ProximityTreeNode(depth=0)
        self._recursive_fit(
            self.root_node,
            X,
            y,
            X,
            y,
            num_candidates_for_selection=num_candidates_for_selection,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

    def _recursive_fit(
        self,
        node,
        X,
        y,
        X_branch,
        y_branch,
        num_candidates_for_selection,
        max_depth,
        min_samples_split,
    ):
        if (
            len(np.unique(y)) == 1
            or node.depth == max_depth
            or len(y) < min_samples_split
        ):
            node.leaf_node = True
            node.num_next_nodes = 0
            return

        node._fit(
            X,
            y,
            X_branch,
            y_branch,
            np.unique(y).tolist(),
            num_candidates_for_selection,
        )

        X_branches, y_branches = node._generate_branched_data(X, y)

        if node.next_nodes is not None:
            for i, next_node in enumerate(node.next_nodes):
                next_node._fit(
                    X,
                    y,
                    X_branches[i],
                    y_branches[i],
                    np.unique(y_branches[i]).tolist(),
                    num_candidates_for_selection,
                )

                if len(X_branches[i]) != 0 and len(y_branches[i]) != 0:
                    self._recursive_fit(
                        next_node,
                        X,
                        y,
                        X_branches[i],
                        y_branches[i],
                        num_candidates_for_selection,
                        max_depth,
                        min_samples_split,
                    )

    def _predict(self, X) -> np.ndarray:
        # TODO
        pass
