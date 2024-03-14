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
from aeon.exceptions import NotFittedError

from gini import *

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

class ProximityTreeNode:
    def __init__(
        self,
        depth,
    ):
        """
        Initialize ProximityTreeNode.

        Parameters:
        -----------
        depth : int
            Depth of the node.
        """

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
        """
        Fit the node.

        Parameters:
        -----------
        X : np.ndarray
            Data.
        y : np.ndarray
            Target values.
        X_incoming : np.ndarray
            Incoming data.
        y_incoming : np.ndarray
            Incoming target values.
        incoming_classes : List
            List of incoming classes.
        num_candidates_for_selection : int, optional
            Number of candidates for selection. Defaults to 5.
        """
        self.incoming_classes = incoming_classes

        if len(incoming_classes) == 1:
            self.leaf_node = True
            self.num_next_nodes = 0

        if (
            not self.leaf_node
            and (not self.is_fit)
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
        """
        Generate candidate splitters.

        Parameters:
        -----------
        X : np.ndarray
            Data.
        y : np.ndarray
            Target values.
        num_candidates : int, optional
            Number of candidates. Defaults to 5.

        Returns:
        --------
        List[callable], np.ndarray
            Distance measures and exemplars.
        """
        _measures = []
        _exemplars = [[] for _ in range(num_candidates)]
        for j in range(num_candidates):
            _measures.append(random.choice(DISTANCE_MEASURES))
            for i in self.incoming_classes:
                _val = random.choice(X[y == i])
                _exemplars[j].append(_val)
        _exemplars = np.array(_exemplars)
        return _measures, _exemplars

    def _generate_best_splitters(
        self,
        X_incoming_train,  # Array of X values (ndarray)
        y_incoming_train,  # Array of y values (ndarray)
        exemplars,  # List of exemplars, each exemplar contains a series for each class
        distance_measures: List,  # List of distance measures
    ):
        """
        Generate splitters that maximize the difference in Gini
        Difference.

        Parameters:
        -----------
        X_incoming_train : np.ndarray
            Incoming data.
        y_incoming_train : np.ndarray
            Incoming target values.
        exemplars : np.ndarray
            Exemplars.
        distance_measures : List[callable]
            Distance measures.

        Returns:
        --------
        Tuple[callable, np.ndarray]
            Best distance measure and exemplars.
        """
        scores = np.zeros(len(exemplars))
        y_branch = [[] for _ in range(len(np.unique(y_incoming_train)))]
        for i, exemplar in enumerate(exemplars):
            dist = distance_measures[i]
            _distance = np.zeros((len(X_incoming_train), len(exemplars[0])))
            for idxs, series in enumerate(exemplar):
                for idxx, x_values in enumerate(X_incoming_train):
                    _distance[idxx, idxs] = func_with_params(
                        dist, np.array(x_values), np.array(series)
                    )

            indices = np.argmin(_distance, axis=1)
            for yvals, _idx in zip(y_incoming_train, indices):
                y_branch[_idx].append(yvals)

            scores[i] = gini_difference(y_incoming_train, y_branch)

        idx = np.argmax(scores)
        return distance_measures[idx], np.array(exemplars[idx])

    def _generate_branched_data(
        self,
        X_incoming,
        y_incoming: np.ndarray = None,  # Only useful while fitting the Model
    ):
        """
        Generate branched data.

        Parameters:
        -----------
        X_incoming : np.ndarray
            Incoming data.
        y_incoming : np.ndarray, optional
            Incoming target values. Defaults to None.

        Returns:
        --------
        Union[np.ndarray, [np.ndarray, np.ndarray]]
            Branched data or branched data and target values.
        """
        if not self.is_fit:
            raise NotFittedError("Fit the Node to get branched data.")

        dist = self.distance_measures
        X_branch = [[] for _ in range(self.num_next_nodes)]
        _distance = np.zeros((X_incoming.shape[0], len(self.exemplars)))
        for idxs, series in enumerate(self.exemplars):
            for idxx, x_values in enumerate(X_incoming):
                _distance[idxx, idxs] = func_with_params(dist, x_values, series)

        indices = np.argmin(_distance, axis=1)
        for xvals, _idx in zip(X_incoming, indices):
            X_branch[_idx].append(xvals)

        if y_incoming is not None:
            y_branch = [[] for _ in range(self.num_next_nodes)]
            for yvals, _idx in zip(y_incoming, indices):
                y_branch[_idx].append(yvals)
            return np.array(X_branch), np.array(y_branch)

        return np.array(X_branch)

    def _generate_next_nodes(self):
        """Generate next nodes."""
        self.next_nodes = []
        if not self.leaf_node:
            for _ in range(self.num_next_nodes):
                self.next_nodes.append(ProximityTreeNode(depth=self.depth + 1))
