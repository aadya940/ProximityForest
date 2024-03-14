import numpy as np
import random
from typing import List

from sklearn.preprocessing import LabelEncoder
from aeon.classification.base import BaseClassifier
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

# Sometimes the Tests fail (Tree Depth Assertion).
# Returns different depths

# TODO : Set random_state param

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


class ProximityTreeClassifier(BaseClassifier):
    """Initialize ProximityTreeClassifier."""

    def __init__(self, max_depth=5):
        super().__init__()
        self.max_depth = max_depth
        self.root_node = None
        self.tree_depth = 0

    def _fit(self, X, y, num_candidates_for_selection=5, min_samples_split=2):
        """
        Fit the classifier.

        Parameters:
        -----------
        X : np.ndarray
            Data.
        y : np.ndarray
            Target values.
        num_candidates_for_selection : int, optional
            Number of candidates for selection. Defaults to 5.
        max_depth : int, optional
            Maximum depth of the tree. Defaults to 5.
        min_samples_split : int, optional
            Minimum number of samples required to split a node. Defaults to 2.
        """
        (
            self.n_instances_,
            self.n_atts_,
        ) = X.shape  # Num rows = n_instances, Num cols = n_atts_
        self.classes_ = np.unique(y)  # Classes
        self.n_classes_ = self.classes_.shape[0]  # Number of Class
        self._class_dictionary = {}
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index  # Class Indices

        # escape if only one class seen
        if self.n_classes_ == 1:
            self._is_fitted = True
            self.root_node = ProximityTreeNode(depth=0)
            self.root_node.leaf_node = True
            self.root_node.num_next_nodes = 0
            return

        if not np.issubdtype(self.classes_.dtype, np.integer):
            le = LabelEncoder()
            y = le.fit_transform(y)

        self.root_node = ProximityTreeNode(depth=0)
        self._recursive_fit(
            self.root_node,
            X,
            y,
            X,
            y,
            num_candidates_for_selection=num_candidates_for_selection,
            max_depth=self.max_depth,
            min_samples_split=min_samples_split,
        )
        self._is_fitted = True

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
        """
        Recursively Build and Fit the Tree.

        Parameters:
        -----------
        node : ProximityTreeNode
            Current node.
        X : np.ndarray
            Data.
        y : np.ndarray
            Target values.
        X_branch : np.ndarray
            Branched data.
        y_branch : np.ndarray
            Branched target values.
        num_candidates_for_selection : int
            Number of candidates for selection.
        max_depth : int
            Maximum depth of the tree.
        min_samples_split : int
            Minimum number of samples required to split a node.
        """
        if (
            len(np.unique(y)) == 1
            or node.depth == max_depth
            or len(y) < min_samples_split
        ) or (len(X_branch) == 0 or len(y_branch) == 0):
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

        if len(node.next_nodes) > 0:
            X_branches, y_branches = node._generate_branched_data(X, y)
            for i, next_node in enumerate(node.next_nodes):
                next_node._fit(
                    X,
                    y,
                    X_branches[i],
                    y_branches[i],
                    np.unique(y_branches[i]).tolist(),
                    num_candidates_for_selection,
                )

                if next_node.depth > self.tree_depth:
                    self.tree_depth = next_node.depth

                if (len(X_branches[i]) != 0) and (len(y_branches[i]) != 0):
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
        if not self._is_fitted:
            raise NotFittedError("Can't predict if the Model is not Fitted")

        raise NotImplementedError

    def get_tree_depth(self):
        """
        Get the depth of the tree.

        Returns:
        --------
        int
            Depth of the tree.
        """
        return self.tree_depth


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
