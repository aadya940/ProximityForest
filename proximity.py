import numpy as np
import random
from typing import List

from sklearn.preprocessing import LabelEncoder
from aeon.classification.base import BaseClassifier
from gini import *
from node import *


from aeon.exceptions import NotFittedError

# TODO : Set random_state param


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


