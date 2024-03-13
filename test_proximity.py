from proximity import (
    ProximityTreeClassifier,
    ProximityTreeNode,
    DISTANCE_MEASURES,
    func_with_params,
)
import numpy as np
import unittest


def test_node():
    X = np.random.normal(0, 1, (15, 3))
    y = np.array([np.random.choice([1, 2, 3]) for _ in range(15)])

    node = ProximityTreeNode(depth=0)
    node._fit(X, y, X, y, [1, 2, 3])

    print(node.exemplars)
    print(node.depth)
    print(node.incoming_classes)
    print(node.leaf_node)
    print(node.next_nodes)
    print(node.distance_measures.__name__)
    print(func_with_params(node.distance_measures, X[5], X[6]))

    measures, exemplars = node._generate_candidate_splitters(X, y, num_candidates=15)

    print("PRINTING CANDIDATE EXEMPLARS : ")
    print("-------------------------------")
    for i, j in zip(measures, exemplars):
        print(i.__name__)
        print(j)

    print("\n")

    dm, exem = node._generate_best_splitters(
        X, y, exemplars=exemplars, distance_measures=measures
    )

    print("Best Distance measure is : ", dm.__name__)
    print("Best Exemplars are : \n")
    print(exem)

    print("Generate Branched Data : ")
    print("-------------------------")
    branched_data = node._generate_branched_data(X)
    for idx, data in enumerate(branched_data):
        print(f"Branch {idx}: ")
        print("---------------")
        print(np.array(data))
        print("\n")


class TestProximityTreeClassifier(unittest.TestCase):
    def setUp(self):
        self.X_train = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]])
        self.y_train = np.array([0, 0, 1, 1, 1])  # Two classes
        self.X_test = np.array([[1, 1, 1], [5, 5, 5]])

    def test_fit(self):
        clf = ProximityTreeClassifier()
        clf._fit(self.X_train, self.y_train)
        assert clf.root_node is not None
        assert clf.root_node.next_nodes is not None
        assert isinstance(clf.root_node.next_nodes, list)
        assert len(clf.root_node.next_nodes) == 2


if __name__ == "__main__":
    print("Test the NODE: \n")
    print("-----------------")
    test_node()

    print("TEST THE CLASSIFIER:  \n")
    print("--------------------")
    unittest.main()
