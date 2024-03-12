from proximity import ProximityTreeNode, DISTANCE_MEASURES, func_with_params
import numpy as np

X = np.random.normal(0, 1, (15, 3))
y = np.array([np.random.choice([1, 2, 3]) for _ in range(15)])

node = ProximityTreeNode(depth = 0)
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

dm, exem = node._generate_best_splitters(X, y, exemplars=exemplars, distance_measures=measures)

print("Best Distance measure is : ", dm.__name__)
print("Best Exemplars are : \n")
print(exem)

print("Generate Branched Data : ")
print("-------------------------")
for idx, data in enumerate(node._generate_branched_data(X)):
    print(f"Branch {idx}: ")
    print("---------------")
    print(np.array(data))
    print("\n")
