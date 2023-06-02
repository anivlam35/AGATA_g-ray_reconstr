import json, numpy as np

with open('scrapped_data_mul1_small_graph_mode.json', "r") as json_file:
    json_data = json.loads(json_file.read())


X, Y, A = json_data["node_features"], json_data["fully_absorbed"], json_data["adjacency_matrix"]

A = [np.array(A.get(i), dtype = 'int64') for i in A.keys()]
X = [np.array(X.get(i), dtype = 'float64') for i in X.keys()]

print(X[:3])
print(A[:3])