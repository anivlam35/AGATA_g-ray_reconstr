import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import tensorflow as tf
import numpy as np


import scipy.sparse as sp
from spektral.data import Dataset, DisjointLoader, Graph
from spektral.layers import GCNConv
from keras.layers import Input, Dense, Dropout
from keras.models import Model


with open('scrapped_data_mul1_small_graph_mode.json', "r") as json_file:
    json_data = json.loads(json_file.read())


X, Y, A = json_data["node_features"], json_data["fully_absorbed"], json_data["adjacency_matrix"]

number_of_graphs = len(X)
A = [np.array(A.get(i), dtype = 'int64') for i in A.keys()]
X = [np.array(X.get(i), dtype = 'float64') for i in X.keys()]


class MyDataset(Dataset):
    def __init__(self, n_samples, **kwargs):
        self.n_samples = n_samples
        super().__init__(**kwargs)

    def read(self):
        def make_graph(j):
            a = np.ones(A[j].shape[0]) - np.identity(A[j].shape[0])
#             e = sp.csr_matrix(E[j])
            return Graph(x=X[j], a=a)

        # We must return a list of Graph objects
        return [make_graph(j) for j in range(self.n_samples)]


learning_rate = 0.0001  # Learning rate
epochs = 200  # Number of training epochs
es_patience = 30  # Patience for early stopping
batch_size = 500  # Batch size

data = MyDataset(number_of_graphs)


data_tr = data[0:int(len(data)*0.6)]
data_va = data[int(len(data)*0.6):int(len(data)*0.8)]
data_te = data[int(len(data)*0.8):len(data)]


# loader_tr = DisjointLoader(data_tr, batch_size=batch_size)
# loader_va = DisjointLoader(data_va, node_level=True, batch_size=batch_size)
# loader_te = DisjointLoader(data_te, node_level=True, batch_size=batch_size)
# loader_main = DisjointLoader(data, node_level=True, batch_size=batch_size)

def create_model(n_features, n_classes):
    X_in = Input(shape=(None,))
    A_in = Input(shape=(None, None))

    # Define the GCN layers with Dropout
    gc1 = GCNConv(16, activation='relu')([X_in, A_in])
    drop1 = Dropout(0.2)(gc1)
    gc2 = GCNConv(16, activation='relu')([drop1, A_in])
    drop2 = Dropout(0.2)(gc2)

    # Define the output layer
    output = Dense(n_classes, activation='softmax')(drop2)

    # Create the Keras model
    model = Model(inputs=[X_in, A_in], outputs=output)

    return model
