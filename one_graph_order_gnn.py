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
            return Graph(x=X[j], a=a, a_target=A[j])

        self.graphs = [make_graph(j) for j in range(self.n_samples)]
        # self.targets = A
        return self.graphs

    # def get_element(self, i):
    #     target = tf.reshape(A[i], [-1])
    #
    #     return self.graphs[i], target


# Create the GNN model
class GNNModel(tf.keras.Model):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(32, activation='relu')
        self.conv2 = GCNConv(2, activation='softmax')

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        return x

learning_rate = 0.001  # Learning rate
epochs = 200  # Number of training epochs
es_patience = 30  # Patience for early stopping
batch_size = 512  # Batch size

data = MyDataset(number_of_graphs, target=A)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model = GNNModel()
print(model.summary())
exit()


data_tr = data[0:int(len(data)*0.6)]
data_va = data[int(len(data)*0.6):int(len(data)*0.8)]
data_te = data[int(len(data)*0.8):len(data)]

loader_tr = DisjointLoader(data_tr, batch_size=batch_size, node_level=False)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_te, batch_size=batch_size)
loader_main = DisjointLoader(data, batch_size=batch_size)

print(*loader_tr.__next__())
exit()

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    for batch in loader_tr.load():
        # Unpack the batch data
        graphs = batch
        print(graphs)
        x = graphs.x
        a = graphs.a
        print(a.shape, a)
        print(a_target.shape, a_target)

        with tf.GradientTape() as tape:
            # Forward pass through the model to get the predicted adjacency matrix
            y_pred = model([x, a])
            print(y_pred.shape, y_pred)
            exit()
            y_pred = tf.reshape(y_pred, [-1])

            # Compute the binary cross-entropy loss between the predicted and target matrices
            loss = tf.keras.losses.binary_crossentropy(graphs.a_target, y_pred)

        # Compute and apply the gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(f'Loss = {loss:.4f}')

# Evaluate the model on the entire dataset
y_pred_list = []

for i in range(len(data_va)):
    graph, target = data_va[i]

    # Normalize the adjacency matrix using the same transform used during training
    graph.apply_transform(NormalizeAdj())

    # Predict the adjacency matrix using the trained model
    y_pred = model.predict([graph.x, graph.a])
    y_pred = np.reshape(y_pred, [3, 3])
    y_pred_list.append(y_pred)

# Print the true and predicted adjacency matrices for each graph:
for i in range(len(dataset)):
    graph, target = dataset[i]
    print(f'Graph {i+1}:')
    print('Target adjacency matrix:')
    print(target.reshape([3, 3]))
    print('Predicted adjacency matrix:')
    print(np.round(y_pred_list[i], 2))

# def create_model(n_features, n_classes):
#     X_in = Input(shape=(len(X[0][0]),))
#     A_in = Input(shape=(None, None))
#
#     # Define the GCN layers with Dropout
#     gc1 = GCNConv(16, activation='relu')([X_in, A_in])
#     drop1 = Dropout(0.2)(gc1)
#     gc2 = GCNConv(16, activation='relu')([drop1, A_in])
#     drop2 = Dropout(0.2)(gc2)
#
#     # Define the output layer
#     output = Dense(n_classes, activation='softmax')(drop2)
#
#     # Create the Keras model
#     model = Model(inputs=[X_in, A_in], outputs=output)
#
#     return model

