#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np
import random


# In[2]:


def r_calculation(dx, dy, dz):
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

def energy_weighted_baryceter_coordinates(row1, row2):
        packed_coords = (row1[['x','y','z']].apply(lambda x: x * row1['edep']) +
               row2[['x','y','z']].apply(lambda x: x * row2['edep']))/(row1['edep'] + row2['edep'])
        packed_coords['edep'] = row1['edep'] + row2['edep']
        
        return packed_coords

def reset_interaction_index(df):
    df = df.reset_index()
    if 'interaction_num' in df.columns:
        df = df.drop('interaction_num', axis=1)
    df['interaction_num'] = df.groupby('event_num').cumcount()
    df.set_index(['event_num', 'interaction_num'], inplace=True)
    return df

def dr_calculation(df):
    df[['dx','dy','dz']] = df.loc[:,['x','y','z']].groupby(level=0).diff().fillna(0)
    df['dr'] = df.apply(lambda x: r_calculation(x.dx, x.dy, x.dz), axis=1)
    df = df.drop(['dx','dy','dz'], axis=1)
    return df

loop = 0

def one_level_packing(df):
    global loop
    df = dr_calculation(df)
    #idp - indexes to pack
    idp = df.loc[(df.dr <= 5) & (df.dr != 0)].index.tolist()
    iter = 1
    for indx in idp:
        packed_coords = energy_weighted_baryceter_coordinates(df.loc[indx], 
                                                              df.loc[tuple([indx[0], indx[1] - 1])])
        df.loc[indx, ['x', 'y', 'z', 'edep']] = packed_coords.values
        df = df.drop(tuple([indx[0], indx[1] - 1]), axis=0)
        print(loop, iter)
        iter += 1
    df = reset_interaction_index(df)
    return df


# In[3]:


file_name = 'scrapped_data_mul1_big'
data = pd.read_csv(file_name + '.csv')
data = reset_interaction_index(data)


# In[4]:


data


# In[5]:


data = dr_calculation(data)
while not data.loc[(data.dr <= 5) & (data.dr != 0)].empty:
    data = one_level_packing(data)
data = reset_interaction_index(data)
data = data.drop('dr', axis=1)


# In[ ]:


data


# In[ ]:


new_data = data.groupby('event_num').sum('edep')
new_data


# In[ ]:


new_data['fully_absorbed'] = (new_data.edep == 1000)
absorption = new_data['fully_absorbed']


# # Smearing

# In[ ]:


smeared_data = data.copy()
smeared_data[['x', 'y', 'z']] = smeared_data[['x', 'y', 'z']].apply(lambda x: x / 10)
# smeared_data['edep'] = smeared_data['edep'].apply(lambda x: x / 1000)
smeared_data


# In[ ]:


def position_sigma_distribution(energy):
    return (.27+.62*np.sqrt(0.1/energy))/2.35
def energy_sigma_distribution(energy):
    res_at_1333 =2.43;
    return np.sqrt(1 + energy*res_at_1333)/2.35
def gaussian(x, sigma):
    return round(random.gauss(x, sigma), 3)


# In[ ]:


smeared_data[['x', 'y', 'z']] = smeared_data[['x', 'y', 'z']].apply(
    lambda x_i: gaussian(x_i, position_sigma_distribution(smeared_data.edep)))
smeared_data['edep'] = smeared_data['edep'].apply(lambda x: gaussian(x, energy_sigma_distribution(x)))
smeared_data


# In[ ]:


# smeared_data.to_csv(file_name + '_PS.csv')


# # Preparing data for Graph making

# In[ ]:


data_fp = smeared_data.copy()
data_fp = data_fp.drop(['index', 'crystal', 'slice_sect'], axis=1)
data_fp


# In[ ]:


data_fp['node_features'] = data_fp[['edep', 'x', 'y', 'z']].apply(lambda x: np.array([i for i in x]), axis=1)
data_fp.time[1][1]


# In[ ]:


node_features = data_fp.groupby('event_num').apply(lambda x: np.array([row for row in x.node_features]))
node_features 


# In[ ]:


prepared_data = node_features.to_frame().join(absorption.to_frame()).rename(columns={0:'node_features'})
prepared_data = prepared_data.loc[prepared_data.node_features.apply(lambda x: len(x) != 1)]

prepared_data


# In[ ]:


# def adjacency_matrix_creating(l):
#     matrix = []
#     for i in range(1, l):
#         row = [0 for _ in range(l)]
#         row[i] = 1
#         matrix.append(np.array(row))
#     matrix.append(np.array([0 for _ in range(l)]))
#     return np.array(matrix)
def shuffling_sequence_creating(x):
    res_list = list(range(len(x)))
    random.shuffle(res_list)
    return res_list

def nodes_shuffling(seq, node_features):
    return np.array([node_features[i] for i in seq])

def shuffled_adjacency_matrix(seq):
    l = len(seq)
    matrix = np.array([np.array([0 for _ in range(l)]) for _ in range(l)])
    for k in range(l - 1):
        i = seq.index(k)
        j = seq.index(k + 1)
        matrix[i][j] = 1
    return matrix


# In[ ]:


prepared_data['shuffling_sequence'] = prepared_data['node_features'].apply(
    lambda x: shuffling_sequence_creating(x))
prepared_data['adjacency_matrix'] = prepared_data['shuffling_sequence'].apply(
    lambda x: shuffled_adjacency_matrix(x))
prepared_data['node_features'] = prepared_data.apply(
    lambda x: nodes_shuffling(x.shuffling_sequence, x.node_features), axis=1)
prepared_data = prepared_data.drop('shuffling_sequence', axis=1)
prepared_data.to_json(file_name + '_graph_mode.json')


# ## Shuffling graphs's nodes

# In[ ]:


# def shuffle_sequence_creating(x):
#     res_list = list(range(len(x)))
#     random.shuffle(res_list)
#     return res_list

# def nodes_shuffling(seq, node_features):
#     return np.array([node_features[i] for i in seq])

# def shuffled_adjacency_matrix(seq):
#     l = len(seq)
#     matrix = np.array([np.array([0 for _ in range(l)]) for _ in range(l)])
#     for k in range(l - 1):
#         i = seq.index(k)
#         j = seq.index(k + 1)
#         matrix[i][j] = 1
#     return matrix


# In[ ]:


# prepared_data['shuffle_sequence'] = prepared_data['node_features'].apply(
#     lambda x: shuffle_sequence_creating(x))
# prepared_data['new_adj_mtrx'] = prepared_data['shuffle_sequence'].apply(lambda x: adj_matrx_shuffling(x))
# prepared_data
# prepared_data['new_n_ftrs'] = prepared_data[['node_features', 'shuffle_sequence']].apply(
#     lambda x: nodes_shuffling(x.shuffle_sequence, x.node_features), axis=1)
# prepared_data[['node_features', 'shuffle_sequence', 'new_n_ftrs']].loc[2]

