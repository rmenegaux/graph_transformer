from data import GraphDataset
from positional_encoding import NodePositionalEmbeddings, AttentionPositionalEmbeddings
from torch_geometric.datasets import ZINC

import torch
import numpy as np
import pandas as pd
import ipdb

ZINC_PATH = '/scratch/curan/rmenegau/torch_datasets/ZINC2'

dataset = {
        split: ZINC(root=ZINC_PATH, subset=True, split=split) for split in ['train', 'val', 'test']
    }
trainset = GraphDataset(dataset['train'])
valset = GraphDataset(dataset['val'])
# testset = GraphDataset(dataset['test'])

# # node pe
# node_pe_params = {
#             "node_pe": "rand_walk",
#             "p_steps": 16
#         }
# NodePE = NodePositionalEmbeddings[node_pe_params['node_pe']](**node_pe_params)
# for dset in [trainset, valset, testset]:
#     dset.compute_node_pe(NodePE, standardize=True)
# node_pe_dim = NodePE.get_embedding_dimension()

# num_nodes = []
# node_pe_list = []

# for i, graph in enumerate(trainset.dataset):
#     num_nodes.append(graph.x.size(0))
#     node_pe_list.append(graph.node_pe)

# all_node_pe = torch.cat(node_pe_list, dim=0)

# graph_dataframe = pd.DataFrame({'num_nodes': num_nodes, 'node_pe': [pe.mean(0) for pe in node_pe_list]})
# mean_per_nodes_pandas = graph_dataframe.groupby('num_nodes').sum() / graph_dataframe.groupby('num_nodes').count()
# mean_per_nodes = torch.stack(tuple(mean_per_nodes_pandas['node_pe']))
# ipdb.set_trace()


attention_pe_params = {
            "attention_pe": "multi_RW",
            "multi_attention_pe": "per_layer",
            "zero_diag": False,
            "p_steps": 16,
            "beta": 0.25
        }
AttentionPE = AttentionPositionalEmbeddings[attention_pe_params['attention_pe']](**attention_pe_params)
#for dset in [trainset, valset, testset]:
#    dset.compute_attention_pe(AttentionPE, update_stats=True, standardize=True)
valset.compute_attention_pe(AttentionPE, update_stats=True, standardize=False)
attention_pe_dim = AttentionPE.get_dimension()

attention_pe_list = []
num_nodes = []
for i, graph in enumerate(valset.dataset):
    num_nodes.append(graph.x.size(0))
    attention_pe_list.append(graph.attention_pe)
AttentionPE.get_statistics()

ipdb.set_trace()
AttentionPE.standardize(graph.attention_pe)
print('finished')
# graph_dataframe.group

# if i%10==0: print(i); 
# def compute_statistics(dset):
#     count = 0
#     num_nodes = []
#     node_pe = []
#     attention_pe = []

#     for graph in dset.dataset:
#         num_nodes.append(graph.x.size(0))
#         node_pe.append(graph.node_pe)
#         attention_pe