from data import GraphDataset
from positional_encoding import NodePositionalEmbeddings, AttentionPositionalEmbeddings
from torch_geometric.datasets import ZINC
import torch_geometric

import graph_tool as gt
import graph_tool.topology as top
import networkx as nx

import torch
import numpy as np
import pandas as pd
import ipdb

ZINC_PATH = '/scratch/curan/rmenegau/torch_datasets/ZINC'

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
            "beta": 1
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

# def get_rings(edge_index, max_k=7):
#     if isinstance(edge_index, torch.Tensor):
#         edge_index = edge_index.numpy()

#     edge_list = edge_index.T
#     graph_gt = gt.Graph(directed=False)
#     graph_gt.add_edge_list(edge_list)
#     gt.stats.remove_self_loops(graph_gt)
#     gt.stats.remove_parallel_edges(graph_gt)
#     # We represent rings with their original node ordering
#     # so that we can easily read out the boundaries
#     # The use of the `sorted_rings` set allows to discard
#     # different isomorphisms which are however associated
#     # to the same original ring – this happens due to the intrinsic
#     # symmetries of cycles
#     rings = set()
#     sorted_rings = set()
#     for k in range(3, max_k+1):
#         pattern = nx.cycle_graph(k)
#         pattern_edge_list = list(pattern.edges)
#         pattern_gt = gt.Graph(directed=False)
#         pattern_gt.add_edge_list(pattern_edge_list)
#         sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
#                                            generator=True)
#         sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
#         for iso in sub_iso_sets:
#             if tuple(sorted(iso)) not in sorted_rings:
#                 rings.add(iso)
#                 sorted_rings.add(tuple(sorted(iso)))
#                 # Remove rings that are composed of 2 smaller rings
#     small_rings = set()
#     small_rings.update(rings)
#     # for a in rings:
#     #     set_a = set(a)
#     #     for b in rings:
#     #         set_b = set(b)
#     #         if set_b != set_a:
#     #             for c in rings:
#     #                 set_c = set(c)
#     #                 if set_c != set_b and set_c != set_a:
#     #                     d, e, f = sorted([set_a, set_b, set_c], key=len)
#     #                     if e != f and d.union(e).issubset(f):
#     #                         small_rings.discard(tuple(sorted(f)))
#     rings = list(small_rings)
#     return rings

# rings = get_rings(graph.edge_index)
# graph_nx = torch_geometric.utils.to_networkx(graph, to_undirected=True)

# import matplotlib.pyplot as plt
# plt.figure()
# nx.draw(graph_nx, with_labels=True)
# plt.savefig("last_graph.png")


print('finished')