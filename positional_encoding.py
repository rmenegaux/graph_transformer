import torch

import torch_geometric.utils as utils

from scipy.linalg import expm

def standardize(M, mean, std, threshold=1e-6):
    return (M - mean) / std.clamp(min=threshold)

def compute_RW_from_adjacency(A, add_self_loops=False):
    '''
    Returns the random walk transition matrix for an adjacency matrix A
    '''
    if add_self_loops:
        I = torch.eye(*A.size(), out=torch.empty_like(A))
        A = I + A
    D = A.sum(dim=-1, keepdim=True)
    D[D == 0] = 1 # Prevent any division by 0 errors
    return A / D # A D^-1

def compute_normalized_RW_from_adjacency(A, add_self_loops=False, epsilon=0.125):
    '''
    Returns the random walk transition matrix for an adjacency matrix A
    '''
    if add_self_loops:
        I = torch.eye(*A.size(), out=torch.empty_like(A))
        A = I + A
    D = A.sum(dim=-1)
    D[D == 0] = 1 # Prevent any division by 0 errors
    D = D.pow(-epsilon)
    return A * D.view(-1, 1) * D.view(1, -1) # D_eps A D_eps
    
def get_laplacian_from_adjacency(A):
    RW = compute_RW_from_adjacency(A)
    I = torch.eye(*RW.size(), out=torch.empty_like(RW))
    return I - RW

def RW_kernel_from_adjacency(A, beta=0.25, p_steps=1):
    '''
    Returns the random walk kernel matrix for an adjacency matrix A
    '''
    L = get_laplacian_from_adjacency(A)
    I = torch.eye(*L.size(), out=torch.empty_like(L))
    k_RW = I - beta * L
    k_RW_power = k_RW
    for power in range(p_steps-1):
        k_RW_power = k_RW_power @ k_RW
    return k_RW_power


class RandomWalkNodePE(object):
    '''
    Returns a p_step-dimensional vector p for each node,
    with p_i = RW^i, the probability of landing back on that node after i steps in the graph
    '''
    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        # Keep track of statistics
        self.sum_ = 0
        self.sum_squares_ = 0
        self.num_nodes_ = 0

    def get_embedding_dimension(self):
        return self.p_steps

    def get_statistics(self):
        '''
        Return mean and standard deviation of positional embeddings computed with `update_stats=True`
        '''
        mean = self.sum_ / self.num_nodes_
        var = self.sum_squares_ / self.num_nodes_ - mean**2
        return mean, torch.sqrt(var)

    def __call__(self, graph, update_stats=True):
        num_nodes = len(graph.x)
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        RW = compute_RW_from_adjacency(A)
        RW_power = RW
        node_pe = torch.zeros((num_nodes, self.p_steps))
        node_pe[:, 0] = RW.diagonal()
        for power in range(self.p_steps-1):
            RW_power = RW @ RW_power
            node_pe[:, power + 1] = RW_power.diagonal()
        if update_stats:
            self.num_nodes_ += num_nodes
            self.sum_ += node_pe.sum(0)
            self.sum_squares_ += (node_pe**2).sum(0)

        return node_pe

class IterableNodePE(object):
    '''
    A disguised list, containing precomputed positional encodings. Careful indexing is required
    '''
    def __init__(self, pe_list, **parameters):
        self.current_index = 0
        self.pe_list = pe_list
        self.embedding_dimension = parameters.get('embedding_dimension', None)

    def get_embedding_dimension(self):
        return self.embedding_dimension

    def __call__(self, graph):
        node_pe = self.pe_list[self.current_index]
        self.current_index += 1
        return node_pe


class BaseAttentionPEWrong_(object):
    '''
    Base class for computing edge positional encodings

    FIXME: Implement a `process_dataset` method
    '''

    def __init__(self, **parameters):
        '''
        Parameters that are applicable to any Attention PE
        '''
        self.zero_diag = parameters.get('zero_diag', False)
        # Maybe keep track of running means rather than sums, to avoid overflow
        # For now no problem as the quantities summed are << 1 (< n_nodes total per graph)
        self._running_stats = {
            'sum_off_diagonal': 0,
            'sum_diagonal': 0,
            'sum_squares_off_diagonal': 0,
            'sum_squares_diagonal': 0,
            'num_nodes': 0,
        }

    def __call__(self, graph, update_stats=True):
        K = self.compute_attention_pe(graph)

        if self.zero_diag:
            K.fill_diagonal_(0)

        if update_stats:
            self._running_stats['num_nodes'] += K.size(0)
            diagonal = K.diagonal()
            diagonal_sum = diagonal.sum(-1)
            diagonal_squared_sum = (diagonal**2).sum(-1)
            self._running_stats['sum_diagonal'] += diagonal_sum
            self._running_stats['sum_squares_diagonal'] += diagonal_squared_sum
            self._running_stats['sum_off_diagonal'] += K.sum((0, 1)) - diagonal_sum
            self._running_stats['sum_squares_off_diagonal'] += (K**2).sum((0, 1)) - diagonal_squared_sum

        return K


    def get_statistics(self):
        num_nodes = self._running_stats['num_nodes']
        if num_nodes == 0:
            print('No statistics available')
            return None
        num_off_diagonal = num_nodes * (num_nodes - 1)
        mean_diagonal = self._running_stats['sum_diagonal'] / num_nodes
        mean_off_diagonal = self._running_stats['sum_off_diagonal'] / num_off_diagonal
        var_diagonal = self._running_stats['sum_squares_diagonal'] / num_nodes - mean_diagonal**2
        var_off_diagonal = self._running_stats['sum_squares_off_diagonal'] / num_off_diagonal - mean_off_diagonal**2
        return {
            'mean_diagonal': mean_diagonal,
            'mean_off_diagonal': mean_off_diagonal,
            'std_diagonal': torch.sqrt(var_diagonal),
            'std_off_diagonal': torch.sqrt(var_off_diagonal),
            'num_nodes': num_nodes,
        }

    def standardize(self, attention_pe):
        '''
        Return a standardized copy of `attention_pe`
        '''
        stats = self.get_statistics()
        # Store a copy of the standardized diagonal
        diagonal = standardize(attention_pe.diagonal().transpose(0, 1), stats['mean_diagonal'], stats['std_diagonal'])
        # Standardize off diagonal elements
        attention_pe_std = standardize(attention_pe, stats['mean_off_diagonal'], stats['std_off_diagonal'])
        # Set diagonal
        num_nodes = attention_pe.size(0)
        attention_pe_std[range(num_nodes), range(num_nodes)] = diagonal

        return attention_pe_std

    def compute_attention_pe(self):
        pass

    def get_dimension(self):
        '''
        Returns the size of K's last dimension
        '''
        return 1

class BaseAttentionPE(object):
    '''
    Base class for computing edge positional encodings

    FIXME: Implement a `process_dataset` method
    '''

    def __init__(self, **parameters):
        '''
        Parameters that are applicable to any Attention PE
        '''
        self.zero_diag = parameters.get('zero_diag', False)
        # Maybe keep track of running means rather than sums, to avoid overflow
        # For now no problem as the quantities summed are << 1 (< n_nodes total per graph)
        self._running_stats = {
            'sum': 0,
            'sum_squares': 0,
            'num_edges': 0,
        }

    def __call__(self, graph, update_stats=True):
        K = self.compute_attention_pe(graph)

        if self.zero_diag:
            K.fill_diagonal_(0)

        if update_stats:
            self._running_stats['num_edges'] += K.size(0)**2
            self._running_stats['sum'] += K.sum((0, 1))
            self._running_stats['sum_squares'] += (K**2).sum((0, 1))

        return K

    def get_statistics(self):
        num_edges = self._running_stats['num_edges']
        if num_edges == 0:
            print('No statistics available')
            return None
        mean = self._running_stats['sum'] / num_edges
        var = self._running_stats['sum_squares'] / num_edges - mean**2
        return {
            'mean': mean,
            'std': torch.sqrt(var),
            'num_edges': num_edges,
        }

    def standardize(self, attention_pe):
        '''
        Return a standardized copy of `attention_pe`
        '''
        stats = self.get_statistics()

        return standardize(attention_pe, stats['mean'], stats['std'])

    def compute_attention_pe(self):
        pass

    def get_dimension(self):
        '''
        Returns the size of K's last dimension
        '''
        return 1


class RandomWalkAttentionPE(BaseAttentionPE):

    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.beta = parameters.get('beta', 0.25)
        super().__init__(**parameters)

    def compute_attention_pe(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        k_RW_power = RW_kernel_from_adjacency(A, beta=self.beta, p_steps=self.p_steps)
        
        return k_RW_power

    def get_dimension(self):
        return 1


class DiffusionAttentionPE(BaseAttentionPE):
    def __init__(self, **parameters):
        self.beta = parameters.get('beta', 0.5)
        super().__init__(**parameters)

    def compute_attention_pe(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        L = get_laplacian_from_adjacency(A)
        attention_pe = expm(-self.beta * L.numpy())
        return torch.from_numpy(attention_pe)
    
    def get_dimension(self):
        return 1


class EdgeRWAttentionPE(BaseAttentionPE):
    '''
    Computes a separate random walk kernel for each edge type
    '''
    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.beta = parameters.get('beta', 0.25)
        self.num_edge_type = parameters.get('num_edge_type', 3)
        super().__init__(**parameters)

    def compute_attention_pe(self, graph):
        k_RW_power = []
        for edge_type in range(self.num_edge_type):
            # Build adjacency matrix for each edge type
            edge_attr = (graph.edge_attr == edge_type + 1).long()
            A = utils.to_dense_adj(graph.edge_index, edge_attr=edge_attr).squeeze()
            k_RW_power.append(RW_kernel_from_adjacency(A, beta=self.beta, p_steps=self.p_steps))
        return torch.stack(k_RW_power, dim=-1)

    def get_dimension(self):
        return self.num_edge_type


class MultiRWAttentionPE(BaseAttentionPE):
    '''
    Computes the random walk kernel for all number of steps from 1 to self.p_steps
    '''
    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.stride = parameters.get('stride', 1)
        self.beta = parameters.get('beta', 0.25)
        super().__init__(**parameters)
    
    def compute_attention_pe(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        k_RW_0 = RW_kernel_from_adjacency(A, beta=self.beta, p_steps=1)
        # k_RW_0 = compute_normalized_RW_from_adjacency(A)
        k_RW_power = k_RW_0
        k_RW_all_powers = [k_RW_0]
        for i in range(self.p_steps-1):
            for _ in range(self.stride):
                k_RW_power = k_RW_power @ k_RW_0
            k_RW_all_powers.append(k_RW_power)
        attention_pe = torch.stack(k_RW_all_powers, dim=-1)
        return attention_pe

    def get_dimension(self):
        return self.p_steps

class MultiDiffusionAttentionPE(BaseAttentionPE):
    '''
    Computes the diffusion kernel for all number of steps from 1 to self.p_steps
    '''
    def __init__(self, **parameters):
        self.p_steps = parameters.get('p_steps', 16)
        self.beta = parameters.get('beta', 0.25)
        super().__init__(**parameters)
    
    def compute_attention_pe(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        L = get_laplacian_from_adjacency(A)
        k_diff_0 = torch.from_numpy(expm(-self.beta * L.numpy()))
        k_diff_power = k_diff_0
        k_diff_all_powers = [k_diff_0]
        for i in range(self.p_steps-1):
            k_diff_power = k_diff_power @ k_diff_0
            k_diff_all_powers.append(k_diff_power)
        return torch.stack(k_diff_all_powers, dim=-1)

    def get_dimension(self):
        return self.p_steps


class AdjacencyAttentionPE(BaseAttentionPE):

    def __init__(self, **parameters):
        super().__init__(**parameters)

    def compute_attention_pe(self, graph):
        A = utils.to_dense_adj(graph.edge_index).squeeze()
        return A / A.sum(dim=-1)

    def get_dimension(self):
        return 1


NodePositionalEmbeddings = {
    'rand_walk': RandomWalkNodePE,
    'gckn': IterableNodePE
}

AttentionPositionalEmbeddings = {
    'rand_walk': RandomWalkAttentionPE,
    'edge_RW': EdgeRWAttentionPE,
    'multi_RW': MultiRWAttentionPE,
    'multi_diffusion': MultiDiffusionAttentionPE,
    'adj': AdjacencyAttentionPE,
    'diffusion': DiffusionAttentionPE,
}