import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
    GraphiT-GT
    
"""

"""
    Single Attention Head
"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, in_dim_edges, out_dim, num_heads, double_attention=False,
                 use_bias=False, adaptive_edge_PE=True, use_edge_features=False):
        super().__init__()
        
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.double_attention = double_attention
        self.use_edge_features = use_edge_features
        self.adaptive_edge_PE = adaptive_edge_PE
        
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        if self.use_edge_features:
            self.E = nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)

        if self.double_attention:
            self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.E_2 = nn.Linear(in_dim_edges, out_dim * num_heads, bias=use_bias)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        
    def forward(self, h, e, k_RW=None, mask=None, adj=None):
        
        Q_h = self.Q(h) # [n_batch, num_nodes, out_dim * num_heads]
        K_h = self.K(h)
        V_h = self.V(h)

        n_batch = Q_h.size()[0]
        num_nodes = Q_h.size()[1]

        # Reshaping into [num_heads, num_nodes, feat_dim] to 
        # get projections for multi-head attention
        Q_h = Q_h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim)
        K_h = K_h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim)
        V_h = V_h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim).transpose(2, 1) # [n_batch, num_heads, num_nodes, out_dim]

        if self.double_attention:
            Q_2h = self.Q_2(h) # [n_batch, num_nodes, out_dim * num_heads]
            K_2h = self.K_2(h)

            Q_2h = Q_2h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim)
            K_2h = K_2h.reshape(n_batch, num_nodes, self.num_heads, self.out_dim)
            
        # Normalize by sqrt(head dimension)
        scaling = float(self.out_dim) ** -0.5
        K_h = K_h * scaling

        if self.use_edge_features:

            E = self.E(e)   # [n_batch, num_nodes * num_nodes, out_dim * num_heads]
            E = E.reshape(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)

            if self.double_attention:
                edge_filter = adj.view(n_batch, num_nodes, num_nodes, 1, 1)
    
                E = E * edge_filter
                scores = torch.einsum('bihk,bjhk,bijhk->bhij', Q_h, K_h, E)

                E_2 = self.E_2(e)
                E_2 = E_2.reshape(n_batch, num_nodes, num_nodes, self.num_heads, self.out_dim)
                E_2 = E_2 * (~edge_filter)
                scores = scores + torch.einsum('bihk,bjhk,bijhk->bhij', Q_2h, K_2h, E_2)
            else:
                # attention(i, j) = sum(Q_i * K_j * E_ij)
                scores = torch.einsum('bihk,bjhk,bijhk->bhij', Q_h, K_h, E)
        else:
            if self.double_attention:
                edge_filter = adj.view(n_batch, num_nodes, num_nodes, 1, 1)
                scores_1 = torch.einsum('bihk,bjhk->bhij', Q_h, K_h)
                scores_2 = torch.einsum('bihk,bjhk->bhij', Q_2h, K_2h)
                scores = edge_filter * scores_1 + (~edge_filter) * scores_2
            else:
                # attention(i, j) = sum(Q_i * K_j)
                scores = torch.einsum('bihk,bjhk->bhij', Q_h, K_h)

        # Apply exponential and clamp for numerical stability
        scores = torch.exp(scores.clamp(-5, 5)) # [n_batch, num_heads, num_nodes, num_nodes]

        # Make sure attention scores for padding are 0
        if mask is not None:
            scores = scores * mask.view(-1, 1, num_nodes, 1) * mask.view(-1, 1, 1, num_nodes)

        if self.adaptive_edge_PE:
            # Introduce new dimension for the different heads
            k_RW = k_RW.unsqueeze(1)
            scores = scores * k_RW
        
        softmax_denom = scores.sum(-1, keepdim=True).clamp(min=1e-6) # [n_batch, num_heads, num_nodes, 1]

        h = scores @ V_h # [n_batch, num_heads, num_nodes, out_dim]
        # Normalize scores
        h = h / softmax_denom
        # Concatenate attention heads
        h = h.transpose(2, 1).reshape(-1, num_nodes, self.num_heads * self.out_dim) # [n_batch, num_nodes, out_dim * num_heads]

        return h
    

class GraphiT_GT_Layer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, **layer_params):
                #  double_attention=False, dropout=0.0,
                #  layer_norm=False, batch_norm=True, residual=True, adaptive_edge_PE=False,
                #  use_edge_features=True, update_edge_features=False, update_pos_enc=False, use_bias=False
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = layer_params['dropout']
        self.residual = layer_params['residual']
        self.layer_norm = layer_params['layer_norm']     
        self.batch_norm = layer_params['batch_norm']
        self.feedforward = layer_params['feedforward']
        self.update_edge_features = layer_params['update_edge_features']
        self.update_pos_enc = layer_params['update_pos_enc']
        self.concat_h_p = layer_params['concat_h_p']

        attention_params = {
            param: layer_params[param] for param in ['double_attention', 'use_bias', 'adaptive_edge_PE', 'use_edge_features']
        }
        # in_dim*2 if positional embeddings are concatenated rather than summed
        in_dim_h = in_dim*2 if self.concat_h_p else in_dim
        self.attention_h = MultiHeadAttentionLayer(in_dim_h, in_dim, out_dim//num_heads, num_heads, **attention_params)
        self.O_h = nn.Linear(out_dim, out_dim)
        
        if self.update_pos_enc:
            self.attention_p = MultiHeadAttentionLayer(in_dim, in_dim, out_dim//num_heads, num_heads, **attention_params)
            self.O_p = nn.Linear(out_dim, out_dim)
        
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        if self.feedforward:
            self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
            self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

        if self.update_edge_features:
            self.B1 = nn.Linear(out_dim, out_dim)
            self.B2 = nn.Linear(out_dim, out_dim)
            self.E12 = nn.Linear(out_dim, out_dim)
            # self.batch_norm_e = nn.BatchNorm1d(out_dim)

    def forward_edges(self, h, e):
        '''
        Update edge features
        '''
        e_in = e
        B1_h = self.B1(h).unsqueeze(1)
        B2_h = self.B2(h).unsqueeze(2)
        # n_batch, n_nodes, n_features = B1_h.size()
        E12 = self.E12(e) #.reshape(n_batch, n_nodes, n_nodes, n_features)
        # e = torch.einsum('bik,bjk,bijk->bijk', B1_h, B2_h, E12)
        e = B1_h + B2_h + E12
        #e_out = e_out.reshape(n_batch, n_nodes * n_nodes, n_features)
        # e = self.batch_norm_e(e)
        e = e_in + F.relu(e)
        return e

    def forward_p(self, p, e, k_RW=None, mask=None, adj=None):
        '''
        Update positional encoding p
        '''
        p_in1 = p # for residual connection
    
        p = self.attention_p(p, e, k_RW=k_RW, mask=mask, adj=adj)  
        p = F.dropout(p, self.dropout, training=self.training)
        p = self.O_p(p)
        p = torch.tanh(p)
        if self.residual:
            p = p_in1 + p # residual connection

        return p

    def feed_forward_block(self, h):
        '''
        Add dense layers to the self-attention
        '''
        # # FFN for h
        h_in2 = h # for second residual connection
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection       
    
        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h.transpose(1,2)).transpose(1,2)
        return h

    def forward(self, h, p, e, k_RW=None, mask=None, adj=None):

        h_in1 = h # for first residual connection
        
        # [START] For calculation of h -----------------------------------------------------------------
        
        if self.concat_h_p:
            h = torch.cat((h, p), dim=-1)
        elif p is not None:
            h = h + p
        # multi-head attention out
        h = self.attention_h(h, e, k_RW=k_RW, mask=mask, adj=adj)
        
        if self.update_edge_features: 
            e = self.forward_edges(h_in1, e)
        # #Concat multi-head outputs
        # h = h_attn_out.view(-1, self.out_channels)
       
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h # residual connection
            
        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            # Apparently have to do this double transpose for 3D input 
            h = self.batch_norm1_h(h.transpose(1,2)).transpose(1,2)

        if self.feedforward:
            h = self.feed_forward_block(h)         
                
        if self.update_pos_enc:
            p = self.forward_p(p, e, k_RW=k_RW, mask=mask, adj=adj)
            # TODO: check if this is the right place, or if we should put it before the batch norm for ex

        return h, p, e
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y