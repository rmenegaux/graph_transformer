# -*- coding: utf-8 -*-
import os
import torch
from gckn.gckn_fast import gckn_fast_cpu
if torch.cuda.is_available():
    try:
        from gckn.gckn_fast import gckn_fast_cuda
    except:
        pass


def path_conv_forward(path_indices, features):
    if features.is_cuda:
        output = gckn_fast_cuda.path_conv_forward(path_indices, features)
    else:
        output = gckn_fast_cpu.path_conv_forward(path_indices, features)
    return output

def path_conv_backward(grad_input, grad_output, path_indices):
    if grad_output.is_cuda:
        gckn_fast_cuda.path_conv_backward(grad_input, grad_output, path_indices)
    else:
        gckn_fast_cpu.path_conv_backward(grad_input, grad_output, path_indices)

class PathConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, path_indices, features):
        if features.is_cuda:
            output = gckn_fast_cuda.path_conv_forward(path_indices, features)
        else:
            output = gckn_fast_cpu.path_conv_forward(path_indices, features)
        ctx.save_for_backward(path_indices)
        ctx.size = features.size()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_zeros(ctx.size)
        if grad_output.is_cuda:
            gckn_fast_cuda.path_conv_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables)
        else:
            gckn_fast_cpu.path_conv_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables)
        return None, grad_input

def path_conv(path_indices, features, edges_info):
    import torch.nn.functional as F
    path_indices_margot = path_indices * features.shape[1] + torch.arange(path_indices.shape[1])
    features_margot = features.permute(2,0,1).view(features.shape[2],-1).t()
    
    # output = F.embedding(path_indices, features.view(features.shape[0], -1)).view(
    #            path_indices.shape[0], path_indices.shape[1], features.shape[1], features.shape[2])

    # output: all_paths x path_size x path_size x hidden_size
    # -> all_paths x hidden_size x path_size
    
    
    # output = output.diagonal(dim1=1, dim2=2) # demander à dexiong
    
    output_margot = F.embedding(path_indices_margot, features_margot).permute(0,2,1)


    #output = output.mean(dim=-1)
    output_margot = output_margot.mean(dim=-1)

    output_edges=None
    output_margot_edges=None
    if edges_info is not None:
        path_indices_margot_edges = edges_info['paths_edges'].type(torch.LongTensor) * \
            edges_info['edge_features'].shape[1] + torch.arange(edges_info['paths_edges'].shape[1])
        features_margot_edges = edges_info['edge_features'].permute(2,0,1).view(edges_info['edge_features'].shape[2],-1).t()
        output_margot_edges = F.embedding(path_indices_margot_edges, features_margot_edges).permute(0,2,1)
        output_margot_edges = output_margot_edges.mean(dim=-1)

        # output_edges_2d = F.embedding(edges_info['paths_edges'].type(torch.LongTensor), edges_info['edge_features_todel'].view(edges_info['edge_features'].shape[0], -1))
        # output_edges = output_edges_2d.permute(0,2,1).type(torch.FloatTensor).mean(dim=-1)
    
    # return output, output_edges
    return output_margot, output_margot_edges

def test(cuda=False):
    torch.manual_seed(1234)

    path_size = 5
    n_nodes = 100
    hidden_size = 32
    n_paths = 100000
    x = torch.randn(n_nodes, path_size, hidden_size)
    path_indices = torch.randint(0, n_nodes, (n_paths, path_size))
    
    if cuda:
        x = x.cuda()
        path_indices = path_indices.cuda()

    x.requires_grad_()
    print('start')
    out = PathConv.apply(path_indices, x)
    out1 = out.data
    out = out.mean()
    out.backward()
    grad1 = x.grad.data
    x.grad = None

    out = path_conv(path_indices, x)
    out2 = out.data
    out = out.mean()
    out.backward()
    grad2 = x.grad.data
    # print(out1)
    # print(out2)
    print(torch.max(torch.abs(out1 - out2)))
    print(torch.max(torch.abs(grad1 - grad2)))

    import time
    forward = 0
    backward = 0
    n_iter = 10
    for _ in range(n_iter):
        start = time.time()
        out = PathConv.apply(path_indices, x)
        if cuda:
            torch.cuda.synchronize()
        forward += time.time() - start

        out = out.mean()
        start = time.time()
        out.backward()
        if cuda:
            torch.cuda.synchronize()
        backward += time.time() - start

    print('Mine Forward: {:.3f} ms | Backward {:.3f} ms'.format(forward * 1e3/n_iter, backward * 1e3/n_iter))

    import time
    forward = 0
    backward = 0
    n_iter = 10
    for _ in range(n_iter):
        start = time.time()
        out = path_conv(path_indices, x)
        if cuda:
            torch.cuda.synchronize()
        forward += time.time() - start

        out = out.mean()
        start = time.time()
        out.backward()
        if cuda:
            torch.cuda.synchronize()
        backward += time.time() - start

    print('Pytorch Forward: {:.3f} ms | Backward {:.3f} ms'.format(forward * 1e3/n_iter, backward * 1e3/n_iter))


if __name__ == "__main__":
    test(cuda=False)
