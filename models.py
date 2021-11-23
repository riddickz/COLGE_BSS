import torch
import torch.nn.functional as F
import random
#from gensim.models import Word2Vec
import networkx as nx
import numpy as np
import torch
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric.utils import from_scipy_sparse_matrix
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GCN2Conv,GCNConv
from torch.nn import Linear
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class GCN_QN_1(torch.nn.Module):
    def __init__(self,reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T):

        super(GCN_QN_1, self).__init__()
        self.reg_hidden = reg_hidden
        self.embed_dim = embed_dim
        self.T = T
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling

        self.ft = 6
        self.mu_1 = torch.nn.Parameter(torch.Tensor(self.ft, embed_dim))
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)

        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim, True)
        torch.nn.init.normal_(self.mu_2.weight, mean=0, std=0.01)

        self.mu_3 = torch.nn.Linear(self.ft, 1, True)

        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)
        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            post_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)


        self.q_1 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)
        self.q = torch.nn.Linear(2 * embed_dim, 1,bias=True)
        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = torch.nn.Linear(self.reg_hidden, 1)
        else:
            self.q = torch.nn.Linear(2 * embed_dim, 1)
        torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)

    def forward(self, xv, adj):
        if len(xv.size()) <3:
            xv = xv.permute(1, 0).unsqueeze(0) #torch.Size([1, 10, 6])
            adj = adj.unsqueeze(0).float() # torch.Size([1, 10, 10])
        else:
            xv = torch.permute(xv, (0,2,1))

        minibatch_size = xv.shape[0]
        nbr_node = xv.shape[1]

        diag = torch.ones(nbr_node)
        I = torch.diag(diag).expand(minibatch_size,nbr_node,nbr_node)
        adj_=adj+I

        D = torch.sum(adj,dim=1)
        zero_selec = np.where(D.detach().numpy() == 0)
        D[zero_selec[0], zero_selec[1]] = 0.01
        d = []
        for vec in D:
            #d.append(torch.diag(torch.rsqrt(vec)))
            d.append(torch.diag(vec))
        d=torch.stack(d)

        #res = torch.zeros(minibatch_size,nbr_node,nbr_node)
        #D_=res.as_strided(res.size(), [res.stride(0), res.size(2) + 1]).copy_(D)

        #gv=torch.matmul(torch.matmul(d,adj_),d)
        gv=torch.matmul(torch.inverse(d),adj_)

        for t in range(self.T):
            if t == 0:
                #mu = self.mu_1(xv).clamp(0)
                mu = torch.matmul(xv, self.mu_1).clamp(0)
                #mu.transpose_(1,2)
                #mu_2 = self.mu_2(torch.matmul(adj, mu_init))
                #mu = torch.add(mu_1, mu_2).clamp(0)

            else:
                #mu_1 = self.mu_1(xv)
                mu_1 = torch.matmul(xv, self.mu_1).clamp(0)
                #mu_1.transpose_(1,2)
                # before pooling:
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0)

                mu_pool = torch.matmul(gv, mu)

                for i in range(self.len_post_pooling):

                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)



                mu_2 = self.mu_2(mu_pool)
                mu = torch.add(mu_1, mu_2).clamp(0)

        q_1 = self.q_1(torch.matmul(xv.transpose(1,2),mu))
        q_1_ = self.mu_3(q_1.transpose(1,2)).transpose(1,2).expand(minibatch_size,nbr_node,self.embed_dim)
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1_, q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            q = self.q(q_reg)
        else:
            q_=q_.clamp(0)
            q = self.q(q_)
        return q

class GCN_Naive(torch.nn.Module):
    def __init__(self,input_channels,output_channels,hidden_channels):
        super(GCN_Naive, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels, normalize=True)
        self.conv2 = GCNConv(hidden_channels, output_channels, normalize=True)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class GCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,edge_weight,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim, edge_weight)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x

class GCN2_Net(torch.nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(GCN2_Net, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(input_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, output_channels))

        self.convs = torch.nn.ModuleList()

        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights,normalize=True))

        self.dropout = dropout


    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, edge_index)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x


class GATLayer(nn.Module):

    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out))  # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False)  # Returns indices where the adjacency matrix is not 0 => edges
        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ],
            dim=-1)  # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats