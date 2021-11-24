import torch
import torch.nn.functional as F
import torch.nn as nn
from labml_helpers.module import Module
import torch_geometric as tg
from torch_geometric.nn import GCN2Conv, GCNConv
from torch.nn import Linear


class GCN_PYG(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim, edge_weight,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN_PYG, self).__init__()
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
        for i in range(self.layer_num - 2):
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
                         shared_weights, normalize=True))

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


def normalize(A):
    A = A + torch.eye(A.size(1))
    d = torch.sum(A, dim=2)
    # D = D^-1/2
    D = torch.diag_embed(torch.pow(d, -0.5))
    return D.bmm(A).bmm(D)


class GCN_Naive(nn.Module):
    def __init__(self, c_in, c_out, c_hidden):
        super(GCN_Naive, self).__init__()
        self.fc1 = nn.Linear(c_in, c_hidden, bias=False)
        self.fc2 = nn.Linear(c_hidden, c_hidden, bias=False)
        self.fc3 = nn.Linear(c_hidden, c_out, bias=False)
        self.leakyReLU = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        num_neighbours = A.sum(dim=-1, keepdims=True)
        A = normalize(A)
        H = self.leakyReLU(self.fc1(A.bmm(X)))
        H = self.leakyReLU(self.fc2(A.bmm(H)))
        H = H / num_neighbours
        H = self.fc3(A.bmm(H))
        return H


class GATv2(Module):
    """
    ## Graph Attention Network v2 (GATv2)
    """

    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_heads: int, dropout: float,
                 share_weights: bool = True):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        * `share_weights` if set to True, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()

        # First graph attention layer where we concatenate the heads
        self.layer1 = GraphAttentionV2Layer(in_features, n_hidden, n_heads,
                                            is_concat=True, dropout=dropout, share_weights=share_weights)
        # Activation function after first graph attention layer
        self.activation = nn.ELU()
        # Final graph attention layer where we average the heads
        self.output = GraphAttentionV2Layer(n_hidden, n_classes, 1,
                                            is_concat=False, dropout=dropout, share_weights=share_weights)
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        # Apply dropout to the input
        x = self.dropout(x)
        # First graph attention layer
        x = self.layer1(x, adj_mat)
        # Activation function
        x = self.activation(x)
        # Dropout
        x = self.dropout(x)
        # Output layer
        x = self.output(x, adj_mat)
        # x = self.activation(x)
        return x


class GraphAttentionV2Layer(Module):

    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False):

        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights

        # Calculate the number of dimensions per head
        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        # Linear layer for initial source transformation;
        self.linear_l = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        if share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)

        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)

        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=2)

        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):

        # Number of nodes
        batch_size = h.shape[0]
        n_nodes = h.shape[1]

        # The initial transformations, for each head.
        g_l = self.linear_l(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        g_r = self.linear_r(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)

        # #### Calculate attention score

        # where each node embedding is repeated `n_nodes` times.
        g_l_repeat = g_l.repeat(1, n_nodes, 1, 1)
        g_r_repeat_interleave = g_r.repeat_interleave(n_nodes, dim=1)

        g_sum = g_l_repeat + g_r_repeat_interleave
        g_sum = g_sum.view(batch_size, n_nodes, n_nodes, self.n_heads, self.n_hidden)

        # `e` is of shape `[batch, n_nodes, n_nodes, n_heads, 1]`
        e = self.attn(self.activation(g_sum))
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        # Mask $e_ij$ based on adjacency matrix.
        e_flat = torch.flatten(e, start_dim=1)
        adj_mat_flat = torch.flatten(adj_mat, start_dim=1)
        e_flat = e_flat.masked_fill(adj_mat_flat == 0, float('-inf'))
        e = e_flat.unflatten(1, (n_nodes, n_nodes)).unsqueeze(3)

        # We then normalize attention scores (or coefficients)
        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        # Calculate final output for each head
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g_r)

        if self.is_concat:
            # Concatenate the heads
            return attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        else:
            # Take the mean of the heads
            return attn_res.mean(dim=2)


def test_GATv2():
    model = GATv2(in_features=8, n_hidden=8, n_classes=1, n_heads=1, dropout=0.1, share_weights=False)
    x = torch.rand(10, 20, 8)
    a = torch.randint(2, (20, 20))
    a = (a + a.t()).clamp(max=1)
    a = a.unsqueeze(0).repeat(10, 1, 1)
    out = model(x, a)
    print(out.shape)


def test_GCN_naive():
    model = GCN_Naive(c_in=8, c_out=1, c_hidden=8)
    x = torch.rand(10, 20, 8)
    a = torch.randint(2, (20, 20))
    a = (a + a.t()).clamp(max=1)
    a = a.unsqueeze(0).repeat(10, 1, 1)
    out = model(x, a)
    print(out.shape)


if __name__ == "__main__":
    test_GCN_naive()
    test_GATv2()
