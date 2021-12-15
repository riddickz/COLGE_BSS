import torch
import torch.nn as nn
from labml_helpers.module import Module
import os
import torch.nn.functional as f

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def normalize(A):
    A = A + torch.eye(A.size(1))
    d = torch.sum(A, dim=2)
    D = torch.diag_embed(torch.pow(d, -0.5)) # D = D^-1/2
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

    def __init__(self, in_features: int, n_hidden: int, n_classes: int, n_nodes: int, n_heads: int, dropout: float,
                 share_weights: bool = True, residual: bool = True):
        """
        * `in_features` is the number of features per node
        * `n_hidden` is the number of features in the first graph attention layer
        * `n_classes` is the number of classes
        * `n_heads` is the number of heads in the graph attention layers
        * `dropout` is the dropout probability
        * `share_weights` if set to True, the same matrix will be applied to the source and the target node of every edge
        """
        super().__init__()
        self.in_features = in_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.share_weights = share_weights
        self.residual = residual


        self.linear = nn.Linear(in_features=self.in_features, out_features= self.n_hidden, bias=True)

        self.gat_layer = GraphAttentionV2Layer(self.n_hidden, self.n_hidden, self.n_heads,
                                                is_concat=False, dropout=dropout, share_weights=self.share_weights,
                                                n_nodes=self.n_nodes)
        self.gat_layer2 = GraphAttentionV2Layer(2*self.n_hidden, self.n_hidden, self.n_heads,
                                                is_concat=False, dropout=dropout, share_weights=self.share_weights,
                                                n_nodes=self.n_nodes)

        self.linear2 = nn.Linear(in_features=2*self.n_hidden, out_features=2*self.n_hidden, bias=True)

        self.dropout = nn.Dropout(dropout)

        # self.layer_norm1_h = nn.LayerNorm(self.n_hidden)
        # self.layer_norm2_h = nn.LayerNorm(self.n_hidden*2)
        # self.batch_norm1_h = nn.BatchNorm1d(self.n_node)


        # self.act_elu = nn.ELU()
        self.act_tahn = nn.Tanh()

        self.MLP = nn.Sequential(
            nn.Linear(in_features=3*self.n_hidden, out_features= 2*self.n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2*self.n_hidden, out_features=self.n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=self.n_hidden, out_features=n_classes,bias=True),
        )

        self.softmax = nn.Softmax(dim=n_classes)


    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor, mask=None):
        # x[:,:,1] =x[:,:,1]/20
        # x[:,:,2] =x[:,:,2]/10
        # x[:,:,5] =x[:,:,5]/35
        # x[:,:,6] =x[:,:,6]/35
        # x[:,:,7] =x[:,:,7]/(10/30)
        # x = torch.cat((x[:,:,1:3], x[:,:,5:]), dim=2)
        # h_in = torch.cat((x[:,:,1:3], x[:,:,6:]), dim=2)  # for first residual connection
        # residual = torch.cat((x[:,:,1:3], x[:,:,6:8]), dim=2)

        h_in = self.linear(x)
        # 1.[START] GAT -----------------------------------------------------------------
        out1 = self.gat_layer(h_in, adj_mat)
        out1 = self.act_tahn(out1)
        out1 = torch.cat((out1, h_in), dim=2)
        # out1 = self.batch_norm1_h(out1)
        out1 = self.linear2(out1)
        out1 = self.act_tahn(out1)

        out2 = self.gat_layer2(out1, adj_mat)
        out2 = self.act_tahn(out2)
        out2 = torch.cat((out2, out1), dim=2)
        # out2 = self.batch_norm1_h(out2)

        # 1.[END] GAT -----------------------------------------------------------------

        # 2.[START] MLP -----------------------------------------------------------------
        q = self.MLP(out2)
        q = self.softmax(q)
        # 2.[END] MLP -----------------------------------------------------------------

        return q


class GraphAttentionV2Layer(Module):

    def __init__(self, in_features: int, out_features: int, n_heads: int,
                 is_concat: bool = True,
                 dropout: float = 0.6,
                 leaky_relu_negative_slope: float = 0.2,
                 share_weights: bool = False,
                 n_nodes:int = 10):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        # self.adaptive_edge_PE = adaptive_edge_PE
        self.n_nodes = n_nodes
        self.is_concat = is_concat
        self.dropout = dropout
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.share_weights = share_weights


        # Calculate the number of dimensions per head
        if self.is_concat:
            assert self.out_features  % self.n_heads == 0
            self.n_hidden = self.out_features  // self.n_heads
        else:
            self.n_hidden = self.out_features

        # Linear layer for initial source transformation;
        self.linear_l = nn.Linear(self.in_features, self.n_hidden * self.n_heads, bias=False)

        if self.share_weights:
            self.linear_r = self.linear_l
        else:
            self.linear_r = nn.Linear(self.in_features, self.n_hidden * self.n_heads, bias=False)

        # Linear layer to compute attention score e_ij
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)

        # The activation for attention score e_ij
        self.activation = nn.Tanh() #nn.LeakyReLU(negative_slope=self.leaky_relu_negative_slope)
        self.lin_n_node = nn.Linear(self.n_nodes**2, self.n_nodes**2, bias=False)
        # Softmax to compute attention alpha_ij
        self.softmax = nn.Softmax(dim=2)

        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(self.dropout )

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, mask=None):

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

        # fill node self edge and normalize
        d = torch.min(adj_mat.masked_fill(adj_mat == 0, float('inf')), dim=2)[0]/2
        adj = adj_mat + torch.diag_embed(d)
        adj_norm = f.normalize(adj.float(),p=2,dim=2)

        # flatten adj and invert dist
        adj_flat = torch.flatten(adj_norm.unsqueeze(-1).repeat(1,1,1,self.n_heads), start_dim=1).to(device)
        # e_flat = e_flat.masked_fill(adj_mat_flat == 0, float('-inf'))
        adj_flat_inv = torch.pow(adj_flat, -1)

        # attention weighed by edge weight
        mask_no_edge = torch.ones_like(adj_flat_inv)
        mask_no_edge[adj_flat_inv == float('inf')] = float('0')

        adj_flat_inv[adj_flat_inv == float('inf')] = float('0')
        edge_att = self.activation(self.lin_n_node(adj_flat_inv)) * mask_no_edge

        e_flat = e_flat * edge_att
        e_flat = e_flat.masked_fill(e_flat == 0, float('-10000'))
        e = e_flat.unflatten(1, (n_nodes, n_nodes,self.n_heads))
        #
        # if mask is not None:
        #     mask1 = (mask).unsqueeze(2).unsqueeze(3).repeat(1, 1, e.size(2), e.size(3)).to(device)
        #     # mask1 = torch.eq(mask1, mask1.permute(0,2,1,3))
        #     mask2 = 1 - mask1 *mask1.permute(0,2,1,3)
        #     e_ = e.masked_fill_((mask2), float('-10000'))
        #
        # else:
        #     e_ = e

        e_ = e

        # We then normalize attention scores (or coefficients)
        a = self.softmax(e_)

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
    dim_in = 8+4
    model = GATv2(in_features=dim_in, n_hidden=dim_in*2, n_classes=1, n_heads=1, dropout=0.1, share_weights=False)
    x = torch.rand(2, 20, 14)
    a = torch.randint(2, (20, 20))
    a = (a + a.t()).clamp(max=1)
    a = a.unsqueeze(0).repeat(2, 1, 1)
    out = model(x,a)
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
    # test_GCN_naive()
    test_GATv2()
