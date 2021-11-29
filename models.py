import torch
import torch.nn as nn
from labml_helpers.module import Module
import os

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
        self.gat_layer1 = GraphAttentionV2Layer(in_features, n_hidden, n_heads,
                                                is_concat=False, dropout=dropout, share_weights=share_weights)

        self.gat_layer = GraphAttentionV2Layer(n_hidden, n_hidden, n_heads,
                                                is_concat=False, dropout=dropout, share_weights=share_weights)



        # # Final graph attention layer where we average the heads
        # self.gat_output = GraphAttentionV2Layer(n_hidden, n_classes, n_classes,
        #                                         is_concat=False, dropout=dropout, share_weights=share_weights)

        # self.linear1 = nn.Linear(n_hidden, n_hidden, bias=True)
        # self.linear2 = nn.Linear(n_hidden, n_classes, bias=True)

        self.activation = nn.ELU()

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=n_hidden+4, out_features= n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_classes),
        )

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor, mask):
        x = x.to(device)
        residual = torch.cat((x[:,:,1:3], x[:,:,6:8]), dim=2)
        # 1. Obtain node embeddings
        h = self.activation(self.gat_layer1(x, adj_mat,mask))
        # h = self.dropout(h)

        h = self.gat_layer(h, adj_mat,mask=None)
        h = self.activation(h)

        h = self.gat_layer(h, adj_mat,mask=None)

        # # 2. Readout layer
        # x_graph = torch.mean(x_node, 1).unsqueeze(1).repeat(1,x_node.size(1),1) # global mean pool # TODO check graph embed

        # # 3. Apply final projection
        # out = torch.cat((x_node, x_graph),dim=2)
        h = torch.cat((h, residual), dim=2)

        # Linear layer
        q = self.linear_layers(h)
        return q


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

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, mask):

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
        adj_mat_flat = torch.flatten(adj_mat.unsqueeze(-1).repeat(1,1,1,self.n_heads), start_dim=1).to(device)
        e_flat = e_flat.masked_fill(adj_mat_flat == 0, float('-inf'))
        e = e_flat.unflatten(1, (n_nodes, n_nodes,self.n_heads))

        e_ = e
        if mask is not None:
            mask = (1- mask).unsqueeze(2).unsqueeze(3).repeat(1, 1, e.size(2), e.size(3)).to(device)
            mask = torch.eq(mask, mask.permute(0,2,1,3))
            mask_value = torch.finfo(e.dtype).min
            e_ = e.masked_fill_((~mask), mask_value)

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
    model = GATv2(in_features=8, n_hidden=8, n_classes=1, n_heads=1, dropout=0.1, share_weights=False)
    x = torch.rand(2, 20, 8)
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
