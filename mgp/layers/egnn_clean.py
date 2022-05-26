from torch import nn
import torch

from torch_scatter.composite import scatter_softmax
from torch.autograd import grad
from torch_scatter import scatter_add

from torch.nn import LayerNorm
import math

class E_GCL(nn.Module):

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False, use_layer_norm=True):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.use_layer_norm = use_layer_norm
        edge_coords_nf = 1

        if use_layer_norm:
            self.node_ln = LayerNorm(input_nf)
            self.edge_ln = LayerNorm(hidden_nf)
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        # radial = torch.norm(coord_diff, dim=1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            # norm = radial.detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, edge_mask=None, update_coords=True):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        h0 = h
        if self.use_layer_norm:
            h = self.node_ln(h)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        if self.use_layer_norm:
            edge_feat = self.edge_ln(edge_feat)
        if edge_mask is not None:
            edge_feat = edge_feat * edge_mask
        if update_coords:
            coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if self.residual:
            h = h0 + h
        return h, coord, edge_attr


class EGNN_last(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False, use_layer_norm=False):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN_last, self).__init__()
        self.hidden_nf = hidden_nf
        # self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        if use_layer_norm:
            self.final_ln = LayerNorm(self.hidden_nf)
        self.use_layer_norm = use_layer_norm
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh, use_layer_norm=use_layer_norm))


    def forward(self, h, x, edges, edge_attr, edge_mask=None):
        h = self.embedding_in(h)
        for i in range(0, self.n_layers - 1):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, edge_mask=edge_mask, update_coords=False)
        h, x, _ = self._modules["gcl_%d" % (self.n_layers - 1)](h, edges, x, edge_attr=edge_attr, edge_mask=edge_mask, update_coords=True)
        if self.use_layer_norm:
            h = self.final_ln(h)
        # h = self.embedding_out(h)
        return h, x

class EGNN_finetune_last(EGNN_last):
    def __init__(self, in_node_nf, hidden_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False, use_layer_norm=False):
        EGNN_last.__init__(self, in_node_nf, hidden_nf, in_edge_nf, act_fn, n_layers, residual, attention, normalize, tanh, use_layer_norm)
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))

    def forward(self, h, x, edges, edge_attr, n_nodes, edge_mask=None, node_mask=None, adapter=None):
        x_ = x.clone()
        h, x = EGNN_last.forward(self, h, x, edges, edge_attr, edge_mask=edge_mask)
        h = self.node_dec(h)
        if node_mask is not None:
            h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)

class EGNN_md_last(EGNN_last):
    def __init__(self, in_node_nf, hidden_nf, in_edge_nf=0, act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False, mean=None, std=None):
        EGNN_last.__init__(self, in_node_nf, hidden_nf, in_edge_nf, act_fn, n_layers, residual, attention, normalize, tanh)

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))

        mean = torch.scalar_tensor(0 if mean is None else mean)
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1 if std is None else std)
        self.register_buffer("std", std)

        self.reset_parameters()

    def reset_parameters(self):
        for n,p in self.named_parameters():
            if 'weight' in n:
                nn.init.kaiming_uniform_(p, a=0)     
         

    def forward(self, h, x, edges, edge_attr, batch, edge_mask=None, md_type='predict'):
        x_ = x.clone()
        if md_type == 'gradient':
            x_.requires_grad_(True)
        h, x__ = EGNN_last.forward(self, h, x_, edges, edge_attr, edge_mask=edge_mask)
        h = self.graph_dec(h)
        pred = scatter_add(h, batch, dim=0)
        pred = pred * self.std + self.mean
        if md_type == 'predict':
            dy = x__ - x_
        elif md_type == 'gradient':
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(pred)]
            dy = - grad(
                [pred],
                [x_],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            dy = None
        return pred.squeeze(1), dy



def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def get_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)

    edges = [rows, cols]
    return edges


def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr



