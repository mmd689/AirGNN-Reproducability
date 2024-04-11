import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import APPNP, GCNConv, GCN2Conv, GATConv
from torch_geometric.nn import Sequential


class Model(torch.nn.Module):
    def __init__(self, dataset, args):
        super(Model, self).__init__()
        model = args.model
        if model == 'AirGNN':
            self.model = AirGNN(dataset, args)
        elif model == 'APPNP':
            self.model = APPNP_(dataset, args)
        elif model == 'MLP':
            self.model = MLP(dataset, args)
        elif model == 'GCN':
            self.model = GCN(dataset, args)
        elif model == 'GCNII':
            self.model = GCNII(dataset, args)
        elif model == 'GAT':
            self.model = GAT(dataset, args)
        else:
            raise ValueError('Not Implemented')

    def reset_parameters(self):
        self.model.reset_parameters()

    def forward(self, data):
        return self.model(data)


class MLP(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MLP, self).__init__()
        self.dropout = args.dropout
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, y = data.x, data.adj_t, data.y
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class APPNP_(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_, self).__init__()
        self.dropout = args.dropout
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop = APPNP(K=args.K, alpha=args.alpha)
        print(self.prop)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index, y = data.x, data.adj_t, data.y
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN, self).__init__()
        self.dropout = args.dropout
        self.gcn1 = GCNConv(dataset.num_features, args.hidden)
        self.gcn2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()

    def forward(self, data):
        x, edge_index, y = data.x, data.adj_t, data.y
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.gcn2(x, edge_index))
        return F.log_softmax(x, dim=1)


class AirGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(AirGNN, self).__init__()
        self.dropout = args.dropout
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop = AdaptiveMessagePassing(K=args.K, alpha=args.alpha, mode=args.model, args=args)
        print(self.prop)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, edge_index, y = data.x, data.adj_t, data.y
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x, score = self.prop(x, edge_index)
        return F.log_softmax(x, dim=1), score


class GCNII(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCNII, self).__init__()
        self.dropout = args.dropout
        self.nlayers = args.layers
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.gcn2 = []
        for i in range(self.nlayers):
            self.gcn2.append(GCN2Conv(args.hidden, args.alpha, theta=args.lambda_amp, layer=i+1))
        self.lin2 = Linear(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for gcn in self.gcn2:
            gcn.reset_parameters()

    def forward(self, data):
        x, edge_index, y = data.x, data.adj_t, data.y
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        h = []
        h.append(x)
        for i in range(self.nlayers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu((self.gcn2[i])(x, h[0], edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT, self).__init__()
        self.dropout = args.dropout
        self.gat1 = GATConv(dataset.num_features, args.hidden, heads=8)
        self.gat2 = GATConv(args.hidden * 8, dataset.num_classes)

    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()

    def forward(self, data):
        x, edge_index, y = data.x, data.adj_t, data.y
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)


class AdaptiveMessagePassing(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 K: int,
                 alpha: float,
                 dropout: float = 0.,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 mode: str = None,
                 node_num: int = None,
                 args=None,
                 **kwargs):

        super(AdaptiveMessagePassing, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.alpha = alpha
        self.mode = mode
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self.node_num = node_num
        self.args = args
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                raise ValueError('Only support SparseTensor now')

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        add_self_loops=self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if self.K <= 0:
            return x
        hh = x

        x, score = self.amp_forward(x=x, hh=hh, edge_index=edge_index, K=self.K)

        return x, score

    def amp_forward(self, x, hh, K, edge_index):
        lambda_amp = self.args.lambda_amp
        gamma = 1 / (2 * (1 - lambda_amp))  ## or simply gamma = 1

        for k in range(K):
            y = x - gamma * 2 * (1 - lambda_amp) * self.compute_LX(x=x, edge_index=edge_index)  # Equation (9)
            oo, score = self.proximal_L21(x=y - hh, lambda_=gamma * lambda_amp)
            x = hh + oo # Equation (11) and (12)
        return x, score

    def proximal_L21(self, x: Tensor, lambda_):
        row_norm = torch.norm(x, p=2, dim=1)
        score = torch.clamp(row_norm - lambda_, min=0)
        index = torch.where(row_norm > 0)             #  Deal with the case when the row_norm is 0
        score[index] = score[index] / row_norm[index] # score is the adaptive score in Equation (14)
        return score.unsqueeze(1) * x, score

    def compute_LX(self, x, edge_index, edge_weight=None):
        x = x - self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={}, mode={}, dropout={}, lambda_amp={})'.format(self.__class__.__name__, self.K,
                                                               self.alpha, self.mode, self.dropout,
                                                               self.args.lambda_amp)

