import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

BN = True

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


class DiscreteEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=10, max_num_values=500): #10
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels) 
                    for i in range(max_num_features)])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()
            
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = 0
        for i in range(x.size(1)):
            out += self.embeddings[i](x[:, i])
        return out

class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=BN, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i==0 else n_hid, 
                                     n_hid if i<nlayer-1 else nout, 
                                     bias=True if (i==nlayer-1 and not with_final_activation and bias) # TODO: revise later
                                        or (not with_norm) else False) # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i<nlayer-1 else nout) if with_norm else Identity()
                                     for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin==nout) ## TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)  

        # if self.residual:
        #     x = x + previous_x  
        return x 

class VNUpdate(nn.Module):
    def __init__(self, dim, with_norm=BN):
        """
        Intermediate update layer for the virtual node
        :param dim: Dimension of the latent node embeddings
        :param config: Python Dict with the configuration of the CRaWl network
        """
        super().__init__()
        self.mlp = MLP(dim, dim, with_norm=with_norm, with_final_activation=True, bias=not BN)

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, vn, x, batch):
        G = global_add_pool(x, batch)
        if vn is not None:
            G += vn
        vn = self.mlp(G)
        x += vn[batch]
        return vn, x

#TODO: add general aggregator layer
import torch
from torch_scatter import scatter
from torch_geometric.utils import degree
class Aggregator(nn.Module):
    def __init__(self, nin, nout, aggregators=['mean', 'min', 'max', 'std']):
        super().__init__()
        self.aggregators = aggregators
        self.deg_embedder = nn.Embedding(100, nin) 
        self.output_encoder = MLP((len(aggregators) + 1) * nin, nout, 1, True)

        # TODO: add key-based attention aggregation 
    def reset_parameters(self):
        self.deg_embedder.reset_parameters()
        self.output_encoder.reset_parameters()

    def forward(self, inputs, index, dim_size=None):
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(F.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')  
            outs.append(out)

        outs.append(self.deg_embedder(degree(index, dim_size, dtype=index.dtype)))
        out = torch.cat(outs, dim=-1)
        return self.output_encoder(out)


