import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import core.model_utils.pyg_gnn_wrapper as gnn_wrapper 
from core.model_utils.elements import MLP, DiscreteEncoder, Identity, Aggregator, VNAgg
from core.model_utils.ppgn import PPGN
from torch_geometric.nn.inits import reset

BN = True

class GNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, nin, nout, nlayer, gnn_type, dropout=0, bn=BN, bias=True, res=True):
        super().__init__()
        # TODO: consider remove input and output encoder for nhead=1?
        # self.input_encoder = MLP(nin, nin, nlayer=2, with_final_activation=True) #if nin!=nout else nn.Identity()
        self.convs = nn.ModuleList([getattr(gnn_wrapper, gnn_type)(nin, nin, bias=not bn) for _ in range(nlayer)]) # set bias=False for BN
        self.norms = nn.ModuleList([nn.BatchNorm1d(nin) if bn else Identity() for _ in range(nlayer)])
        self.output_encoder = MLP(nin, nout, nlayer=1, with_final_activation=False, bias=bias) if nin!=nout else Identity() 
        # !!!!!! attention! change output encoder layer from 2 to 1
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):
        # self.input_encoder.reset_parameters()
        self.output_encoder.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
     
    def forward(self, x, edge_index, edge_attr, batch):
        # x = self.input_encoder(x)
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x += previous_x 
                previous_x = x

        x = self.output_encoder(x)
        return x

class SubgraphGNNKernel(nn.Module):
    # Use GNN to encode subgraphs
    # gnn_types: a list of GNN types
    def __init__(self, nin, nout, nlayer, gnn_types, dropout=0, 
                       hop_dim=16, 
                       bias=True, 
                       res=True,
                       pooling='mean',
                       embs=(0,1,2),
                       embs_combine_mode='add',
                       subsampling=False, 
                       online=True):
        super().__init__()
        assert max(embs) <= 2 and min(embs) >= 0
        assert embs_combine_mode in ['add', 'concat']

        l = 1
        use_hops = hop_dim > 0
        nhid = nout // len(gnn_types)
        self.hop_embedder = nn.Embedding(20, hop_dim)
        # self.gnns = nn.ModuleList([GNN(nin+hop_dim if use_hops else nin, nhid, nlayer, gnn_type) for gnn_type in gnn_types])
        self.gnns = nn.ModuleList()
        for gnn_type in gnn_types:
            if gnn_type == 'PPGN':
                gnn = PPGN(nin+hop_dim if use_hops else nin, nhid, nlayer)
            else:
                gnn = GNN(nin+hop_dim if use_hops else nin, nhid, nlayer, gnn_type, dropout=dropout, res=res)
            self.gnns.append(gnn)
        self.centroid_transform = MLP(nout, nout, nlayer=l, with_final_activation=True)
        self.subgraph_transform = MLP(nout, nout, nlayer=l, with_final_activation=True)
        self.context_transform =  MLP(nout, nout, nlayer=l, with_final_activation=True)

        # subsampling = True
        self.out_encoder = MLP(nout if subsampling or embs_combine_mode=='add' else nout*len(embs), nout, nlayer=l, 
                               with_final_activation=False, bias=bias, with_norm=True)

        self.use_hops = use_hops
        self.gate_mapper_subgraph = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_context = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_centroid = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid()) # add this one to scale centroid embedding
        self.subsampling = subsampling

        # dropout = 0
        self.dropout = dropout
        self.online = online
        self.pooling = pooling
        self.embs = embs
        self.embs_combine_mode = embs_combine_mode

    def reset_parameters(self):
        self.hop_embedder.reset_parameters()
        for gnn in self.gnns:
            gnn.reset_parameters()
        self.centroid_transform.reset_parameters()
        self.subgraph_transform.reset_parameters()
        self.context_transform.reset_parameters()
        self.out_encoder.reset_parameters()
        reset(self.gate_mapper_context)
        reset(self.gate_mapper_subgraph)
        reset(self.gate_mapper_centroid)

    def forward(self, data):
        # prepare x, edge_index, edge_attr for the combined subgraphs
        combined_subgraphs_x = data.x[data.subgraphs_nodes_mapper] # lift up the embeddings, positional encoding
        combined_subgraphs_edge_index = data.combined_subgraphs
        combined_subgraphs_edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        combined_subgraphs_batch = data.subgraphs_batch
        if self.use_hops:
            hop_emb = self.hop_embedder(data.hop_indicator+1)  # +1 to make -1(not calculated part: too far away) as 0. 
            combined_subgraphs_x = torch.cat([combined_subgraphs_x, hop_emb], dim=-1)

        combined_subgraphs_x = torch.cat([gnn(combined_subgraphs_x, combined_subgraphs_edge_index, combined_subgraphs_edge_attr, combined_subgraphs_batch)
                                          for gnn in self.gnns], dim=-1) # -1 dim = nout

        if self.subsampling:
            # can only use context_x when use subsampling without pooling. 
            if self.use_hops: # test this
                combined_subgraphs_x = combined_subgraphs_x * self.gate_mapper_context(hop_emb)
            x = scatter(combined_subgraphs_x , data.subgraphs_nodes_mapper, dim=0, reduce=self.pooling)

            if self.pooling == 'add' and self.training and self.online: # only scale it up during training part
                x = x * data.subsampling_scale.unsqueeze(-1)
        else:
            # get subgraph level embeddings
            if len(self.embs) == 1:
                # TODO: test whether we should directly output or pass to a transform layer.
                if self.embs[0] == 0:
                    centroid_x = combined_subgraphs_x[(data.subgraphs_nodes_mapper == combined_subgraphs_batch)]
                    x = centroid_x
                if self.embs[0] == 1:
                    subgraph_x = combined_subgraphs_x * self.gate_mapper_subgraph(hop_emb)
                    subgraph_x = scatter(subgraph_x, combined_subgraphs_batch, dim=0, reduce=self.pooling)
                    x = subgraph_x
                if self.embs[0] == 2:
                    context_x = combined_subgraphs_x * self.gate_mapper_context(hop_emb)
                    context_x = scatter(context_x, data.subgraphs_nodes_mapper, dim=0, reduce=self.pooling)
                    x = context_x
            else:
                # centroid_x = self.centroid_transform(F.dropout(combined_subgraphs_x[(data.subgraphs_nodes_mapper == combined_subgraphs_batch)], self.dropout, training=self.training))
                # TODO: how many MLP layers we should use??
                centroid_x = combined_subgraphs_x[(data.subgraphs_nodes_mapper == combined_subgraphs_batch)]
                subgraph_x = self.subgraph_transform(F.dropout(combined_subgraphs_x, self.dropout, training=self.training))
                context_x = self.context_transform(F.dropout(combined_subgraphs_x, self.dropout, training=self.training))
                if self.use_hops:
                    centroid_x = centroid_x * self.gate_mapper_centroid(hop_emb[(data.subgraphs_nodes_mapper == combined_subgraphs_batch)]) 
                    subgraph_x = subgraph_x * self.gate_mapper_subgraph(hop_emb)
                    context_x = context_x * self.gate_mapper_context(hop_emb)
                subgraph_x = scatter(subgraph_x, combined_subgraphs_batch, dim=0, reduce=self.pooling)
                context_x = scatter(context_x, data.subgraphs_nodes_mapper, dim=0, reduce=self.pooling)
                x = [centroid_x, subgraph_x, context_x]
                x = [x[i] for i in self.embs]
                if self.embs_combine_mode == 'add':
                    x = sum(x)
                else:
                    x = torch.cat(x, dim=-1)
        
        # last part is only essential for embs_combine_mode = 'concat', can be ignored when overfitting
        x = self.out_encoder(F.dropout(x, self.dropout, training=self.training)) 
        return x


class GNNAsKernel(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer_outer, nlayer_inner, gnn_types, 
                        dropout=0, 
                        hop_dim=0, 
                        node_embedding=False, 
                        use_normal_gnn=False, 
                        bn=BN, 
                        vn=False, 
                        res=True, 
                        pooling='mean',
                        embs=(0,1,2),
                        embs_combine_mode='add',
                        subsampling=False, 
                        online=True):
        super().__init__()
        # nfeat_in is None: discrete input features
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, nhid, 1)
        # layers
        # edge_emd_dim = nhid if nlayer_inner == 0 else nhid // len(gnn_types)
        edge_emd_dim = nhid if nlayer_inner == 0 else nhid + hop_dim 
        self.edge_encoders = nn.ModuleList([DiscreteEncoder(edge_emd_dim) if nfeat_edge is None else MLP(nfeat_edge, edge_emd_dim, 1)
                                            for _ in range(nlayer_outer)])

        self.subgraph_layers = nn.ModuleList([SubgraphGNNKernel(nhid, nhid, nlayer_inner, gnn_types, dropout, 
                                                                hop_dim=hop_dim, 
                                                                bias=not bn,
                                                                res=res, 
                                                                pooling=pooling,
                                                                embs=embs,
                                                                embs_combine_mode=embs_combine_mode,
                                                                subsampling=subsampling, 
                                                                online=online) 
                                              for _ in range(nlayer_outer)])
                                              
        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer_outer)])
        self.output_decoder = MLP(nhid, nout, nlayer=2, with_final_activation=False)
        
        # traditional layers: for comparison 
        if gnn_types[0] != 'PPGN':
            self.traditional_gnns = nn.ModuleList([getattr(gnn_wrapper, gnn_types[0])(nhid, nhid, bias=not bn) for _ in range(nlayer_outer)])
        else:
            # currently not support PPGN! 
            self.traditional_gnns = nn.ModuleList(Identity() for _ in range(nlayer_outer))
        # virtual node
        if vn:
            self.v0 = nn.Parameter(torch.zeros(1, nhid), requires_grad=True)
            self.vn_aggregator = VNAgg(nhid)

        # record params
        self.dropout = dropout
        self.num_inner_layers = nlayer_inner
        self.node_embedding = node_embedding
        self.use_normal_gnn = use_normal_gnn
        self.hop_dim = hop_dim
        self.vn = vn
        self.res = res
        self.pooling = pooling


    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.output_decoder.reset_parameters()
        for edge_encoder, layer, norm, old in zip(self.edge_encoders, self.subgraph_layers, self.norms, self.traditional_gnns):
            edge_encoder.reset_parameters()
            layer.reset_parameters()
            norm.reset_parameters()
            old.reset_parameters()
        if self.vn:
            self.vn_aggregator.reset_parameters()
            torch.nn.init.constant_(self.v0, 0)

    def forward(self, data):

        x = self.input_encoder(data.x.squeeze())
        # TODO: rethink how to deal with edge_attr = None
        ori_edge_attr = data.edge_attr 
        if ori_edge_attr is None:
            ori_edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))

        if self.vn:
            virtual_node = self.v0.expand(data.num_graphs, self.v0.shape[-1])
            x = x + virtual_node[data.batch]

        previous_x = x # for residual connection
        for i, (edge_encoder, subgraph_layer, normal_gnn, norm) in enumerate(zip(self.edge_encoders, 
                                        self.subgraph_layers, self.traditional_gnns, self.norms)):
            # if ori_edge_attr is not None: # Encode edge attr for each layer
            data.edge_attr = edge_encoder(ori_edge_attr) 
            data.x = x
            # with record_function(f"gnn conv {i}"):
            if self.num_inner_layers == 0: 
                # standard message passing nn 
                x = normal_gnn(data.x, data.edge_index, data.edge_attr)
            else:
                if self.use_normal_gnn:
                    x = subgraph_layer(data) + normal_gnn(data.x, data.edge_index, data.edge_attr[:,:-self.hop_dim])
                else:
                    x = subgraph_layer(data)

            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.res:
                x += previous_x
                previous_x = x # for residual connection
            
            if self.vn:
                virtual_node = self.vn_aggregator(virtual_node, x, data.batch)
                virtual_node = F.dropout(virtual_node, self.dropout, training=self.training)
                x = x + virtual_node[data.batch]

        if not self.node_embedding:
            # TODO: maybe use a transformation layer before scatter?s
            x = scatter(x, data.batch, dim=0, reduce=self.pooling)
            
        x = F.dropout(x, self.dropout, training=self.training)  
        x = self.output_decoder(x) 
        return x 
