from core.model import * 


class GNNAsKernelEnhanced(nn.Module):
    def __init__(self,  nfeat_node, nfeat_edge, nhid, nout, nlayer_outer, nlayer_inner, 
                        gnn_types, 
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
                        mlp_layers=1,
                        subsampling=False, 
                        num_transforms=1,
                        online=True):
        super().__init__()
        # special case: PPGN
        nlayer_ppgn = nlayer_inner
        if gnn_types[0] == 'PPGN' and nlayer_inner == 0:
            # normal PPGN
            nlayer_ppgn = nlayer_outer
            nlayer_outer = 1

        # nfeat_in is None: discrete input features
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, nhid, 1)
        # layers
        # edge_emd_dim = nhid if nlayer_inner == 0 else nhid // len(gnn_types)
        edge_emd_dim = nhid if nlayer_inner == 0 else nhid + hop_dim 
        self.edge_encoders = nn.ModuleList([DiscreteEncoder(edge_emd_dim) if nfeat_edge is None else MLP(nfeat_edge, edge_emd_dim, 1)
                                            for _ in range(nlayer_outer)])

        self.subgraph_layers = nn.ModuleList([nn.ModuleList([
                                                SubgraphGNNKernel(nhid, nhid, nlayer_inner, gnn_types, dropout, 
                                                                hop_dim=hop_dim, 
                                                                bias=not bn,
                                                                res=res, 
                                                                pooling=pooling,
                                                                embs=embs,
                                                                embs_combine_mode=embs_combine_mode,
                                                                mlp_layers=mlp_layers,
                                                                subsampling=subsampling, 
                                                                online=online)
                                                for _ in range(num_transforms)]
                                              )
                                              for _ in range(nlayer_outer)])
                                              
        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer_outer)])
        self.output_decoder = MLP(nhid, nout, nlayer=2, with_final_activation=False)
        
        # traditional layers: for comparison 
        if gnn_types[0] != 'PPGN':
            self.traditional_gnns = nn.ModuleList([getattr(gnn_wrapper, gnn_types[0])(nhid, nhid, bias=not bn) for _ in range(nlayer_outer)])
            # For correctly counting number of parameters
            # self.traditional_gnns = nn.ModuleList(Identity() for _ in range(nlayer_outer))
        else:
            self.traditional_gnns = nn.ModuleList(PPGN(nhid, nhid, nlayer_ppgn) for _ in range(nlayer_outer))
       # virtual node
        self.vn_aggregators = nn.ModuleList([VNUpdate(nhid) for _ in range(nlayer_outer)])
        # For correctly counting number of parameters
        # self.vn_aggregators = nn.ModuleList(Identity() for _ in range(nlayer_outer))

        # record params
        self.gnn_type = gnn_types[0]
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
        for edge_encoder, norm, old, vn in zip(self.edge_encoders, self.norms, self.traditional_gnns, self.vn_aggregators):
            edge_encoder.reset_parameters()
            norm.reset_parameters()
            old.reset_parameters()
            vn.reset_parameters()

        for list_layers in self.subgraph_layers:
            for layer in list_layers:
                layer.reset_parameters()



    def forward(self, data_list):
        # assume we have list of data as input 
        x = data_list[0].x if len(data_list[0].x.shape) <= 2 else data_list[0].x.squeeze(-1)
        x = self.input_encoder(x)
    
        # TODO: rethink how to deal with edge_attr = None
        ori_edge_attr =  data_list[0].edge_attr 
        if ori_edge_attr is None:
            ori_edge_attr =  data_list[0].edge_index.new_zeros(data_list[0].edge_index.size(-1))

        previous_x = x # for residual connection
        virtual_node = None
        for i, (edge_encoder, subgraph_layer, normal_gnn, norm, vn_aggregator) in enumerate(zip(self.edge_encoders, 
                                        self.subgraph_layers, self.traditional_gnns, self.norms, self.vn_aggregators)):
            # if ori_edge_attr is not None: # Encode edge attr for each layer
            edge_attr = edge_encoder(ori_edge_attr) 
            for data in data_list:
                data.edge_attr = edge_attr
                data.x = x
            if self.num_inner_layers == 0: 
                # standard message passing nn 
                if self.gnn_type == 'PPGN':
                    x = normal_gnn(data.x, data.edge_index, data.edge_attr, data.batch)
                else:
                    x = normal_gnn(data.x, data.edge_index, data.edge_attr)
            else:
                if self.use_normal_gnn:
                    edge_attr = data.edge_attr[:,:-self.hop_dim] if self.hop_dim > 0 else data.edge_attr
                    if self.gnn_type == 'PPGN':
                        x = sum([layer(d) for layer, d in zip(subgraph_layer, data_list)]) + normal_gnn(data.x, data.edge_index, edge_attr, data.batch)
                    else:
                        x = sum([layer(d) for layer, d in zip(subgraph_layer, data_list)]) + normal_gnn(data.x, data.edge_index, edge_attr)
                else:
                    x = sum([layer(d) for layer, d in zip(subgraph_layer, data_list)])

            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.vn:
                virtual_node, x = vn_aggregator(virtual_node, x, data.batch)

            if self.res:
                x += previous_x
                previous_x = x # for residual connection

        if not self.node_embedding:
            # TODO: maybe use a transformation layer before scatter?s
            x = scatter(x, data.batch, dim=0, reduce=self.pooling)
            
        x = F.dropout(x, self.dropout, training=self.training)  
        x = self.output_decoder(x) 
        return x 