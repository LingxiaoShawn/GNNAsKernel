import os, torch, numpy as np
import pickle
from core.data_utils import graph_algorithms
from core.data_utils.graph_generation import GraphType, generate_graph
from inspect import signature

from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import dense_to_sparse

class GraphPropertyDataset(InMemoryDataset):
    # parameters for generating the dataset
    seed=1234
    graph_type='RANDOM'
    extrapolation=False
    nodes_labels=["eccentricity", "graph_laplacian_features", "sssp"]
    graph_labels = ["is_connected", "diameter", "spectral_radius"]

    def __init__(self, root, split, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ["generated_data.pkl"]

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        # generate dataset
        print("Generating dataset...")
        genereate_dataset(root=self.raw_dir, seed=self.seed, graph_type=self.graph_type,
                          extrapolation=self.extrapolation, 
                          nodes_labels=self.nodes_labels, 
                          graph_labels=self.graph_labels)

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            (adj, features, node_labels, graph_labels) = pickle.load(f)

        # normalize labels
        max_node_labels = torch.cat([nls.max(0)[0].max(0)[0].unsqueeze(0) for nls in node_labels['train']]).max(0)[0]
        max_graph_labels = torch.cat([gls.max(0)[0].unsqueeze(0) for gls in graph_labels['train']]).max(0)[0]
        for dset in node_labels.keys():
            node_labels[dset] = [nls / max_node_labels for nls in node_labels[dset]]
            graph_labels[dset] = [gls / max_graph_labels for gls in graph_labels[dset]]

        graphs = to_torch_geom(adj, features, node_labels, graph_labels)
        for key, data_list in graphs.items():
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            torch.save((data, slices), os.path.join(self.processed_dir, f'{key}.pt'))
        
def to_torch_geom(adj, features, node_labels, graph_labels):
    graphs = {}
    for key in adj.keys():      # train, val, test
        graphs[key] = []
        for i in range(len(adj[key])):          # Graph of a given size
            batch_i = []
            for j in range(adj[key][i].shape[0]):       # Number of graphs
                graph_adj = adj[key][i][j]
                graph = Data(x=features[key][i][j],
                             edge_index=dense_to_sparse(graph_adj)[0],
                             y=graph_labels[key][i][j].unsqueeze(0),
                             pos=node_labels[key][i][j])
                batch_i.append(graph)

            graphs[key].extend(batch_i)
    return graphs

def genereate_dataset(root='data', seed=1234, graph_type='RANDOM', extrapolation=False,
                      nodes_labels=["eccentricity", "graph_laplacian_features", "sssp"],
                      graph_labels = ["is_connected", "diameter", "spectral_radius"]):
 
    if not os.path.exists(root):
        os.makedirs(root)

    if 'sssp' in nodes_labels:
        sssp = True
        nodes_labels.remove('sssp')
    else:
        sssp = False

    nodes_labels_algs = list(map(lambda s: getattr(graph_algorithms, s), nodes_labels))
    graph_labels_algs = list(map(lambda s: getattr(graph_algorithms, s), graph_labels))

    def get_nodes_labels(A, F, initial=None):
        labels = [] if initial is None else [initial]

        for f in nodes_labels_algs:
            params = signature(f).parameters
            labels.append(f(A, F) if 'F' in params else f(A))
        return np.swapaxes(np.stack(labels), 0, 1)

    def get_graph_labels(A, F):
        labels = []
        for f in graph_labels_algs:
            params = signature(f).parameters
            labels.append(f(A, F) if 'F' in params else f(A))
        return np.asarray(labels).flatten()  

    GenerateGraphPropertyDataset(n_graphs={'train': [512] * 10, 'val': [128] * 5, 'default': [256] * 5},
                                N={**{'train': range(15, 25), 'val': range(15, 25)}, **(
                                    {'test-(20,25)': range(20, 25), 'test-(25,30)': range(25, 30),
                                        'test-(30,35)': range(30, 35), 'test-(35,40)': range(35, 40),
                                        'test-(40,45)': range(40, 45), 'test-(45,50)': range(45, 50),
                                        'test-(60,65)': range(60, 65), 'test-(75,80)': range(75, 80),
                                        'test-(95,100)': range(95, 100)} if extrapolation else
                                    {'test': range(15, 25)})},
                                seed=seed, graph_type=getattr(GraphType, graph_type),
                                get_nodes_labels=get_nodes_labels, get_graph_labels=get_graph_labels,
                                sssp=True, filename=f"{root}/generated_data.pkl")

class GenerateGraphPropertyDataset:
    def __init__(self, n_graphs, N, seed, graph_type, get_nodes_labels, get_graph_labels, print_every=20, sssp=True, filename="./data/multitask_dataset.pkl"):
        self.adj = {}
        self.features = {}
        self.nodes_labels = {}
        self.graph_labels = {}

        def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd=""):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print('\r{} |{}| {}% {}'.format(prefix, bar, percent, suffix), end=printEnd)

        def to_categorical(x, N):
            v = np.zeros(N)
            v[x] = 1
            return v

        for dset in N.keys():
            if dset not in n_graphs:
                n_graphs[dset] = n_graphs['default']

            total_n_graphs = sum(n_graphs[dset])

            set_adj = [[] for _ in n_graphs[dset]]
            set_features = [[] for _ in n_graphs[dset]]
            set_nodes_labels = [[] for _ in n_graphs[dset]]
            set_graph_labels = [[] for _ in n_graphs[dset]]
            generated = 0

            progress_bar(0, total_n_graphs, prefix='Generating {:20}\t\t'.format(dset),
                         suffix='({} of {})'.format(0, total_n_graphs))

            for batch, batch_size in enumerate(n_graphs[dset]):
                for i in range(batch_size):
                    # generate a random graph of type graph_type and size N
                    seed += 1
                    adj, features, type = generate_graph(N[dset][batch], graph_type, seed=seed)

                    while np.min(np.max(adj, 0)) == 0.0:
                        # remove graph with singleton nodes
                        seed += 1
                        adj, features, _ = generate_graph(N[dset][batch], type, seed=seed)

                    generated += 1
                    if generated % print_every == 0:
                        progress_bar(generated, total_n_graphs, prefix='Generating {:20}\t\t'.format(dset),
                                     suffix='({} of {})'.format(generated, total_n_graphs))

                    # make sure there are no self connection
                    assert np.all(
                        np.multiply(adj, np.eye(N[dset][batch])) == np.zeros((N[dset][batch], N[dset][batch])))

                    if sssp:
                        # define the source node
                        source_node = np.random.randint(0, N[dset][batch])

                    # compute the labels with graph_algorithms; if sssp add the sssp
                    node_labels = get_nodes_labels(adj, features,
                                                   graph_algorithms.all_pairs_shortest_paths(adj, 0)[source_node]
                                                   if sssp else None)
                    graph_labels = get_graph_labels(adj, features)
                    if sssp:
                        # add the 1-hot feature determining the starting node
                        features = np.stack([to_categorical(source_node, N[dset][batch]), features], axis=1)

                    set_adj[batch].append(adj)
                    set_features[batch].append(features)
                    set_nodes_labels[batch].append(node_labels)
                    set_graph_labels[batch].append(graph_labels)

            self.adj[dset] = [torch.from_numpy(np.asarray(adjs)).float() for adjs in set_adj]
            self.features[dset] = [torch.from_numpy(np.asarray(fs)).float() for fs in set_features]
            self.nodes_labels[dset] = [torch.from_numpy(np.asarray(nls)).float() for nls in set_nodes_labels]
            self.graph_labels[dset] = [torch.from_numpy(np.asarray(gls)).float() for gls in set_graph_labels]
            progress_bar(total_n_graphs, total_n_graphs, prefix='Generating {:20}\t\t'.format(dset),
                         suffix='({} of {})'.format(total_n_graphs, total_n_graphs), printEnd='\n')

        self.save_as_pickle(filename)

    def save_as_pickle(self, filename):
        """" Saves the data into a pickle file at filename """
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, 'wb') as f:
            pickle.dump((self.adj, self.features, self.nodes_labels, self.graph_labels), f)


if __name__ == '__main__':
    dataset = GraphPropertyDataset(root='data/pna-simulation', split='train')
