import os, torch, numpy as np
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
import shutil
import networkx as nx 
import pickle

class CyclesDataset(InMemoryDataset):
    url = 'https://drive.switch.ch/index.php/s/hv65hmY48GrRAoN/download'
    def __init__(self, root, train, k=8, n=50, proportion=1.0, n_samples=10000, transform=None, pre_transform=None, pre_filter=None):
        self.train = train
        self.k, self.n, self.n_samples = k, n, n_samples
        self.s = 'train' if train else 'test'
        self.proportion = proportion
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}cycles_n{}_{}samples_{}.pt'.format(self.k, self.n, self.n_samples, self.s)]

    @property
    def processed_file_names(self):
        return [f'processed_{self.k}cycles_n{self.n}_{self.n_samples}samples_{self.s}_{self.proportion}.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        self.build_dataset()

    def build_dataset(self):
        """ Given pickle files, split the dataset into one per value of n
        Run once before running the experiments. """
        n_samples = 10000
        for k in [4, 6, 8]:
            with open(os.path.join(self.raw_dir, f'datasets_kcycle_k={k}_nsamples={n_samples}.pickle'), 'rb') as f:
                datasets_params, datasets = pickle.load(f)
                # Split by graph size
                for params, dataset in zip(datasets_params, datasets):
                    n = params['n']
                    train, test = dataset[:n_samples], dataset[n_samples:]
                    torch.save(train, os.path.join(self.raw_dir, f'{k}cycles_n{n}_{n_samples}samples_train.pt'))
                    torch.save(test, os.path.join(self.raw_dir, f'{k}cycles_n{n}_{n_samples}samples_test.pt'))
                    
    def process(self):
        # Read data into huge `Data` list.
        dataset = torch.load(os.path.join(self.raw_dir, f'{self.k}cycles_n{self.n}_{self.n_samples}samples_{self.s}.pt'))

        data_list = []
        for sample in dataset:
            graph, y, label = sample
            edge_list = nx.to_edgelist(graph)
            edges = [np.array([edge[0], edge[1]]) for edge in edge_list]
            edges2 = [np.array([edge[1], edge[0]]) for edge in edge_list]

            edge_index = torch.tensor(np.array(edges + edges2).T, dtype=torch.long)

            x = torch.ones(graph.number_of_nodes(), 1, dtype=torch.float)
            y = torch.tensor([1], dtype=torch.long) if label == 'has-kcycle' else torch.tensor([0], dtype=torch.long)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=None, y=y))
        # Subsample the data
        if self.train:
            all_data = len(data_list)
            to_select = int(all_data * self.proportion)
            print(to_select, "samples were selected")
            data_list = data_list[:to_select]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    print(CyclesDataset('data/CYCLE', False))