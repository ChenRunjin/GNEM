import os
import pandas as pd
from torch.utils.data import Dataset


class MatchingDataset(Dataset):
    def __init__(self, path, tableA, tableB):
        assert os.path.isfile(path), "{} is not a file".format(path)

        self.data = pd.read_csv(path)

        self.tableA = tableA
        self.tableB = tableB

        self.ltable_id_list = [(i, 'l') for i in self.data["ltable_id"].unique().tolist()]
        self.rtable_id_list = [(i, 'r') for i in self.data["rtable_id"].unique().tolist()]
        self.center_id_list = self.ltable_id_list + self.rtable_id_list
        self.examples = [self._make_example(*i) for i in self.center_id_list]

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x

    def _make_example(self, center_id, type='l'):
        if type == 'l':
            neighbor = self.data[self.data["ltable_id"] == center_id]
            neighbor_ids = neighbor["rtable_id"].values.tolist()
            center_example = self.tableA[self.tableA["id"] == center_id].values.tolist()[0]
            neighbor_examples = [self.tableB[self.tableB["id"] == i].values.tolist()[0] for i in neighbor_ids]
            neighbor_masks = [1] * len(neighbor_ids)
        elif type == 'r':
            neighbor = self.data[self.data["rtable_id"] == center_id]
            neighbor_ids = neighbor["ltable_id"].values.tolist()
            center_example = self.tableB[self.tableB["id"] == center_id].values.tolist()[0]
            neighbor_examples = [self.tableA[self.tableA["id"] == i].values.tolist()[0] for i in neighbor_ids]
            neighbor_masks = [1] * len(neighbor_ids)
        else:
            raise NotImplementedError
        labels = neighbor["label"].values.tolist()

        example = {
            "type": type,
            "center": center_example,
            "neighbors": neighbor_examples,
            "neighbors_mask": neighbor_masks,
            "labels": labels,
        }

        return example


class MergedMatchingDataset(Dataset):
    def __init__(self, path,  tableA, tableB, other_path=None):
        if other_path and not isinstance(other_path, list):
            other_path = [other_path]
        assert os.path.isfile(path), "{} is not a file".format(path)
        for p in other_path:
            if p:
                assert os.path.isfile(p), "{} is not a file".format(p)

        self.data = pd.read_csv(path)
        if other_path:
            self.other_data = pd.read_csv(other_path[0])
            for p in other_path[1:]:
                if p:
                    self.other_data = pd.concat([self.other_data, pd.read_csv(p)], axis=0)
        else:
            self.other_data = None

        self.tableA = tableA
        self.tableB = tableB

        self.ltable_id_list = [(i, 'l') for i in self.data["ltable_id"].unique().tolist()]
        self.rtable_id_list = [(i, 'r') for i in self.data["rtable_id"].unique().tolist()]
        self.center_id_list = self.ltable_id_list + self.rtable_id_list
        self.examples = [self._make_example(*i) for i in self.center_id_list]

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x

    def _make_example(self, center_id, type='l'):
        if type == 'l':
            self_neighbor = self.data[self.data["ltable_id"] == center_id]
            self_neighbor_ids = self_neighbor["rtable_id"].values.tolist()
            other_neighbor = self.other_data[self.other_data["ltable_id"] == center_id]
            other_neighbor_ids = other_neighbor["rtable_id"].values.tolist()
            neighbor_ids = self_neighbor_ids + other_neighbor_ids
            center_example = self.tableA[self.tableA["id"] == center_id].values.tolist()[0]
            neighbor_examples = [self.tableB[self.tableB["id"] == i].values.tolist()[0] for i in neighbor_ids]
            # 0 for not use label, 1 for use label
            neighbor_masks = [1] * len(self_neighbor_ids) + [0] * len(other_neighbor_ids)

        elif type == 'r':
            self_neighbor = self.data[self.data["rtable_id"] == center_id]
            self_neighbor_ids = self_neighbor["ltable_id"].values.tolist()
            other_neighbor = self.other_data[self.other_data["rtable_id"] == center_id]
            other_neighbor_ids = other_neighbor["ltable_id"].values.tolist()
            neighbor_ids = self_neighbor_ids + other_neighbor_ids
            center_example = self.tableB[self.tableB["id"] == center_id].values.tolist()[0]
            neighbor_examples = [self.tableA[self.tableA["id"] == i].values.tolist()[0] for i in neighbor_ids]
            # 0 for not use label, 1 for use label
            neighbor_masks = [1] * len(self_neighbor_ids) + [0] * len(other_neighbor_ids)

        else:
            raise NotImplementedError
        labels = self_neighbor["label"].values.tolist() + other_neighbor["label"].values.tolist()

        example = {
            "type": type,
            "center": center_example,
            "neighbors": neighbor_examples,
            "neighbors_mask": neighbor_masks,
            "labels": labels
        }

        return example


def collate_fn(batch):
    return batch

