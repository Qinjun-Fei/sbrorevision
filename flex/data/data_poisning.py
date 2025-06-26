from typing import Callable
from dataclasses import replace
import functools
from flex.data import Dataset
from flex.data.lazy_indexable import LazyIndexable
import numpy as np

def data_poisoner(poison_func: Callable):
    @functools.wraps(poison_func)
    def _poison_dataset_(dataset: Dataset, *args, **kwargs) -> Dataset:
        flip_ratio = kwargs.get("nodes_label_flip_ratio")[args[0]]
        labels_set = kwargs.get("labels_set")
        if flip_ratio > 0:
            new_y_data = poison_func(dataset.y_data, flip_ratio,labels_set)
            new_dataset = replace(dataset, y_data=new_y_data)
        else:
            return dataset
        return new_dataset

    return _poison_dataset_

@data_poisoner
def flip_labels(y_data: LazyIndexable, flip_ratio: float,classes:list) -> LazyIndexable:
    num_to_flip = int(len(y_data) * flip_ratio)
    indices_to_flip = np.random.choice(len(y_data), num_to_flip, replace=False)
    new_y_data = y_data.to_numpy()

    for idx in indices_to_flip:
        original_label = y_data[idx]
        new_label = np.random.choice([label for label in classes if label != original_label])
        new_y_data[idx] = new_label

    return LazyIndexable(np.array(new_y_data), len(new_y_data))

# Select clients to use label flipping to perform data poisoning attacks
def assign_flip_ratios(nodes_ids, client_flip_ratio, label_flip_ratio):
    """
    Assign flip ratios to nodes based on given client flip ratios and label flip ratios.

    Parameters:
    -----------
    nodes_ids : list
        List of node IDs.
    client_flip_ratio : list
        List of ratios or counts of nodes to flip. All elements must be either float (0 to 1) or int.
    label_flip_ratio : list
        List of label flip ratios (0 to 1) corresponding to each client flip ratio.

    Returns:
    --------
    node_flip_dict : dict
        Dictionary where keys are node IDs and values are the assigned label flip ratios.
    label_flip_dict : dict
        Dictionary where keys are label flip ratios and values are lists of node IDs.
    """
    import random

    assert len(client_flip_ratio) == len(
        label_flip_ratio), "client_flip_ratio and label_flip_ratio must have the same length"

    nodes_ids = list(nodes_ids)
    if all(isinstance(x, float) for x in client_flip_ratio):
        total_nodes = len(nodes_ids)
        num_nodes_to_flip = [int(total_nodes * ratio) for ratio in client_flip_ratio]
    elif all(isinstance(x, int) for x in client_flip_ratio):
        num_nodes_to_flip = client_flip_ratio
    else:
        raise ValueError("client_flip_ratio must be all floats or all ints")

    node_flip_dict = {node_id: 0 for node_id in nodes_ids}
    label_flip_dict = {ratio: [] for ratio in label_flip_ratio}
    label_flip_dict[0] = []

    for num, ratio in zip(num_nodes_to_flip, label_flip_ratio):
        selected_nodes = random.sample(nodes_ids, num)
        for node in selected_nodes:
            node_flip_dict[node] = ratio
            label_flip_dict[ratio].append(node)
            nodes_ids.remove(node)

    # Add remaining nodes to the label_flip_dict with key 0
    for node in nodes_ids:
        label_flip_dict[0].append(node)

    return node_flip_dict, label_flip_dict
