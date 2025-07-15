from copy import deepcopy
import os
import torch
import numpy as np
import random

from flex.pool import select_nodes
from flex.common import save_flex_model,load_flex_model
import torch.nn as nnf
from utils import SeedManager

# Set Parameters
DATASET_NAME = 'cifar'
INITIAL_MODEL_PATH = "../result/normal/cifar_1009_0812/CIFAR10_CNN.pkl"
done_result_file = '../result/normal/cifar_1009_0812/result_final.json'

# DATASET_NAME = 'svhn'
# INITIAL_MODEL_PATH = "../result/normal/svhn_1008_2055/SVHN_CNN.pkl"
# done_result_file = '../result/normal/svhn_1008_2055/result_final.json'

# DATASET_NAME = 'emnist'
# INITIAL_MODEL_PATH = "../result/normal/emnist_1008_1515/EMNIST_CNN.pkl"
# done_result_file = '../result/normal/emnist_1008_1515/result_final.json'

# DATASET_NAME = 'fashionmnist'
# INITIAL_MODEL_PATH = "../result/normal/fashionmnist_1009_0004/FashionMNIST_CNN.pkl"
# done_result_file = '../result/normal/fashionmnist_1009_0004/result_final.json'

SAVE_DIR_BASE = f"../result/sp_run_validation/"

def get_experiment_data_params(group_name):
    noise_params = {
        'client_flip_ratio': [0.2, 0.2, 0.2, 0.2],
        'label_flip_ratio': [0.9, 0.8, 0.7, 0.6]
    }
    if group_name == 'emnist':
        data_params = {
            'data_name': "emnist_letters",
            'title': 'EMNIST-letter',
            'model_name': "EMNIST_CNN",
            'optimizer': "sgd",
            'initial_lr': 0.01,
            'final_lr': 0.001,
            'total_rounds': 300,
            'batch_size': 16,
            'training_size': 10000,
            'validation_size': 2000
        }
    elif group_name == "cifar":
        data_params = {
            'data_name': "cifar10",
            'title': 'CIFAR-10',
            'model_name': "CIFAR10_CNN",
            'optimizer': "sgd",
            'initial_lr': 0.01,
            'final_lr': 0.001,
            'total_rounds': 300,
            'batch_size': 16,
            'training_size': 10000,
            'validation_size': 2000
        }
    elif  group_name == 'fashionmnist':
        data_params = {
            'data_name': "fashionmnist",
            'title': 'FashionMnist',
            'model_name': "FashionMNIST_CNN",
            'optimizer': "sgd",
            'initial_lr': 0.01,
            'final_lr': 0.001,
            'total_rounds': 300,
            'batch_size': 16,
            'training_size': 10000,
            'validation_size': 2000
        }
    elif group_name == 'svhn':
        data_params = {
            'data_name': "svhn",
            'title': 'SVHN',
            'model_name': "SVHN_CNN",
            'optimizer': "sgd",
            'initial_lr': 0.1,
            'final_lr': 0.001,
            'total_rounds': 300,
            'batch_size': 16,
            'training_size': 10000,
            'validation_size': 2000
        }
    else:
        raise ValueError(f"Unknown dataset name: {group_name}")
    common_params = {
        'seed': 1,
        'eachclient_epoch': 1,
        'ω': 10,
        'ψ': 5,
        'α': 0.15,
        'β': 0.3,
        'γ': 1,
        'n_nodes': 40,
        'Budget': 45,
        'Bid_mean': 10,
        'Bid_stddev': 1,
        'n_rounds': 300
    }
    return {**data_params, **noise_params, **common_params}

experiment_parameters = get_experiment_data_params(DATASET_NAME)

from types import SimpleNamespace
exparam = SimpleNamespace(**experiment_parameters)
# Create directory if it does not exist
import datetime
date_str = datetime.datetime.now().strftime('%m%d_%H%M')
file_path = os.path.join(SAVE_DIR_BASE, '_'.join([DATASET_NAME, date_str]))
from flex.common import create_directory
create_directory(file_path)

# Save the parameters
import yaml
with open(os.path.join(file_path, 'params.yaml'), 'w') as f:
    yaml.dump(experiment_parameters, f)

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Set seed for reproducibility
def set_seeds(seed_value=1):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)  # Set seed for NumPy
    random.seed(seed_value)  # Set seed for Python's Random
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seeds(exparam.seed)

# Load the dataset
from flex.datasets import load
train_dataset, tests_dataset, _,unused_train = load(exparam.data_name, out_dir="../data_original/", training_size=exparam.training_size
                                                 , validation_size=exparam.validation_size, test_size=None,return_train_unused = True)

# Configure the federated data distribution
from flex.data import FedDataDistribution, FedDatasetConfig
config = FedDatasetConfig(
    n_nodes=exparam.n_nodes,
    shuffle=True,
    replacement=False,
    seed=exparam.seed
)
federated_data = FedDataDistribution.from_config(train_dataset, config)

# Assign flip ratios to nodes based on given client flip ratios and label flip ratios
from flex.data.data_poisning import flip_labels, assign_flip_ratios
nodes_label_flip_ratio, label_flip_ratio_nodes = assign_flip_ratios(federated_data.data.keys(),
                                                                    exparam.client_flip_ratio, exparam.label_flip_ratio)
Bid_dict = {node_id: np.random.normal(exparam.Bid_mean, exparam.Bid_stddev) for node_id in
            federated_data.data.keys()}
Budget = exparam.Budget
labels_set = set(train_dataset.y_data)
federated_data = federated_data.apply(flip_labels,
                                      nodes_label_flip_ratio=nodes_label_flip_ratio,
                                      label_flip_ratio_nodes=label_flip_ratio_nodes,
                                      labels_set=labels_set,
                                      transfer_node_ids=True)
server_id = "server"
federated_data[server_id] = tests_dataset

import importlib
import inspect
import os
module = importlib.import_module('flex.model')
if hasattr(module, exparam.model_name):
    ModelNet = getattr(module, exparam.model_name)
    class_code = inspect.getsource(ModelNet)
    class_filename = f"{exparam.model_name}_class_definition.py"
    class_filepath = os.path.join(file_path, class_filename)
    with open(class_filepath, 'w') as f:
        f.write(class_code)
    print(f"Class definition for {exparam.model_name} saved to {class_filepath}")
else:
    raise ValueError(f"No class named {exparam.model_name} found in {'flex.model'}")

from flex.pool import init_server_model
from flex.model import FlexModel
@init_server_model
def build_server_model():
    server_flex_model = FlexModel()
    set_seeds(exparam.seed)
    server_flex_model["model"] = ModelNet()
    # Required to store this for later stages of the FL training process
    server_flex_model["criterion"] = torch.nn.CrossEntropyLoss()
    if exparam.optimizer == "Adam":
        server_flex_model["optimizer_func"] = torch.optim.Adam
        server_flex_model["optimizer_kwargs"] = {'lr': exparam.initial_lr}
    else:
        server_flex_model["optimizer_func"] = torch.optim.SGD
        server_flex_model["optimizer_kwargs"] = {"lr": exparam.initial_lr, "momentum": 0.9}

    return server_flex_model

from flex.data import Dataset
from torch.utils.data import DataLoader

def exponential_lr(round, total_rounds, initial_lr, final_lr):
    return initial_lr * (final_lr / initial_lr) ** (round / (total_rounds-1))
def train(client_flex_model: FlexModel, client_data: Dataset, **kwargs):
    rounds = kwargs.get("round", 0)
    client_dataloader = DataLoader(client_data, batch_size=exparam.batch_size, shuffle=True)
    model = client_flex_model["model"]
    optimizer = client_flex_model["optimizer_func"](
        model.parameters(), **client_flex_model["optimizer_kwargs"]
    )
    if exparam.optimizer == "sgd":
        new_lr = exponential_lr(rounds, exparam.total_rounds, exparam.initial_lr, exparam.final_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    model = model.train()
    model = model.to(device)
    criterion = client_flex_model["criterion"]
    for _ in range(exparam.eachclient_epoch):
        for imgs, labels in client_dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
def evaluate_global_model(server_flex_model: FlexModel, test_data: Dataset):
    model = server_flex_model["model"]
    model.eval()
    test_loss = 0
    test_acc = 0
    total_count = 0
    model = model.to(device)
    criterion = server_flex_model["criterion"]
    # get test data as a torchvision object
    test_dataloader = DataLoader(
        test_data, batch_size=256, shuffle=False, pin_memory=True
    )
    losses = []
    with torch.no_grad():
        for data, target in test_dataloader:
            total_count += target.size(0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            pred = output.data.max(1, keepdim=True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss = sum(losses) / len(losses)
    test_acc /= total_count
    return test_loss, test_acc

from flex.pool import deploy_server_model_pt
from flex.pool.utilsSVselection import update_reputation, random_select_clients_within_budget,random_select_clients_within_budget_in_normal
from flex.pool import collect_clients_weights_pt
from collections import defaultdict, deque
from flex.pool import FlexPool
from flex.pool import fed_avg
from flex.pool import set_aggregated_weights_pt
from flex.pool.utilsSVselection import exact_shapley_value
import matplotlib.pyplot as plt

validation_sizes_to_test = [200, 1000, 2000]
validation_flips_to_test = [0.0, 0.3, 0.7]


def sbro_training(selection_history_lastrounds, flex_pool, Reputation_dict, acc_history_list, sv_record,
                  SBRO_selected_client_id_list, clients_per_round,round,val_dataset):
    recent_selection_counts = {client_id: sum(selection_history_lastrounds[client_id]) for client_id in
                               selection_history_lastrounds.keys()}
    SBRO_selected_pool = flex_pool.clients.select(
        clients_per_round(Reputation_dict, Bid_dict, Budget, exparam.γ, exparam.α, exparam.β, round,
                          recent_selection_counts))
    SBRO_selected_clients = SBRO_selected_pool.clients
    SBRO_selected_client_id_list.append(SBRO_selected_clients.actor_ids)
    # Record the selection history of the selected clients
    for client_id in SBRO_selected_clients.actor_ids:
        selection_history_lastrounds[client_id].append(1)  # indicate this client has been selected
    for client_id in selection_history_lastrounds.keys():
        if client_id not in SBRO_selected_clients.actor_ids:
            selection_history_lastrounds[client_id].append(0)  # indicate this client has not been selected
    print(
        f"Selected clients for this round and flip ratio: { {id: nodes_label_flip_ratio[id] for id in SBRO_selected_clients.actor_ids} }")
    # Deploy the server model to the selected clients
    flex_pool.servers.map(deploy_server_model_pt, SBRO_selected_clients)
    # Each selected client trains her model
    SBRO_selected_clients.map(train, round=round)
    # Calculate Shapley values for each client
    shapley_values = exact_shapley_value(SBRO_selected_clients, acc_history_list[-1], val_dataset, device)
    # print(shapley_values)
    # Update reputation and bid for each client
    update_reputation(Reputation_dict, Bid_dict, shapley_values, sv_record, exparam.ω, exparam.ψ, 5)
    for k, v in shapley_values.items():
        sv_record[k].append(v)

    # The aggregador collects weights from the selected clients and aggregates them
    flex_pool.aggregators.map(collect_clients_weights_pt, SBRO_selected_clients)
    flex_pool.aggregators.map(fed_avg)
    # The aggregator send its aggregated weights to the server
    flex_pool.aggregators.map(set_aggregated_weights_pt, flex_pool.servers)
    metrics = flex_pool.servers.map(evaluate_global_model)
    loss, acc = metrics[0]
    print(f" Test acc: {acc:.4f}, test loss: {loss:.4f}")
    acc_history_list.append(acc)


def create_validation(dataset, validation_size: int, flip_ratio: float):
    """
    Create a validation Dataset by sampling from `dataset`
    and optionally flipping a portion of its labels.

    Parameters
    ----------
    dataset : Dataset
        Original (unused) training Dataset.
    validation_size : int
        Number of samples in the validation split.
    flip_ratio : float
        Fraction of labels to flip. 0 means no flipping.

    Returns
    -------
    val_dataset : Dataset
        Validation subset (labels flipped if flip_ratio > 0).
    """
    # 1. Deep-copy to avoid modifying the original dataset
    val_dataset = deepcopy(dataset)
    from flex.data import LazyIndexable
    if validation_size > len(val_dataset):
        raise ValueError("validation_size exceeds dataset size.")
    # 2. Randomly sample validation_size samples from the dataset
    indices = random.sample(range(len(val_dataset)), validation_size)
    val_dataset_cut = val_dataset[indices]

    # 3. 自动推断标签集合并执行 label-flipping

    x_sub = val_dataset_cut.X_data.to_numpy()
    y_sub = val_dataset_cut.y_data.to_numpy()

    classes = np.unique(val_dataset.y_data.to_numpy())
    # ---------- 3. 执行标签翻转 ----------
    if flip_ratio > 0:
        n_flip = int(len(y_sub) * flip_ratio)
        flip_idx = np.random.choice(len(y_sub), n_flip, replace=False)
        for idx in flip_idx:
            old = y_sub[idx]
            new = np.random.choice([c for c in classes if c != old])
            y_sub[idx] = new

    val_dataset_object = Dataset.from_array(x_sub, y_sub)

    return val_dataset_object

with open(done_result_file, 'r') as f:
    done_result = yaml.unsafe_load(f)

def train_n_rounds(n_rounds, clients_per_round, initial_model_path):
    acc_dict = {}
    selected_clients_dict = {}
    sv_record_dict = {}
    FL_pools = {}
    val_dict = {}
    selected_last_five_rounds = {}
    reputation_dict = {}

    initial_server_model = load_flex_model(initial_model_path)

    @init_server_model
    def build_from_saved():
        return deepcopy(initial_server_model)

    sm = SeedManager(42)
    sm.set_global_seed()

    flpool = FlexPool.client_server_pool(fed_dataset=federated_data, server_id=server_id, init_func=build_from_saved)
    metrics_initial = flpool.servers.map(evaluate_global_model)
    loss_initial, acc_initial = metrics_initial[0]
    print(f"Server: inital Test acc: {acc_initial:.4f}, test loss: {loss_initial:.4f}")

    # Initialize FlexPool for SBRO
    for valsize in validation_sizes_to_test:
        for flip in validation_flips_to_test:
            reputation_dict[str(valsize) + str(flip)] = {node_id: [0] for node_id in federated_data.data.keys() if node_id != "server"}
            sm.register_algorithm(str(valsize)+str(flip))
            val_dict[str(valsize) + str(flip)] = create_validation(unused_train, validation_size=valsize, flip_ratio=flip)
            acc_dict[str(valsize) + str(flip)] = [acc_initial]
            selected_clients_dict[str(valsize) + str(flip)] = []
            flpool = FlexPool.client_server_pool(fed_dataset=federated_data, server_id=server_id, init_func=build_from_saved)
            FL_pools[str(valsize) + str(flip)] = flpool
            sv_record_dict[str(valsize) + str(flip)] = {actor_id: [] for actor_id in flpool.clients.actor_ids}
            selected_last_five_rounds[str(valsize) + str(flip)] = {client_id: deque(maxlen=5) for client_id in flpool.clients.actor_ids}

    for i in range(n_rounds):
        print(f"\nRunning round: {i + 1} of {n_rounds}")
        for valsize in validation_sizes_to_test:
            for flip in validation_flips_to_test:
                print('-------------------------------------------------------------------------')
                print(f"Validation size: {valsize}, Flip ratio: {flip}")
                sm.prepare_round(str(valsize) + str(flip),i)
                sbro_training(selected_last_five_rounds[str(valsize) + str(flip)],
                              FL_pools[str(valsize) + str(flip)],
                              reputation_dict[str(valsize) + str(flip)],
                              acc_dict[str(valsize) + str(flip)],
                              sv_record_dict[str(valsize) + str(flip)],
                              selected_clients_dict[str(valsize) + str(flip)],
                                clients_per_round,
                                i,
                                val_dict[str(valsize) + str(flip)]
                                )

        print('-------------------------------------------------------------------------')


        if (i + 1) % 20 == 0:
            data = {
                'Bid_dict': Bid_dict,
                'Reputation_dict': reputation_dict,
                'nodes_label_flip_ratio': nodes_label_flip_ratio,
                'experiment_parameters': experiment_parameters,
                'acc_history_SBRO_list': acc_dict,
                'SBRO_selected_client_id_list': selected_clients_dict,
                'sv_record': sv_record_dict
            }
            x_axis = range(i+2)
            plt.figure(figsize=(10, 6))
            for valsize in validation_sizes_to_test:
                for flip in validation_flips_to_test:
                    plt.plot(x_axis, acc_dict[str(valsize) + str(flip)], label=f'SBRO-FL {valsize}-{flip}')
            plt.plot(x_axis, done_result['acc_history_SBRO_list'][:i+2], label='SBRO-FL')
            plt.plot(x_axis, done_result['acc_history_RS_list'][:i+2], label='RS-FL')
            plt.plot(x_axis, done_result['acc_history_All_list'][:i+2], label='All-FL')
            plt.plot(x_axis, done_result['acc_history_HQRS_list'][:i+2], label='HQRS-FL')
            plt.title(exparam.title, fontsize=20, pad=10)
            plt.xlabel('Round', fontsize=16)
            plt.ylabel('Accuracy', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(False)
            plt.legend(loc='lower right', fontsize=14)
            image_filename = str(i) + '.png'
            image_path = os.path.join(file_path, image_filename)
            plt.savefig(image_path)

            with open(os.path.join(file_path, 'result{}.json'.format(i)), 'w') as f:
                yaml.dump(data, f)

train_n_rounds(exparam.n_rounds, clients_per_round=select_nodes, initial_model_path=INITIAL_MODEL_PATH)

