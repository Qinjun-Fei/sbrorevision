import torch
import numpy as np
import random
from flex.pool import select_nodes
from flex.common import save_flex_model
import torch.nn as nn


# Set Parameters
dataset_name = 'emnist'
price_test = False
save_dir = "../result"
def get_experiment_data_params(group_name):
    noise_params = {
        'client_flip_ratio': [0.2, 0.2, 0.2, 0.2],
        'label_flip_ratio': [0.9, 0.8, 0.7, 0.6]
    }
    if group_name == 'emnist':
        data_params = {
            'data_name': "emnist_letters",
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
    return {**data_params, **noise_params}
def generate_experiment_parameters(dataset_name):
    group_params = get_experiment_data_params(dataset_name)
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
    return {**common_params, **group_params}

experiment_parameters = generate_experiment_parameters(dataset_name)
import os
if price_test:
    experiment_parameters['save_dir'] = os.path.join(save_dir, 'price_test')
else:
    experiment_parameters['save_dir'] = os.path.join(save_dir, 'normal')
from types import SimpleNamespace
exparam = SimpleNamespace(**experiment_parameters)

# Create directory if it does not exist
import datetime
date_str = datetime.datetime.now().strftime('%m%d_%H%M')
file_path = os.path.join(exparam.save_dir, '_'.join(['price',dataset_name, date_str]))
from flex.common import create_directory
create_directory(file_path)

# Save the parameters
import yaml
with open(os.path.join(file_path, 'params.yaml'), 'w') as f:
    yaml.dump(experiment_parameters, f)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
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
train_dataset, tests_dataset, val_dataset = load(exparam.data_name, out_dir="../data_original/", training_size=exparam.training_size
                                                 , validation_size=exparam.validation_size, test_size=None)
# Configure the federated data distribution
from flex.data import FedDataDistribution, FedDatasetConfig
config = FedDatasetConfig(
    n_nodes=exparam.n_nodes,
    shuffle=True,
    replacement=False,
    seed=exparam.seed
)
federated_data = FedDataDistribution.from_config(train_dataset, config)

# set reputation for each node in flex pool, initially all nodes have reputation 0
Reputation_dict = {node_id: [0] for node_id in federated_data.data.keys()}

# Assign flip ratios to nodes based on given client flip ratios and label flip ratios
from flex.data.data_poisning import flip_labels, assign_flip_ratios
nodes_label_flip_ratio, label_flip_ratio_nodes = assign_flip_ratios(federated_data.data.keys(),
                                                                    exparam.client_flip_ratio, exparam.label_flip_ratio)

# Set initial bid for each client
if price_test:
    # set bid according to the flip ratio
    Bid_dict = {}
    for node_id, flip_ratio in nodes_label_flip_ratio.items():
        if flip_ratio == 0:
            Bid_dict[node_id] = 14
        elif flip_ratio == 0.9:
            Bid_dict[node_id] = 6
        elif flip_ratio == 0.8:
            Bid_dict[node_id] = 8
        elif flip_ratio == 0.7:
            Bid_dict[node_id] = 10
        elif flip_ratio == 0.6:
            Bid_dict[node_id] = 12
else:
    # set bid with Gaussian random numbers with a mean of 10 and a variance of 1 as initial bid for each node
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
def train_n_rounds(n_rounds, clients_per_round):
    acc_history_SBRO_list = []
    acc_history_RS_list = []
    acc_history_All_list = []
    acc_history_HQRS_list = []

    SBRO_selected_client_id_list = []

    pool_SBRO = FlexPool.client_server_pool(
        fed_dataset=federated_data, server_id=server_id, init_func=build_server_model
    )

    # record shapley values for each client
    sv_record = defaultdict(list)
    for actor_id in pool_SBRO.clients.actor_ids:
        sv_record[actor_id]

    save_flex_model(file_path, exparam.model_name, pool_SBRO.servers._models['server'])
    metrics = pool_SBRO.servers.map(evaluate_global_model)
    loss, acc = metrics[0]
    print(f"Server: inital Test acc: {acc:.4f}, test loss: {loss:.4f}")

    acc_history_SBRO_list.append(acc)
    acc_history_RS_list.append(acc)
    acc_history_All_list.append(acc)
    acc_history_HQRS_list.append(acc)


    pool_All = FlexPool.client_server_pool(
        fed_dataset=federated_data, server_id=server_id, init_func=build_server_model
    )
    pool_HQRS = FlexPool.client_server_pool(
        fed_dataset=federated_data, server_id=server_id, init_func=build_server_model
    )
    pool_RS = FlexPool.client_server_pool(
        fed_dataset=federated_data, server_id=server_id, init_func=build_server_model
    )

    # record the selection history of in the last 5 rounds
    selection_history_last5rounds = {client_id: deque(maxlen=5) for client_id in pool_SBRO.clients.actor_ids}
    for i in range(n_rounds):

        print(f"\nRunning round: {i + 1} of {n_rounds}")

        recent_selection_counts = {client_id: sum(selection_history_last5rounds[client_id]) for client_id in
                                   selection_history_last5rounds.keys()}

        SBRO_selected_pool = pool_SBRO.clients.select(
            clients_per_round(Reputation_dict, Bid_dict, Budget,exparam.γ, exparam.α, exparam.β,i,recent_selection_counts))
        SBRO_selected_clients = SBRO_selected_pool.clients
        SBRO_selected_client_id_list.append(SBRO_selected_clients.actor_ids)
        # Record the selection history of the selected clients
        for client_id in SBRO_selected_clients.actor_ids:
            selection_history_last5rounds[client_id].append(1)  # indicate this client has been selected
        for client_id in selection_history_last5rounds.keys():
            if client_id not in SBRO_selected_clients.actor_ids:
                selection_history_last5rounds[client_id].append(0)  # indicate this client has not been selected

        print(
            f"Selected clients for this round and flip ratio: { {id: nodes_label_flip_ratio[id] for id in SBRO_selected_clients.actor_ids} }")
        # Deploy the server model to the selected clients
        pool_SBRO.servers.map(deploy_server_model_pt, SBRO_selected_clients)
        # Each selected client trains her model
        SBRO_selected_clients.map(train, round=i)

        # Calculate Shapley values for each client
        shapley_values = exact_shapley_value(SBRO_selected_clients, acc_history_SBRO_list[-1], val_dataset, device)
        # print(shapley_values)
        # Update reputation and bid for each client
        update_reputation(Reputation_dict, Bid_dict, shapley_values,sv_record, exparam.ω, exparam.ψ, 5)

        for k, v in shapley_values.items():
           sv_record[k].append(v)


        # The aggregador collects weights from the selected clients and aggregates them
        pool_SBRO.aggregators.map(collect_clients_weights_pt, SBRO_selected_clients)
        pool_SBRO.aggregators.map(fed_avg)
        # The aggregator send its aggregated weights to the server
        pool_SBRO.aggregators.map(set_aggregated_weights_pt, pool_SBRO.servers)
        metrics = pool_SBRO.servers.map(evaluate_global_model)
        loss, acc = metrics[0]
        print(f"Server: Test acc: {acc:.4f}, test loss: {loss:.4f}")
        acc_history_SBRO_list.append(acc)

        # random select
        RS_clients_list = random_select_clients_within_budget(Bid_dict, Budget)
        RS_selectd_pool = pool_RS.clients.select(RS_clients_list)
        RS_selectd_clients = RS_selectd_pool.clients
        pool_RS.servers.map(deploy_server_model_pt, RS_selectd_clients)
        RS_selectd_clients.map(train, round=i)
        pool_RS.aggregators.map(collect_clients_weights_pt, RS_selectd_clients)
        pool_RS.aggregators.map(fed_avg)
        pool_RS.aggregators.map(set_aggregated_weights_pt, pool_RS.servers)
        metrics = pool_RS.servers.map(evaluate_global_model)
        loss, acc = metrics[0]
        print(f"Server: Random Selection Test acc: {acc:.4f}, test loss: {loss:.4f}")
        acc_history_RS_list.append(acc)

        # Select all clients
        pool_All.servers.map(deploy_server_model_pt, pool_All.clients)
        pool_All.clients.map(train, round=i)
        pool_All.aggregators.map(collect_clients_weights_pt, pool_All.clients)
        pool_All.aggregators.map(fed_avg)
        pool_All.aggregators.map(set_aggregated_weights_pt, pool_All.servers)
        metrics = pool_All.servers.map(evaluate_global_model)
        loss, acc = metrics[0]
        print(f"Server: All clients Test acc: {acc:.4f}, test loss: {loss:.4f}")
        acc_history_All_list.append(acc)


        # High-Quality Random Selection (HQRS)
        HQRS_selected_clients_list = random_select_clients_within_budget_in_normal(Bid_dict, Budget, label_flip_ratio_nodes[0])
        HQRS_selected_clients_pool = pool_HQRS.clients.select(HQRS_selected_clients_list)
        HQRS_selected_clients = HQRS_selected_clients_pool.clients
        pool_HQRS.servers.map(deploy_server_model_pt, HQRS_selected_clients)
        HQRS_selected_clients.map(train, round=i)
        pool_HQRS.aggregators.map(collect_clients_weights_pt, HQRS_selected_clients)
        pool_HQRS.aggregators.map(fed_avg)
        pool_HQRS.aggregators.map(set_aggregated_weights_pt, pool_HQRS.servers)
        metrics = pool_HQRS.servers.map(evaluate_global_model)
        loss, acc = metrics[0]
        print(f"Server: HQRS Test acc: {acc:.4f}, test loss: {loss:.4f}")
        acc_history_HQRS_list.append(acc)
        print('-------------------------------------------------------------------------')


        if (i + 1) % 20 == 0:
            data = {
                'Bid_dict': Bid_dict,
                'Reputation_dict': Reputation_dict,
                'nodes_label_flip_ratio': nodes_label_flip_ratio,
                'experiment_parameters': experiment_parameters,
                'acc_history_SBRO_list': acc_history_SBRO_list,
                'acc_history_RS_list': acc_history_RS_list,
                'acc_history_All_list': acc_history_All_list,
                'acc_history_HQRS_list': acc_history_HQRS_list,
                'SBRO_selected_client_id_list': SBRO_selected_client_id_list,
                'sv_record': sv_record
            }
            x_axis = range(len(acc_history_SBRO_list))
            plt.figure(figsize=(10, 6))
            plt.plot(x_axis, acc_history_SBRO_list, label='SBRO-FL')
            plt.plot(x_axis, acc_history_RS_list, label='RS-FL')
            plt.plot(x_axis, acc_history_All_list, label='All-FL')
            plt.plot(x_axis, acc_history_HQRS_list, label='HQRS-FL')
            plt.title('Federated Learning Accuracy per Round')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.legend()
            image_filename = str(i) + '.png'
            image_path = os.path.join(file_path, image_filename)
            plt.savefig(image_path)

            with open(os.path.join(file_path, 'result{}.json'.format(i)), 'w') as f:
                yaml.dump(data, f)

    data = {
        'Bid_dict': Bid_dict,
        'Reputation_dict': Reputation_dict,
        'nodes_label_flip_ratio': nodes_label_flip_ratio,
        'experiment_parameters': experiment_parameters,
        'acc_history_SBRO_list': acc_history_SBRO_list,
        'acc_history_RS_list': acc_history_RS_list,
        'acc_history_All_list': acc_history_All_list,
        'acc_history_HQRS_list': acc_history_HQRS_list,
        'SBRO_selected_client_id_list': SBRO_selected_client_id_list,
        'sv_record': sv_record
    }
    x_axis = range(len(acc_history_SBRO_list))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, acc_history_SBRO_list, label='SBRO-FL')
    plt.plot(x_axis, acc_history_RS_list, label='RS-FL')
    plt.plot(x_axis, acc_history_All_list, label='All-FL')
    plt.plot(x_axis, acc_history_HQRS_list, label='HQRS-FL')
    # Adding a title with custom font size
    plt.title(dataset_name, fontsize=20)  # Set the title font size to 20
    plt.xlabel('Round', fontsize=14)  # Set the x-axis label font size to 16
    plt.ylabel('Accuracy', fontsize=14)  # Set the label font size to 16
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Adjust the legend font size
    plt.legend(fontsize=16)  # Set the legend font size to 12

    image_filename = dataset_name + '.png'
    image_path = os.path.join(file_path, image_filename)
    plt.savefig(image_path)

    with open(os.path.join(file_path, 'result_final.json'), 'w') as f:
        yaml.dump(data, f)
train_n_rounds(exparam.n_rounds, clients_per_round=select_nodes)

