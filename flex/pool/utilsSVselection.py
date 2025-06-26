import math
import pulp as pp
import torch
import numpy as np
import itertools
from flex.pool import collect_clients_weights_pt
from flex.pool import fed_avg
from torch.utils.data import DataLoader
from math import factorial
import copy
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque
def debug_plot(y_dict, name, highlight_ids=[]):
    ids = range(len(y_dict))
    y = [y_dict[i] for i in range(len(y_dict))]

    plt.figure(figsize=(10, 6))

    # 绘制默认颜色的点
    for i in ids:
        if i in highlight_ids:
            plt.scatter(i, y[i], color='red', label='highlight' if i == highlight_ids[0] else "")
        else:
            plt.scatter(i, y[i], color='blue', label='no sorted' if i == 0 else "")

    plt.title(name)
    plt.xlabel('id')
    plt.ylabel(name)
    plt.legend()
    plt.show()

def prospect_theory(Repuation_dict,γ=2.5, α=0.3, β=0.9):
    values = [v[-1] for v in Repuation_dict.values()]
    R0 = sum(values) / len(values)
    result = {}
    for key, R_list in Repuation_dict.items():
        R = R_list[-1]
        z = R - R0
        if z >= 0:
            result[key] = z ** α
        else:
            result[key] = -γ * ((-z) ** β)
    # debug_plot(result,'prospect_theory')
    # _ = result
    min_value = min(result.values())
    result = {key: value - min_value for key, value in result.items()}
    # debug_plot(_,'softmax')

    return result

def exact_shapley_value(clients, baseline_performance, val_data, device,*args, **kwargs):
    """
    Calculate exact Shapley values for each client in federated learning.

    Parameters:
    -----------
    clients : FlexPool
        FlexPool containing client models.
    baseline_performance : float
        The performance of the global model without any client.
    val_data : Dataset
        The validation dataset to evaluate the model.
    device : str
        The device to use for the model.

    Returns:
    --------
    shapley_values : dict
        Dictionary of Shapley values for each client.
    """
    n = len(clients._models)
    shapley_values = {client_id: 0 for client_id in clients._models}
    cache = {}
    ignore_weights = kwargs.get("ignore_weights", None)
    if ignore_weights is None:
        ignore_weights = ["num_batches_tracked"]
    def get_subset_value(subset):
        """
        Get the performance value of a subset of clients.

        Parameters:
        -----------
        subset : list
            List of client IDs representing the subset.

        Returns:
        --------
        float
            The performance value of the subset.
        """
        subset_key = tuple(sorted(subset))
        if subset_key not in cache:
            if len(subset) == 0:
                subset_acc = baseline_performance
            else:
                subset_params = [collect_clients_weights_pt.__wrapped__(clients._models[client_id]) for client_id in subset]
                aggregated_params = fed_avg.__wrapped__(subset_params)
                model = copy.deepcopy(clients._models[subset[0]]['model'])
                weight_dict = model.state_dict()
                for layer_key, new in zip(weight_dict, aggregated_params):
                    try:
                        if len(new) != 0:  # Do not copy empty layers
                            weight_dict[layer_key].copy_(new)
                    except TypeError:  # new has no len property
                        weight_dict[layer_key].copy_(new)

                model.eval()
                test_acc = 0
                total_count = 0
                model = model.to(device)
                # get test data as a torchvision object
                test_dataloader = DataLoader(
                    val_data, batch_size=256, shuffle=False, pin_memory=True
                )
                losses = []
                with torch.no_grad():
                    for data, target in test_dataloader:
                        total_count += target.size(0)
                        data, target = data.to(device), target.to(device)
                        output = model(data) 
                        pred = output.data.max(1, keepdim=True)[1]
                        test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()
                test_acc /= total_count
                subset_acc = test_acc
            cache[subset_key] = subset_acc
        return cache[subset_key]

    for client in clients._models:
        for k in range(n):
            subsets = itertools.combinations([c for c in clients._models if c != client], k)
            for subset in subsets:
                subset = list(subset)
                subset_with_client = subset + [client]

                subset_acc_with_client = get_subset_value(subset_with_client)
                subset_acc_without_client = get_subset_value(subset)

                marginal_contribution = subset_acc_with_client - subset_acc_without_client
                shapley_values[client] += (marginal_contribution * factorial(k) * factorial(n - k - 1)) / factorial(n)

    return shapley_values

def update_reputation(Repuation_dict, Bid_dict, shapley_values,sv_history, ω, ψ, penalty_growth_rate=1.5):
    """
    Update reputation of nodes based on Shapley values, bid prices, and performance history.

    Parameters:
    -----------
    Repuation_dict : dict
        Dictionary of reputation values for each node.
    Bid_dict : dict
        Dictionary of bid values for each node.
    shapley_values : dict
        Dictionary of Shapley values for each node.
    poorperformanceCount : dict
        Dictionary of poor performance count for each node.
    ω : float
        Reward coefficient for positive contributions.
    ψ : float
        Punishment coefficient for negative contributions (should be negative).
    round: int
        Current round number.

    """
    sum_positive_sv = sum(sv for sv in shapley_values.values() if sv > 0)
    sum_positive_bid = sum(Bid_dict[node_id] for node_id, sv in shapley_values.items() if sv > 0)
    for node_id, sv in shapley_values.items():
        if sv <= 0:
            recent_values = sv_history[node_id][-5:]
            # 计算小于0的次数
            recent_errors = sum(1 for v in recent_values if v < 0)
            # Update bad count
            UD = -ψ * (penalty_growth_rate ** recent_errors)
        else:
            relative_positive_contribution = sv / sum_positive_sv if sum_positive_sv != 0 else 0
            relative_bid = Bid_dict[node_id] / sum_positive_bid if sum_positive_bid != 0 else 0
            exp_term = -relative_positive_contribution / relative_bid
            UD = ω * (1 - np.exp(exp_term))
        # Update reputation
        if node_id in Repuation_dict:
            Repuation_dict[node_id].append(Repuation_dict[node_id][-1] + UD)

def select_nodes(Reputation_dict, Bid_dict, Budget, γ=-2.5, α=0.3, β=0.9, round=1, participation_count_5rounds={}):
    #   convert the reputation values to weights using the prospect theory function
    W = prospect_theory(Reputation_dict, γ, α, β)

    #  calculate the weight threshold
    sorted_data = sorted([v for v in W.values()])
    weight_threshold = np.percentile(sorted_data, 50)

    #  adjust the weights based on the number of times a node has been selected in the last five rounds
    for node_id, count in participation_count_5rounds.items():
        #
        decay_factor = 0.5 ** count
        W[node_id] *= decay_factor

    #   create an optimization problem
    m = pp.LpProblem(sense=pp.LpMaximize)
    x = {node_id: pp.LpVariable(f'x{node_id}', cat='Binary') for node_id in Reputation_dict.keys()}

    #   add the objective function
    epsilon = 1e-6  #
    m += pp.lpSum([(W[node_id] + epsilon) * x[node_id] for node_id in Reputation_dict.keys()])

    #  add the budget constraint
    m += pp.lpSum([Bid_dict[node_id] * x[node_id] for node_id in Reputation_dict.keys()]) <= Budget

    #  add the weight threshold constraint
    for node_id in Reputation_dict.keys():
        m += W[node_id] * x[node_id] >= weight_threshold * x[node_id]

    #  solve the problem
    m.solve(pp.PULP_CBC_CMD(msg=False))

    # get the selected node IDs
    selected_nodes = [node_id for node_id in Reputation_dict.keys() if pp.value(x[node_id]) == 1]

    return selected_nodes

def random_select_clients_within_budget(Bid_dict, Budget):
    """
    Select clients based on their bids and a budget constraint.
    Bid_dict : dict
        Dictionary of bid values for each client.
    Budget : int or float
        The budget for client selection.
    Returns:
    --------
    selected_clients : list
        List of selected client IDs.

    """
    selected_clients = []
    total_cost = 0
    client_ids = list(Bid_dict.keys())

    random.shuffle(client_ids)

    for client_id in client_ids:
        bid = Bid_dict[client_id]
        if total_cost + bid <= Budget:
            selected_clients.append(client_id)
            total_cost += bid
        else:
            break

    return selected_clients

def random_select_clients_within_budget_in_normal(Bid_dict, Budget, good_list):
    """
    Select clients based on their bids and a budget constraint.
    Bid_dict : dict
        Dictionary of bid values for each client.
    Budget : int or float
        The budget for client selection.
    Returns:
    --------
    selected_clients : list
        List of selected client IDs.

    """
    selected_clients = []
    total_cost = 0

    random.shuffle(good_list)

    for client_id in good_list:
        bid = Bid_dict[client_id]
        if total_cost + bid <= Budget:
            selected_clients.append(client_id)
            total_cost += bid
        else:
            break

    return selected_clients