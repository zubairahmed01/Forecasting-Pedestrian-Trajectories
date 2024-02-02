import math
from os import listdir
from os.path import isfile, join
from tokenize import Double

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.loader as geometric_loader
from numpy import linalg as LA
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from scipy.spatial import distance
from torch.autograd import Variable
from torch.nn import Dropout, Linear, ReLU
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data as geom_data
from torch_geometric.nn import (GATConv, GATv2Conv, GCNConv, JumpingKnowledge,
                                NNConv, Sequential, global_mean_pool)
from torch_geometric.utils import (erdos_renyi_graph, from_networkx,
                                   get_self_loop_attr, to_networkx)


def seqToTrajectoryPoints(traj_in_seq):
    no_of_pedisterians = np.shape(traj_in_seq)[0]
    total_trajectory_points = np.shape(traj_in_seq)[2]

    ret = torch.zeros(total_trajectory_points*no_of_pedisterians, 2)
    index_count = 0
    for j in range(total_trajectory_points):
        for i in range(0, no_of_pedisterians):
            a = traj_in_seq[i:i+1, :, j]
            ret[index_count] = a
            index_count += 1

    return ret


def seqToNXGraph(traj_in_seq, traj_in_seq_relative, obsv_length=8, normal_weight=1):
    no_of_pedisterians = np.shape(traj_in_seq)[0]
    total_trajectory_points = np.shape(traj_in_seq)[2]
    G = nx.Graph()

    for j in range(total_trajectory_points):
        edges = []
        node = (j)
        current = node
        node_attribute = traj_in_seq[0, :, j]
        node_attribute_relative = traj_in_seq_relative[0, :, j]

        G.add_node(node, pos=(node_attribute_relative.squeeze()))
        edges.append((current, current, normal_weight))

        # primarry pedistarian
        if j > 0:
            previous = current - 1

            edges.append((previous, current, normal_weight))
            edges.append((current, previous, normal_weight))

            G.add_weighted_edges_from(edges)
            edges = []
        # other pedistarians in the scene
        for i in range(1, no_of_pedisterians):
            node_other_pedisterian = (j)+(i*total_trajectory_points)
            node_attribute_other_pedisterian = traj_in_seq[i:i+1, :, j]
            node_attribute_other_pedisterian_relative = traj_in_seq_relative[i:i+1, :, j]

            G.add_node(node_other_pedisterian, pos=(
                node_attribute_other_pedisterian_relative.squeeze()))
            edges.append((node_other_pedisterian, node_other_pedisterian, 1))

            if (node_other_pedisterian % obsv_length) > 0:
                previous_traj_point_list = (node_other_pedisterian - 1)
                edges.append((previous_traj_point_list,
                             node_other_pedisterian, normal_weight))
                edges.append(
                    (node_other_pedisterian, previous_traj_point_list, normal_weight))
                G.add_weighted_edges_from(edges)

            # add norm weighted edge between primary and other pedisterian
            norm_weight = getPedestrianDistance(
                node_attribute.squeeze(), node_attribute_other_pedisterian.squeeze())
            edges.append((node, node_other_pedisterian, norm_weight))
            edges.append((node_other_pedisterian, node, norm_weight))

            G.add_weighted_edges_from(edges)

        edges = []
    return G


def getPedestrianDistance(p1, p2):
    p1 = p1[:2]
    p2 = p2[:2]
    NORM = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    # NORM = distance.euclidean(p1, p2)
    # NORM = torch.cdist(p1, p2, p=0)

    # if NORM == 0:
    #     return 0
    # NORM = 1/(NORM)

    # return norm as a weight
    return (1 - NORM)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            if '\t' in line:
                delim = '\t'
            else:
                delim = ' '
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def make_social_gan_data(dataList):
    threshold = 0.002
    min_ped = 1
    max_peds_in_frame = 0
    obs_len = 8
    pred_len = 12
    skip = 1
    seq_len = obs_len + pred_len

    num_peds_in_seq = []
    seq_list = []
    seq_list_rel = []
    loss_mask_list = []
    non_linear_ped = []

    for data in dataList:
        frames = np.unique(data[:, 0]).tolist()
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])
        num_sequences = int(
            math.ceil((len(frames) - seq_len + 1) / skip))

        for idx in range(0, num_sequences * skip + 1, skip):
            curr_seq_data = np.concatenate(
                frame_data[idx:idx + seq_len], axis=0)
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            max_peds_in_frame = max(
                max_peds_in_frame, len(peds_in_curr_seq))
            curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                    seq_len))
            curr_seq = np.zeros((len(peds_in_curr_seq), 2, seq_len))
            curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                       seq_len))
            num_peds_considered = 0
            _non_linear_ped = []
            for _, ped_id in enumerate(peds_in_curr_seq):
                curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                             ped_id, :]
                curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                if pad_end - pad_front != seq_len:
                    continue
                curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                curr_ped_seq = curr_ped_seq
                # Make coordinates relative
                rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                rel_curr_ped_seq[:, 1:] = \
                    curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                _idx = num_peds_considered
                curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                # Linear vs Non-Linear Trajectory
                _non_linear_ped.append(
                    poly_fit(curr_ped_seq, pred_len, threshold))
                curr_loss_mask[_idx, pad_front:pad_end] = 1
                num_peds_considered += 1

            if num_peds_considered > min_ped:
                non_linear_ped += _non_linear_ped
                num_peds_in_seq.append(num_peds_considered)
                loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                seq_list.append(curr_seq[:num_peds_considered])
                seq_list_rel.append(curr_seq_rel[:num_peds_considered])

    seq_list = np.concatenate(seq_list, axis=0)
    seq_list_rel = np.concatenate(seq_list_rel, axis=0)
    loss_mask_list = np.concatenate(loss_mask_list, axis=0)
    non_linear_ped = np.asarray(non_linear_ped)

    # Convert numpy -> Torch Tensor
    obs_traj = torch.from_numpy(
        seq_list[:, :, :obs_len]).type(torch.float)
    pred_traj = torch.from_numpy(
        seq_list[:, :, obs_len:]).type(torch.float)
    obs_traj_rel = torch.from_numpy(
        seq_list_rel[:, :, :obs_len]).type(torch.float)
    pred_traj_rel = torch.from_numpy(
        seq_list_rel[:, :, obs_len:]).type(torch.float)
    loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
    non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)

    cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
    seq_start_end = [
        (start, end)
        for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]

    total_sequences = len(seq_list)

    out = [
        seq_start_end,
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        non_linear_ped,
        loss_mask,
        total_sequences
    ]

    return out


def create_normal_trajectory_data(dataList):
    trajectory_data = make_social_gan_data(dataList)

    seq_start_end, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask, _ = trajectory_data

    data_out = []
    for index in range(len(seq_start_end)):
        start, end = seq_start_end[index]
        obs_traj_data = obs_traj[start:end, :]
        obs_traj_rel_data = obs_traj_rel[start:end, :]

        pred_traj_data = pred_traj[start:end, :]
        pred_traj_rel_data = pred_traj_rel[start:end, :]
        non_linear_ped[start:end]
        loss_mask[start:end, :]

        data_out_tuple = (obs_traj_data, obs_traj_rel_data, pred_traj_data,
                          pred_traj_rel_data, non_linear_ped, loss_mask)
        data_out.append(data_out_tuple)

    return data_out


def create_graph_data(trajectory_data, limit_data_divider=1.0):
    seq_start_end, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask, _ = trajectory_data

    # pyG_Obsv_Data = []
    # pyG_Pred_Data = []
    graph_out = []
    limit_data = int(math.floor(len(seq_start_end)/limit_data_divider))
    for index in range(limit_data):
        start, end = seq_start_end[index]
        obs_traj_data = obs_traj[start:end, :]
        obs_traj_rel_data = obs_traj_rel[start:end, :]

        pred_traj_data = pred_traj[start:end, :]
        pred_traj_rel_data = pred_traj_rel[start:end, :]

        Gx = seqToNXGraph(traj_in_seq=obs_traj_data,
                          traj_in_seq_relative=obs_traj_rel_data)

        pyG_Obsv_Data = from_networkx(Gx, ['pos'], ['weight'])
        # pyG_Obsv_Data.append(dataPyG)

        Gy = seqToNXGraph(traj_in_seq=pred_traj_data,
                          traj_in_seq_relative=pred_traj_rel_data)

        pyG_Pred_Data = from_networkx(Gy, ['pos'], ['weight'])
        # pyG_Pred_Data.append(dataPyG)

        data_out = (pyG_Obsv_Data, pyG_Pred_Data)
        graph_out.append(data_out)

    return graph_out


def create_full_graph_data(dataList, limit_data_divider=1.0):
    # # pyG_Obsv_Data = []
    # # pyG_Pred_Data = []
    # graph_out = []
    # for index in range(len(seq_start_end)):
    #     start, end = seq_start_end[index]
    #     obs_traj_data = obs_traj[start:end, :]
    #     obs_traj_rel_data = obs_traj_rel[start:end, :]

    #     pred_traj_data = pred_traj[start:end, :]
    #     pred_traj_rel_data = pred_traj_rel[start:end, :]

    #     G = seqToNXGraph(traj_in_seq=obs_traj_data,
    #                      traj_in_seq_relative=obs_traj_rel_data)

    #     pyG_Obsv_Data = from_networkx(G, ['pos'], ['weight'])
    #     # pyG_Obsv_Data.append(dataPyG)

    #     G = seqToNXGraph(traj_in_seq=pred_traj_data,
    #                      traj_in_seq_relative=pred_traj_rel_data)

    #     pyG_Pred_Data = from_networkx(G, ['pos'], ['weight'])
    #     # pyG_Pred_Data.append(dataPyG)

    #     data_out = (pyG_Obsv_Data, pyG_Pred_Data)
    #     graph_out.append(data_out)

    # return graph_out

    data = make_social_gan_data(dataList)
    return create_graph_data(data, limit_data_divider)


# SOCIALWAYS GRAPH DATA

def add_nodes_to_graph(graph, trajectories, index, start, node_number, normal_weight=1):
    sub_batch_index_start = index*start
    no_of_pedisterians = np.shape(trajectories)[0]
    total_trajectory_points = np.shape(trajectories)[1]

    # list to add edges between the nodes of pedisterians (same time frame)
    ped_edge_nodes_list = [[] for x in range(total_trajectory_points)]

    # add nodes to the graph in same sequence as the data comes in
    for i in range(no_of_pedisterians):
        # ped_node = (sub_batch_index_start+1)*((i+1)*total_trajectory_points)
        # ped_node = ped_node - total_trajectory_points
        for j in range(total_trajectory_points):
            previous = None
            # node = ped_node+(j)
            node = node_number
            node_number = node_number + 1
            node_attribute = trajectories[i, j, :]
            if graph.has_node(node):
                print("Node:", node, "already exists in graph data")
                print(sub_batch_index_start, index, start, i, j)

            # to add conneciton between the nodes of pedisterians
            ped_edge_nodes_list[j].append(
                (node, node_attribute))

            node_attrib = (node_attribute)
            graph.add_node(node, obsv_4d=node_attrib)
            graph.add_weighted_edges_from([(node, node, normal_weight)])
            if j > 0:
                previous = node - 1
                edges = []
                edges.append((previous, node, normal_weight))
                edges.append((node, previous, normal_weight))
                graph.add_weighted_edges_from(edges)
                edges = []
            # print(node, (previous, node))

    for i in range(len(ped_edge_nodes_list)):
        edges = []
        nodes_of_ped = ped_edge_nodes_list[i]
        itr_length = len(nodes_of_ped)
        if itr_length == 1:
            continue
        for j in range(itr_length):
            (node, node_attrib) = nodes_of_ped[j]
            for k in range(itr_length):
                (other_node, other_node_attrib) = nodes_of_ped[k]
                if node == other_node:
                    # print("same nodes:", node, other_node)
                    continue

                calculated_weight = getPedestrianDistance(
                    node_attrib.squeeze(), other_node_attrib.squeeze())
                edges.append((node, other_node, calculated_weight))
                # print("added edges between nodes:", node, other_node)

        graph.add_weighted_edges_from(edges)
        edges = []

    return node_number


def create_socialways_graph_data(x, sub_batches=[]):
    length = len(sub_batches)
    Gx = nx.Graph()
    node_number = 0
    if x.shape[0] > 1 and len(sub_batches) >= 1:
        for index in range(length):
            start, end = sub_batches[index]
            traj_data = x[start:end, :]
            if np.shape(traj_data)[0] == 0:
                # handle this incase of tests have more than 1 trajectories: it fails for the original socialways code too, in attention pooling
                traj_data = x[:, :]
            node_number = add_nodes_to_graph(
                Gx, traj_data, index, start, node_number)

        # print("Total Nodes added:", node_number)
        graph_out = None
        # graph_out = from_networkx(Gx, ['obsv_4d'], ['weight'])
        graph_out = from_networkx(Gx, all, all)

        # new_x = graph_out.x.reshape(64, 8, 4)

        # print(x[:1,:,:])
        # print(new_x[:1,:,:])
        # print("printed")

        return graph_out
    else:
        # added for tests, as the batch size is generally 1
        node_number = add_nodes_to_graph(
            Gx, x, 0, 1, 0)
        graph_out = None
        graph_out = from_networkx(Gx, all, all)
        return graph_out


def create_networkx_graph_from(data):
    Gx = nx.Graph()
    add_nodes_to_graph(
        Gx, data, 0, 1, 0)
    return Gx


def main():
    print("Using torch", torch.__version__)
    gpu_avail = torch.cuda.is_available()
    print(f"Is the GPU available? {gpu_avail}")
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

    print("testing socialways data")
    data_set = './TRAJ/datasets/zara1/train/'
    dataFileList = [
        (data_set + file) for file in listdir(data_set) if isfile(join(data_set, file))]

    npDataList = [np.genfromtxt(file, delimiter='\t')
                  for file in dataFileList]
    limit_files = int(math.floor(len(npDataList)))
    return create_full_graph_data(npDataList[:limit_files])


if __name__ == "__main__":
    main()
