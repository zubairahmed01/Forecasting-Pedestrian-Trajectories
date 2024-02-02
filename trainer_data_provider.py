import argparse
import copy
import math
import os
from os import listdir
from os.path import isfile, join
import sys
import time
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from trainer_graph_data import create_socialways_graph_data, make_social_gan_data
from trainer_model import (AttentionPooling, DecoderFC, DecoderLstm,
                           Discriminator, EmbedSocialFeatures, EncoderLstm,
                           SocialFeatures, predict_cv)
from trainer_utils import get_traj_4d, time_print_util


class Scale(object):
    '''
    Given max and min of a rectangle it computes the scale and shift values to normalize data to [0,1]
    '''

    def __init__(self):
        self.min_x = +math.inf
        self.max_x = -math.inf
        self.min_y = +math.inf
        self.max_y = -math.inf
        self.sx, self.sy = 1, 1

    def calc_scale(self, keep_ratio=True):
        self.sx = 1 / (self.max_x - self.min_x)
        self.sy = 1 / (self.max_y - self.min_y)
        if keep_ratio:
            if self.sx > self.sy:
                self.sx = self.sy
            else:
                self.sy = self.sx

    def normalize(self, data, shift=True, inPlace=True):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        if data.ndim == 1:
            data_copy[0] = (data[0] - self.min_x * shift) * self.sx
            data_copy[1] = (data[1] - self.min_y * shift) * self.sy
        elif data.ndim == 2:
            data_copy[:, 0] = (data[:, 0] - self.min_x * shift) * self.sx
            data_copy[:, 1] = (data[:, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 3:
            data_copy[:, :, 0] = (data[:, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, 1] = (data[:, :, 1] - self.min_y * shift) * self.sy
        elif data.ndim == 4:
            data_copy[:, :, :, 0] = (
                data[:, :, :, 0] - self.min_x * shift) * self.sx
            data_copy[:, :, :, 1] = (
                data[:, :, :, 1] - self.min_y * shift) * self.sy
        else:
            return False
        return data_copy

    def denormalize(self, data, shift=True, inPlace=False):
        if inPlace:
            data_copy = data
        else:
            data_copy = np.copy(data)

        ndim = data.ndim
        if ndim == 1:
            data_copy[0] = data[0] / self.sx + self.min_x * shift
            data_copy[1] = data[1] / self.sy + self.min_y * shift
        elif ndim == 2:
            data_copy[:, 0] = data[:, 0] / self.sx + self.min_x * shift
            data_copy[:, 1] = data[:, 1] / self.sy + self.min_y * shift
        elif ndim == 3:
            data_copy[:, :, 0] = data[:, :, 0] / self.sx + self.min_x * shift
            data_copy[:, :, 1] = data[:, :, 1] / self.sy + self.min_y * shift
        elif ndim == 4:
            data_copy[:, :, :, 0] = data[:, :, :, 0] / \
                self.sx + self.min_x * shift
            data_copy[:, :, :, 1] = data[:, :, :, 1] / \
                self.sy + self.min_y * shift
        else:
            return False

        return data_copy


# Dataset - SocialWays Link: "https://www.dropbox.com/sh/lh1s41d1pqp8cbx/AAD4sB1JAiZIkCL7LHht-S4Ca" - ref "https://github.com/crowdbotp/socialways/issues/5"

def create_socialways_dataset(input_data, npz_out_file):
    parser = BIWIParser()
    parser.load(input_data)
    # parser = CustomDataParser()
    # parser.load(train_dataset_directory)

    obsvs, preds, times, batches = create_dataset(parser.p_data,
                                                  parser.t_data,
                                                  range(
                                                      parser.t_data[0][0], parser.t_data[-1][-1], parser.interval),
                                                  8, 12)
    os.makedirs(os.path.dirname(
        npz_out_file), exist_ok=True)
    np.savez(npz_out_file, obsvs=obsvs, preds=preds,
             times=times, batches=batches)


def create_dataset(p_data, t_data, t_range, n_past=8, n_next=12):
    dataset_t0 = []
    dataset_x = []
    dataset_y = []
    for t in range(t_range.start, t_range.stop, 1):
        for i in range(len(t_data)):
            t0_ind = (np.where(t_data[i] == t))[0]
            tP_ind = (np.where(t_data[i] == t - t_range.step * n_past))[0]
            tF_ind = (np.where(t_data[i] == t +
                      t_range.step * (n_next - 1)))[0]

            if t0_ind.shape[0] == 0 or tP_ind.shape[0] == 0 or tF_ind.shape[0] == 0:
                continue

            t0_ind = t0_ind[0]
            tP_ind = tP_ind[0]
            tF_ind = tF_ind[0]

            x_data = p_data[i][tP_ind:t0_ind]
            y_data = p_data[i][t0_ind:tF_ind + 1]

            if np.shape(x_data)[0] != n_past or np.shape(y_data)[0] != n_next:
                continue

            dataset_t0.append(t)
            dataset_x.append(x_data)
            dataset_y.append(y_data)

    sub_batches = []
    last_included_t = -1000
    min_interval = 1
    for i, t in enumerate(dataset_t0):
        if t > last_included_t + min_interval:
            sub_batches.append([i, i+1])
            last_included_t = t

        if t == last_included_t:
            sub_batches[-1][1] = i + 1

    sub_batches = np.array(sub_batches).astype(np.int16)
    dataset_x_ = []
    dataset_y_ = []
    last_ind = 0
    for ii, sb in enumerate(sub_batches):
        dataset_x_.append(dataset_x[sb[0]:sb[1]])
        dataset_y_.append(dataset_y[sb[0]:sb[1]])
        sb[1] = sb[1] - sb[0] + last_ind
        sb[0] = last_ind
        last_ind = sb[1]

    dataset_x = np.concatenate(dataset_x_)
    dataset_y = np.concatenate(dataset_y_)

    sub_batches = np.array(sub_batches).astype(np.int16)
    dataset_x = np.array(dataset_x).astype(np.float32)
    dataset_y = np.array(dataset_y).astype(np.float32)

    return dataset_x, dataset_y, dataset_t0, sub_batches


class BIWIParser:
    def __init__(self):
        self.scale = Scale()
        self.all_ids = list()
        self.delimit = ' '
        self.p_data = []
        self.v_data = []
        self.t_data = []
        self.min_t = int(sys.maxsize)
        self.max_t = -1
        self.interval = -1

    def load(self, filename, down_sample=1):
        pos_data_dict = dict()
        vel_data_dict = dict()
        time_data_dict = dict()
        self.all_ids.clear()

        time_index = 0
        frame_index = 1
        px_index = 2
        py_index = 4
        vx_index = 5
        vy_index = 7

        # to search for files in a folder?
        file_names = list()
        if '*' in filename:
            files_path = filename[:filename.index('*')]
            extension = filename[filename.index('*') + 1:]
            for file in os.listdir(files_path):
                if file.endswith(extension):
                    file_names.append(files_path + file)
        else:
            if os.path.isfile(filename):
                file_names.append(filename)
            else:
                all_files = os.listdir(filename)
                file_names = [os.path.join(filename, _path)
                              for _path in all_files]

        for file in file_names:
            if not os.path.exists(file):
                raise ValueError("No such file or directory:", file)
            with open(file, 'r') as data_file:
                content = data_file.readlines()
                id_list = list()
                for i, row in enumerate(content):
                    if '\t' in row:
                        self.delimit = '\t'
                    else:
                        self.delimit = ' '
                    row = row.strip().split(self.delimit)
                    while '' in row:
                        row.remove('')
                    if len(row) < 8 and len(row) != 4:
                        continue

                    if len(row) == 4:
                        px_index = 2
                        py_index = 3
                        vx_index = 2
                        vy_index = 3

                    ts = float(row[time_index])
                    id = round(float(row[frame_index]))
                    if ts % down_sample != 0:
                        continue
                    if ts < self.min_t:
                        self.min_t = ts
                    if ts > self.max_t:
                        self.max_t = ts

                    px = float(row[px_index])
                    py = float(row[py_index])
                    vx = float(row[vx_index])
                    vy = float(row[vy_index])

                    if id not in id_list:
                        id_list.append(id)
                        pos_data_dict[id] = list()
                        vel_data_dict[id] = list()
                        time_data_dict[id] = np.empty(0, dtype=int)
                        last_t = ts
                    pos_data_dict[id].append(np.array([px, py]))
                    vel_data_dict[id].append(np.array([vx, vy]))
                    time_data_dict[id] = np.hstack(
                        (time_data_dict[id], np.array([ts])))
            self.all_ids += id_list

        for ped_id, ped_T in time_data_dict.items():
            if len(ped_T) > 1:
                interval = int(round(ped_T[1] - ped_T[0]))
                if interval > 0:
                    self.interval = interval
                    break

        for key, value in pos_data_dict.items():
            poss_i = np.array(value)
            self.p_data.append(poss_i)
            # TODO: you can apply a Kalman filter/smoother on v_data
            vels_i = np.array(vel_data_dict[key])
            self.v_data.append(vels_i)
            self.t_data.append(np.array(time_data_dict[key]).astype(np.int32))

        # calc scale
        for i in range(len(self.p_data)):
            poss_i = np.array(self.p_data[i])
            self.scale.min_x = min(self.scale.min_x, min(poss_i[:, 0]))
            self.scale.max_x = max(self.scale.max_x, max(poss_i[:, 0]))
            self.scale.min_y = min(self.scale.min_y, min(poss_i[:, 1]))
            self.scale.max_y = max(self.scale.max_y, max(poss_i[:, 1]))
        self.scale.calc_scale()


def make_dataset(input_dataset_file, npz_out_file, device, leave_one_out=False, isTest=False):

    if not os.path.exists(npz_out_file):
        create_socialways_dataset(input_dataset_file, npz_out_file)

    print(os.path.dirname(os.path.realpath(__file__)))

    data = np.load(npz_out_file)
    # Data come as NxTx2 numpy nd-arrays where N is the number of trajectories,
    # T is their duration.
    dataset_obsv, dataset_pred, dataset_t, the_batches = data[
        'obsvs'], data['preds'], data['times'], data['batches']

    train_size = len(the_batches)
    if leave_one_out:
        # 4.5/5.0 (90%) of the batches to be used for validation
        train_size = max(1, int(math.floor((len(the_batches) * 4.5) // 5.0)))

        if isTest:
            # TODO: need to fix itt as it exccludes 1st baths for testing.
            # use all the batches for testing
            train_size = 1
    else:
        # 4/5 (80%) of the batches to be used for training
        train_size = max(1, (len(the_batches) * 4) // 5)

    train_batches = the_batches[:train_size]
    # Test batches are the remaining ones
    test_batches = the_batches[train_size:]
    # Size of the observed sub-paths
    n_past = dataset_obsv.shape[1]
    # Size of the sub-paths to predict
    n_next = dataset_pred.shape[1]
    # Number of training samples
    n_train_samples = the_batches[train_size - 1][1]
    # Number of testing samples (the remaining ones)
    n_test_samples = dataset_obsv.shape[0] - n_train_samples
    if n_test_samples == 0:
        n_test_samples = 1
        the_batches = np.array([the_batches[0], the_batches[0]])

    print(npz_out_file, ' # Training samples: ', n_train_samples)
    print(npz_out_file, ' # Test samples: ', n_test_samples)

    # Normalize the spatial data
    scale = Scale()
    scale.max_x = max(
        np.max(dataset_obsv[:, :, 0]), np.max(dataset_pred[:, :, 0]))
    scale.min_x = min(
        np.min(dataset_obsv[:, :, 0]), np.min(dataset_pred[:, :, 0]))
    scale.max_y = max(
        np.max(dataset_obsv[:, :, 1]), np.max(dataset_pred[:, :, 1]))
    scale.min_y = min(
        np.min(dataset_obsv[:, :, 1]), np.min(dataset_pred[:, :, 1]))
    scale.calc_scale(keep_ratio=True)
    dataset_obsv = scale.normalize(dataset_obsv)
    dataset_pred = scale.normalize(dataset_pred)

    # Copy normalized observations/paths to predict into torch GPU tensors
    dataset_obsv = torch.FloatTensor(dataset_obsv).to(device)
    dataset_pred = torch.FloatTensor(dataset_pred).to(device)

    return dataset_obsv, dataset_pred, dataset_t, the_batches, train_batches, test_batches, train_size, n_train_samples, n_test_samples, scale


def create_obsv_data_graph(x, sub_batches):
    graph_batch_data = create_socialways_graph_data(x, sub_batches)
    return graph_batch_data


def create_batched_dataset(obsvData, predData, seqStartEndData, total_batches, train_size, batch_size=64, isTrain=True, device=None, use_graph=True):
    start_time = time.process_time()
    sceneDataList: list = []
    batch_size_accum = 0
    sub_batches = []

    if not isTrain:
        for ii, batch_i in enumerate(seqStartEndData):
            obsv = obsvData[batch_i[0]:batch_i[1]]
            pred = predData[batch_i[0]:batch_i[1]]
            bs = int(batch_i[1] - batch_i[0])

            obsv_4d, pred_4d = get_traj_4d(obsv, pred)

            obsv_graph = create_obsv_data_graph(
                x=obsv_4d, sub_batches=sub_batches)
            obsv_graph = obsv_graph.to(device)

            batchedTrajectories = (
                obsv, pred, obsv_4d, pred_4d, obsv_graph, bs, sub_batches)

            sceneDataList.append(batchedTrajectories)

        end_time = time.process_time()
        time_print_util('Batched Test Dataset Creation time:',
                        start_time, end_time)
        return sceneDataList

    # For all the training batches
    for ii, batch_i in enumerate(seqStartEndData):
        batch_size_accum += batch_i[1] - batch_i[0]
        sub_batches.append(batch_i)

        # FIXME: Just keep it for toy dataset
        # sub_batches = the_batches
        # batch_size_accum = sub_batches[-1][1]
        # ii = train_size-1

        if ii >= train_size - 1 or \
                batch_size_accum + (total_batches[ii + 1][1] - total_batches[ii + 1][0]) > batch_size:
            # Observed partial paths
            obsv = obsvData[sub_batches[0][0]:sub_batches[-1][1]]
            # Future partial paths
            pred = predData[sub_batches[0][0]:sub_batches[-1][1]]
            sub_batches = sub_batches - sub_batches[0][0]
            # May have to fill with 0
            filling_len = batch_size - int(batch_size_accum)
            # obsv = torch.cat((obsv, torch.zeros(filling_len, n_past, 2).to(device)), dim=0)
            # pred = torch.cat((pred, torch.zeros(filling_len, n_next, 2).to(device)), dim=0)

            bs = batch_size_accum

            obsv_4d, pred_4d = get_traj_4d(obsv, pred)

            obsv_graph = None
            if use_graph:
                obsv_graph = create_obsv_data_graph(
                    x=obsv_4d, sub_batches=sub_batches)
                obsv_graph = obsv_graph.to(device)

            batchedTrajectories = (
                obsv, pred, obsv_4d, pred_4d, obsv_graph, bs, sub_batches)

            sceneDataList.append(batchedTrajectories)

            batch_size_accum = 0
            sub_batches = []

    end_time = time.process_time()
    time_print_util('Batched Train Dataset Creation time:',
                    start_time, end_time)
    return sceneDataList


def get_train_test_data(dataset_name, device, batch_size=64, use_graph=True, leave_one_out=False, isTest=False):
    total_data_processing_start_time = time.process_time()
    ret_train_data_list: list = []
    ret_test_data_list: list = []

    input_dataset_files, npz_out_file = getDatasetFiles(
        dataset_name, isLeaveOneOut=leave_one_out, isTest=isTest)

    original_npz_out = npz_out_file
    if len(input_dataset_files) > 1:
        npz_out_file = npz_out_file.replace('.npz', '')

    for count_, input_dataset_file in enumerate(input_dataset_files):

        if '.npz' not in npz_out_file:
            npz_out_file = npz_out_file + '-' + str(count_) + '.npz'

        dataset_obsv, dataset_pred, dataset_t, total_batches, train_batches, test_batches, train_size, n_train_samples, n_test_samples, scale = make_dataset(
            input_dataset_file, npz_out_file, device, leave_one_out=leave_one_out, isTest=isTest)

        if not isTest:
            train_data_list: list = create_batched_dataset(
                dataset_obsv, dataset_pred, train_batches, total_batches, train_size, device=device, batch_size=batch_size, use_graph=use_graph)
            ret_train_data_list.append(
                (train_data_list, n_train_samples, scale))

        test_data_list: list = create_batched_dataset(
            dataset_obsv, dataset_pred,  test_batches, total_batches, train_size, isTrain=False, device=device, batch_size=batch_size, use_graph=use_graph)
        ret_test_data_list.append((test_data_list, n_test_samples, scale))

        npz_out_file = original_npz_out
        if len(input_dataset_files) > 1:
            npz_out_file = npz_out_file.replace('.npz', '')

    total_data_processing_end_time = time.process_time()
    time_print_util('Total Data Processing Time:',
                    total_data_processing_start_time, total_data_processing_end_time)

    return (ret_train_data_list, ret_test_data_list)


def make_complete_socialways_train_test_dataset(input_directory='', batch_size=64):
    socialways_dataset_directory = './TRAJ/raw/traj-datasets-socialways'
    data_file_name = ['obsmat.txt', 'obsmat_px.txt']
    socialways_out_data_directory = './TRAJ/processed/SocialWays-Data'

    os.makedirs(os.path.dirname(
        socialways_out_data_directory), exist_ok=True)

    # first make all the individual datasets
    subfolders = [x.path for x in os.scandir(
        socialways_dataset_directory) if x.is_dir()]

    for data_folder in subfolders:
        data_file_path = None
        data_out_file = socialways_out_data_directory + '/'
        if 'st3' in data_folder:
            data_file_path = data_folder + '/' + data_file_name[1]
        else:
            data_file_path = data_folder + '/' + data_file_name[0]

        if 'eth' in data_folder:
            data_out_file = data_out_file + 'eth'
        elif 'hotel' in data_folder:
            data_out_file = data_out_file + 'hotel'
        elif 'zara01' in data_folder:
            data_out_file = data_out_file + 'zara01'
        elif 'zara02' in data_folder:
            data_out_file = data_out_file + 'zara02'
        elif 'st3' in data_folder:
            data_out_file = data_out_file + 'univ'

        data_out_file = data_out_file + '.npz'
        if not os.path.exists(data_out_file):
            create_socialways_dataset(data_file_path, data_out_file)


def make_complete_socialgan_train_test_dataset(input_directory='', batch_size=64):
    socialways_dataset_directory = './TRAJ/raw/socialGAN_datasets/'
    directories = ['/train/', '/test/']
    socialgan_out_data_directory = './TRAJ/processed/SocialGAN-Data/'

    os.makedirs(os.path.dirname(
        socialgan_out_data_directory), exist_ok=True)

    # first make all the individual datasets
    subfolders = [x.path for x in os.scandir(
        socialways_dataset_directory) if x.is_dir()]

    for data_folder in subfolders:
        if 'socialGAN_datasets/raw' in data_folder:
            continue

        dataset_name = ''

        if 'eth' in data_folder:
            dataset_name = 'eth'
        elif 'hotel' in data_folder:
            dataset_name = 'hotel'
        elif 'univ' in data_folder:
            dataset_name = 'univ'
        elif 'zara1' in data_folder:
            dataset_name = 'zara01'
        elif 'zara2' in data_folder:
            dataset_name = 'zara02'

        for directory in directories:
            data_file_path = data_folder + directory
            data_out_file = socialgan_out_data_directory + \
                dataset_name + '/' + directory.strip('/')
            data_out_file = data_out_file + '.npz'
            if not os.path.exists(data_out_file):
                create_socialways_dataset(data_file_path, data_out_file)


def get_social_ways_data(dataset_name, device, scale_data=True, batch_size=64, use_graph=True):
    socialways_out_data_directory = './TRAJ/processed/SocialWays-Data'

    file_names = ['eth', 'hotel', 'zara01', 'zara02', 'univ']

    if dataset_name not in file_names:
        sys.exit('Please provide correct dataset_name:')

    training_data = []
    test_data = []

    os.makedirs(os.path.dirname(
        socialways_out_data_directory), exist_ok=True)
    subfolders = [x.path for x in os.scandir(
        socialways_out_data_directory) if not x.is_dir()]

    for data_file in subfolders:
        print('Data Loading from file:', data_file)

        if 'univ' in data_file:
            continue

        loaded_data = np.load(data_file)
        dataset_obsv, dataset_pred, dataset_t, the_batches = loaded_data[
            'obsvs'], loaded_data['preds'], loaded_data['times'], loaded_data['batches']

        scale = Scale()
        if scale_data:
            # Normalize the spatial data
            scale.max_x = max(
                np.max(dataset_obsv[:, :, 0]), np.max(dataset_pred[:, :, 0]))
            scale.min_x = min(
                np.min(dataset_obsv[:, :, 0]), np.min(dataset_pred[:, :, 0]))
            scale.max_y = max(
                np.max(dataset_obsv[:, :, 1]), np.max(dataset_pred[:, :, 1]))
            scale.min_y = min(
                np.min(dataset_obsv[:, :, 1]), np.min(dataset_pred[:, :, 1]))
            scale.calc_scale(keep_ratio=True)
            dataset_obsv = scale.normalize(dataset_obsv)
            dataset_pred = scale.normalize(dataset_pred)

        # Copy normalized observations/paths to predict into torch GPU tensors
        dataset_obsv = torch.FloatTensor(dataset_obsv).to(device)
        dataset_pred = torch.FloatTensor(dataset_pred).to(device)

        if dataset_name in data_file:
            test_size = len(the_batches)
            test_data_list: list = create_batched_dataset(
                dataset_obsv, dataset_pred, the_batches, the_batches, test_size, isTrain=False, device=device, batch_size=batch_size, use_graph=use_graph)
            test_data.append((test_data_list, dataset_obsv.shape[0], scale))

            # test_data.append(
            #     (dataset_obsv, dataset_pred, dataset_t, the_batches, scale))
        else:
            train_size = len(the_batches)
            train_data_list: list = create_batched_dataset(
                dataset_obsv, dataset_pred, the_batches, the_batches, train_size, device=device, batch_size=batch_size, use_graph=use_graph)
            training_data.append(
                (train_data_list, dataset_obsv.shape[0], scale))

            # training_data.append((dataset_obsv, dataset_pred,
            #                       dataset_t, the_batches, scale))

    return (training_data, test_data)


def get_social_gan_data(dataset_name, device, scale_data=True, batch_size=64, use_graph=True):
    socialgan_out_data_directory = './TRAJ/processed/SocialGAN-Data/'
    data_names = ['/train.npz', '/test.npz']
    file_names = ['eth', 'hotel', 'zara01', 'zara02', 'univ']

    if dataset_name not in file_names:
        sys.exit('Please provide correct dataset_name:')

    training_data = []
    test_data = []
    os.makedirs(os.path.dirname(
        socialgan_out_data_directory), exist_ok=True)
    subfolders = [x.path for x in os.scandir(
        socialgan_out_data_directory) if x.is_dir()]

    for data_folder in subfolders:
        if dataset_name not in data_folder:
            continue
        for data_file_ in data_names:
            data_file = data_folder + data_file_
            print('Data Loading from file:', data_file)

            loaded_data = np.load(data_file)
            dataset_obsv, dataset_pred, dataset_t, the_batches = loaded_data[
                'obsvs'], loaded_data['preds'], loaded_data['times'], loaded_data['batches']

            scale = Scale()
            if scale_data:
                # Normalize the spatial data
                scale.max_x = max(
                    np.max(dataset_obsv[:, :, 0]), np.max(dataset_pred[:, :, 0]))
                scale.min_x = min(
                    np.min(dataset_obsv[:, :, 0]), np.min(dataset_pred[:, :, 0]))
                scale.max_y = max(
                    np.max(dataset_obsv[:, :, 1]), np.max(dataset_pred[:, :, 1]))
                scale.min_y = min(
                    np.min(dataset_obsv[:, :, 1]), np.min(dataset_pred[:, :, 1]))
                scale.calc_scale(keep_ratio=True)
                dataset_obsv = scale.normalize(dataset_obsv)
                dataset_pred = scale.normalize(dataset_pred)

            # Copy normalized observations/paths to predict into torch GPU tensors
            dataset_obsv = torch.FloatTensor(dataset_obsv).to(device)
            dataset_pred = torch.FloatTensor(dataset_pred).to(device)

            if 'test' in data_file:
                test_size = len(the_batches)
                test_data_list: list = create_batched_dataset(
                    dataset_obsv, dataset_pred, the_batches, the_batches, test_size, isTrain=False, device=device, batch_size=batch_size, use_graph=use_graph)
                test_data.append(
                    (test_data_list, dataset_obsv.shape[0], scale))

                # test_data.append(
                #     (dataset_obsv, dataset_pred, dataset_t, the_batches, scale))
            elif 'train' in data_file:
                train_size = len(the_batches)
                train_data_list: list = create_batched_dataset(
                    dataset_obsv, dataset_pred, the_batches, the_batches, train_size, device=device, batch_size=batch_size, use_graph=use_graph)
                training_data.append(
                    (train_data_list, dataset_obsv.shape[0], scale))

                # training_data.append((dataset_obsv, dataset_pred,
                #                       dataset_t, the_batches, scale))

    return (training_data, test_data)


def get_training_data_for_test(dataset_name, dataset, device, scale_data=True, batch_size=64, parsesr='socialWays', use_graph=True):
    if 'GAN' in parsesr:
        return testSocialGANParsing(device)

    if 'socialWays' not in dataset:
        make_complete_socialgan_train_test_dataset()
        return get_social_gan_data(dataset_name, device, scale_data, batch_size, use_graph=use_graph)

    make_complete_socialways_train_test_dataset()
    return get_social_ways_data(dataset_name, device, scale_data, batch_size, use_graph=use_graph)


def testSocialGANParsing(device, batch_size=64):
    print("testing socialways data")
    data_set_dir = ['./TRAJ/datasets/zara1/train/',
                    './TRAJ/datasets/zara1/val/']

    training_data = []
    test_data = []
    for data_set in data_set_dir:
        dataFileList = [
            (data_set + file) for file in listdir(data_set) if isfile(join(data_set, file))]

        npDataList = [np.genfromtxt(file, delimiter='\t')
                      for file in dataFileList]

        trajectory_data = make_social_gan_data(npDataList)

        seq_start_end, obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped, loss_mask, total_sequences = trajectory_data

        # dataset_obsv = torch.FloatTensor(dataset_obsv).to(device)
        # dataset_pred = torch.FloatTensor(dataset_pred).to(device)
        obs_traj = obs_traj.permute(0, 2, 1).to(device)
        pred_traj = pred_traj.permute(0, 2, 1).to(device)
        obs_traj_rel = obs_traj_rel.permute(0, 2, 1).to(device)
        pred_traj_rel = pred_traj_rel.permute(0, 2, 1).to(device)

        if 'val' in data_set:
            total_size = len(seq_start_end)
            the_batches = np.asarray(seq_start_end)
            data_list: list = create_batched_dataset(
                obs_traj, pred_traj, the_batches, the_batches, total_size, isTrain=False, device=device, batch_size=batch_size)
            scale = None
            test_data.append((data_list, obs_traj.shape[0], scale))
        elif 'train' in data_set:
            total_size = len(seq_start_end)
            the_batches = np.asarray(seq_start_end)
            data_list: list = create_batched_dataset(
                obs_traj, pred_traj, the_batches, the_batches, total_size, isTrain=True, device=device, batch_size=batch_size)
            scale = None
            training_data.append((data_list, obs_traj.shape[0], scale))

    return (training_data, test_data)


def getDatasetFiles(dataset, isLeaveOneOut=False, isTest=False):
    base_input_path = './TRAJ/raw/traj-datasets-socialways/'
    base_input_path2 = './TRAJ/raw/socialGAN_datasets/raw/all_data/'
    base_save_path = "./TRAJ/processed/socialWays/"

    datasetFileNamesList: dict = {'eth': 'eth-8-12.npz',
                                  'hotel': 'hotel-8-12.npz',
                                  'univ': 'univ-8-12.npz',
                                  'zara01': 'zara01-8-12.npz',
                                  'zara02': 'zara02-8-12.npz'}

    inputDataFileList: dict = {'eth': [base_input_path + 'seq_eth/obsmat.txt'],
                               'hotel': [base_input_path + 'seq_hotel/obsmat.txt'],
                               'univ': [base_input_path2 + 'students001.txt', base_input_path2 + 'students003.txt', base_input_path2 + 'uni_examples.txt'],
                               'zara01': [base_input_path + 'data_zara01/obsmat.txt'],
                               'zara02': [base_input_path + 'data_zara02/obsmat.txt']}

    npz_out_file = base_save_path + datasetFileNamesList[dataset]
    input_data_file_list = inputDataFileList[dataset]
    if isLeaveOneOut:
        datasetFileNamesList = {
            'eth': {
                'test': 'leave-out-test-eth-8-12.npz',
                'train': 'leave-out-train-eth-8-12.npz',
            },
            'hotel': {
                'test': 'leave-out-test-hotel-8-12.npz',
                'train': 'leave-out-train-hotel-8-12.npz',
            },
            'univ': {
                'test': 'leave-out-test-univ-8-12.npz',
                'train': 'leave-out-train-univ-8-12.npz',
            },
            'zara01': {
                'test': 'leave-out-test-zara01-8-12.npz',
                'train': 'leave-out-ttrain-zara01-8-12.npz',
            },
            'zara02': {
                'test': 'leave-out-test-zara02-8-12.npz',
                'train': 'leave-out-train-zara02-8-12.npz',
            }
        }

        inputDataFileList = {
            'eth': {
                'test': [base_input_path + 'seq_eth/obsmat.txt'],
                'train': [base_input_path + 'seq_hotel/obsmat.txt',
                          base_input_path2 + 'students001.txt',
                          base_input_path2 + 'students003.txt',
                          base_input_path2 + 'uni_examples.txt',
                          base_input_path + 'data_zara01/obsmat.txt',
                          base_input_path + 'data_zara02/obsmat.txt'],
            },
            'hotel': {
                'test': [base_input_path + 'seq_hotel/obsmat.txt'],
                'train': [base_input_path + 'seq_eth/obsmat.txt',
                          base_input_path2 + 'students001.txt',
                          base_input_path2 + 'students003.txt',
                          base_input_path2 + 'uni_examples.txt',
                          base_input_path + 'data_zara01/obsmat.txt',
                          base_input_path + 'data_zara02/obsmat.txt'],
            },
            'univ': {
                'test': [base_input_path2 + 'students001.txt',
                         base_input_path2 + 'students003.txt'],
                'train': [base_input_path + 'seq_eth/obsmat.txt',
                          base_input_path + 'seq_hotel/obsmat.txt',
                          base_input_path2 + 'uni_examples.txt',
                          base_input_path + 'data_zara01/obsmat.txt',
                          base_input_path + 'data_zara02/obsmat.txt'],
            },
            'zara01': {
                'test': [base_input_path + 'data_zara01/obsmat.txt'],
                'train': [base_input_path + 'seq_eth/obsmat.txt',
                          base_input_path + 'seq_hotel/obsmat.txt',
                          base_input_path2 + 'students001.txt',
                          base_input_path2 + 'students003.txt',
                          base_input_path2 + 'uni_examples.txt',
                          base_input_path + 'data_zara02/obsmat.txt'],
            },
            'zara02': {
                'test': [base_input_path + 'data_zara02/obsmat.txt'],
                'train': [base_input_path + 'seq_eth/obsmat.txt',
                          base_input_path + 'seq_hotel/obsmat.txt',
                          base_input_path2 + 'students001.txt',
                          base_input_path2 + 'students003.txt',
                          base_input_path2 + 'uni_examples.txt',
                          base_input_path + 'data_zara01/obsmat.txt']
            }
        }
        key = 'test' if isTest else 'train'
        npz_out_file = base_save_path + datasetFileNamesList[dataset][key]
        input_data_file_list = inputDataFileList[dataset][key]

    return input_data_file_list, npz_out_file


def main():
    print("Using torch", torch.__version__)
    gpu_avail = torch.cuda.is_available()
    print(f"Is the GPU available? {gpu_avail}")
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

    # get_training_data_for_test('eth', 'socialGAN', device, parsesr='SocialGAN')

    dataset_name = 'univ'

    batch_size: int = 64
    use_graph: bool = False

    n_train_samples = 0
    n_test_samples = 0

    train_data_list_of_list, test_data_list_of_list = get_train_test_data(
        dataset_name, device, batch_size=batch_size, use_graph=use_graph, leave_one_out=True, isTest=True)

    for count_, data_from_list in enumerate(train_data_list_of_list):
        train_data_list, train_size, scale = data_from_list
        n_train_samples = n_train_samples + train_size

    for count_, data_from_list in enumerate(test_data_list_of_list):
        train_data_list, test_size, scale = data_from_list
        n_test_samples = n_test_samples + test_size

    print("Total Train Size:", n_train_samples)
    print("Total Test Size:", n_test_samples)


if __name__ == "__main__":
    main()
