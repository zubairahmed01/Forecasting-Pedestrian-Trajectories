import argparse
import copy
import math
import os
import platform
import pprint
import sys
import time
from datetime import datetime
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.autograd import Variable
from tqdm import trange

from trainer_data_provider import get_train_test_data, get_training_data_for_test
from trainer_model import (
    AttentionPooling,
    DecoderFC,
    DecoderLstm,
    Discriminator,
    EmbedSocialFeatures,
    EncoderLstm,
    SocialFeatures,
    predict_cv,
)
from trainer_utils import (
    calculateAngleFor,
    calculateShiftFor,
    calculateSlopeFor,
    calculateVelocityFor,
    get_actual_noise,
    get_categ_and_cont_codes,
    get_formatted_time,
    get_latent_codes_from_noise,
    get_model_file_name_from,
    str2bool,
    time_print_util,
)

print("Using torch", torch.__version__)
gpu_avail = torch.cuda.is_available()
print(f"Is the GPU available? {gpu_avail}")
if gpu_avail:
    device = torch.device("cuda")
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.device(current_device_index)
    current_device_name = torch.cuda.get_device_name(current_device_index)
    print(
        f"Device: {device} - index: {current_device_index} - name: {current_device_name}")
else:
    device = torch.device("cpu")
    print("Device", device)

current_time = time.time()
time_struct = time.localtime(current_time)
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
# print(formatted_time)

run_date_time = ''


# Parser arguments
parser = argparse.ArgumentParser(
    description='Social Ways (Graph) trajectory prediction.')

parser.add_argument('--epochs', '--e',
                    type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--dataset', '--data',
                    default='hotel',
                    choices=['eth', 'hotel', 'univ', 'zara01', 'zara02'],
                    help='pick a specific dataset (default: "hotel")')
parser.add_argument('--batch-size', '--b',
                    type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--latent-dim', '--ld',
                    type=int, default=10, metavar='N',
                    help='dimension of latent space (default: 10)')
parser.add_argument('--d-learning-rate', '--d-lr',
                    type=float, default=1E-3, metavar='N',
                    help='learning rate of discriminator (default: 1E-3)')
parser.add_argument('--g-learning-rate', '--g-lr',
                    type=float, default=1E-4, metavar='N',
                    help='learning rate of generator (default: 1E-4)')
parser.add_argument('--unrolling-steps', '--unroll',
                    type=int, default=1, metavar='N',
                    help='number of steps to unroll gan (default: 1)')
parser.add_argument('--hidden-size', '--h-size',
                    type=int, default=64, metavar='N',
                    help='size of network intermediate layer (default: 64)')


parser.add_argument('--model', '--m',
                    default='socialWays',
                    choices=['socialWays', 'socialGAN'],
                    help='pick a dataset provider to train'
                         '(default: "socialWays")')
parser.add_argument('--dataset-type', '--data-type',
                    default='socialWays',
                    choices=['socialWays', 'socialGAN'],
                    help='pick a dataset provider to train'
                         '(default: "socialWays")')
parser.add_argument('--dataset-parser', '--data-parser',
                    default='socialWays',
                    choices=['socialWays', 'socialGAN'],
                    help='pick a dataset parser to train/test'
                         '(default: "socialWays")')

parser.add_argument('--l1o', default=False, type=str2bool)
parser.add_argument('--scale-data', default=True, type=str2bool)

parser.add_argument('--lstm-decoder', default=False, type=str2bool)
parser.add_argument('--n-lstm-layers', default=1, type=int)


parser.add_argument('--social-attention', default=True, type=str2bool)
parser.add_argument('--n-social-features', default=3, type=int)


# Graph
parser.add_argument('--graph', default=True, type=str2bool)
parser.add_argument('--n-graph-attention-heads', default=4, type=int)

parser.add_argument('--new-criterion', default=True, type=str2bool,
                    help="by default True for Graphs, and False for non Graph method")
parser.add_argument('--nll', default=False, type=str2bool,
                    help="by default True for Graphs, and False for non Graph method")

parser.add_argument('--info-loss-weight', default=0.9, type=float,
                    help="adjust info loss weight for the model only applicable with new criterion and graphs are used")

# InfoGAN
parser.add_argument('--n-disc-code', default=2, type=int)
parser.add_argument('--n-cont-code', default=2, type=int)
parser.add_argument('--c1h', default=False, type=str2bool)

# Improvements in InfoGAN not being used yet
parser.add_argument('--lambda-disc-code', default=1e-1, type=float)
parser.add_argument('--lambda_cont_code', default=1e-1, type=float)

parser.add_argument('--model-name-append', default='full-04-12', type=str)

parser.add_argument('--code-eval', default=True, type=str2bool)

parser.add_argument('--seed', default=1234, type=int)

args = parser.parse_args()


torch.manual_seed(args.seed)

epochs = args.epochs
leave_one_out = args.l1o
dataset_name = args.dataset
model_name = args.model

use_graph = args.graph
use_social = args.social_attention
new_criterion = args.new_criterion

code_eval = args.code_eval

if use_graph:
    new_criterion = True
else:
    new_criterion = False

# Info GAN
use_info_loss = True
loss_info_w = 0.5

if new_criterion and use_graph:
    loss_info_w = args.info_loss_weight

use_categ_loss = False
use_cont_loss = True
use_nll_loss = False

categorical_code_dim = 0
continous_code_dim = 0
if use_graph:
    categorical_code_dim = args.n_disc_code
    continous_code_dim = args.n_cont_code
    if categorical_code_dim > 0:
        use_categ_loss = True
    else:
        use_categ_loss = False
    if continous_code_dim > 0:
        use_cont_loss = True
    else:
        use_cont_loss = False

    use_nll_loss = args.nll

    # adding info loss to discriminator improves the scores
    discriminator_use_info_loss = True
else:
    continous_code_dim = 2  # original socialWays code dimensions/length
    # socialWays uses info loss for discriminator
    discriminator_use_info_loss = True
    # socialWays only have continous loss
    use_categ_loss = False
    use_cont_loss = True
    use_nll_loss = False

latent_code_dim = args.n_disc_code + args.n_cont_code

# L2 GAN
use_l2_loss = False
use_variety_loss = False
loss_l2_w = 0.5  # WARNING for both L2 and variety

# Graph Attention Network params
head_count = args.n_graph_attention_heads

# Learning Rate
lr_g = args.g_learning_rate
lr_d = args.d_learning_rate

# Batch size
batch_size = args.batch_size

# LSTM hidden size
lstm_decoder = args.lstm_decoder
n_lstm_layers = args.n_lstm_layers
hidden_size = args.hidden_size

num_social_features = args.n_social_features
social_feature_size = args.hidden_size
noise_len = args.hidden_size // 2

n_unrolling_steps = args.unrolling_steps

dataset_method = args.dataset_type
scale_data = args.scale_data
parser = args.dataset_parser

train_data_list_of_list: list = []
test_data_list_of_list: list = []


# fix this hardcoded variable
n_next = 12
one_hot = args.c1h

linear_after_gatconv = False
if (dataset_name in 'hotel' and not leave_one_out):
    linear_after_gatconv = True

# LSTM-based path encoder
encoder = EncoderLstm(hidden_size, n_lstm_layers,
                      use_graph, head_count, linear_after_gatconv=linear_after_gatconv).to(device)
feature_embedder = EmbedSocialFeatures(
    num_social_features, social_feature_size).to(device)
attention = AttentionPooling(hidden_size, social_feature_size).to(device)

# Decoder
decoder = None
if not lstm_decoder:
    decoder = DecoderFC(
        hidden_size + social_feature_size + noise_len).to(device)
# decoder = DecoderLstm(social_feature_size + VEL_VEC_LEN + noise_len, traj_code_len).to(device)
else:
    decoder = DecoderLstm(hidden_size + social_feature_size +
                          noise_len, hidden_size, num_layers=n_lstm_layers).to(device)

# The Generator parameters and their optimizer
predictor_params = chain(attention.parameters(), feature_embedder.parameters(),
                         encoder.parameters(), decoder.parameters())
predictor_optimizer = opt.Adam(predictor_params, lr=lr_g, betas=(0.9, 0.999))

# The Discriminator parameters and their optimizer
D = Discriminator(device, n_next, hidden_size, categorical_code_dim, continous_code_dim,
                  use_categ_loss, use_cont_loss).to(device)
D_optimizer = opt.Adam(D.parameters(), lr=lr_d, betas=(0.9, 0.999))

mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()
nll_loss = nn.GaussianNLLLoss()


def predict(obsv_4d, noise, n_next, sub_batches=[], obsv_graph=None):
    # Batch size
    bs = obsv_4d.shape[0]

    # Adds the velocity component to the observations.
    # This makes of obsv_4d a batch_sizexTx4 tensor
    # obsv_4d = get_traj_4d(obsv_p, [])

    # Initial values for the hidden and cell states (zero)
    lstm_h_c = (torch.zeros((1*n_lstm_layers), bs, encoder.hidden_size).to(device),
                torch.zeros((1*n_lstm_layers), bs, encoder.hidden_size).to(device))
    encoder.init_lstm(lstm_h_c[0], lstm_h_c[1])
    # Apply the encoder to the observed sequence
    # obsv_4d: batch_sizexTx4 tensor
    encoder(obsv_4d, obsv_graph)

    if len(sub_batches) == 0:
        sub_batches = [[0, obsv_4d.size(0)]]

    if use_social:
        features = SocialFeatures(obsv_4d, sub_batches)
        emb_features = feature_embedder(features, sub_batches)
        weighted_features = attention(
            emb_features, encoder.lstm_h[0].squeeze(), sub_batches)
    else:
        weighted_features = torch.zeros_like(encoder.lstm_h[0].squeeze())

    pred_4ds = []
    last_obsv = obsv_4d[:, -1]
    # For all the steps to predict, applies a step of the decoder
    for ii in range(n_next):
        # Takes the current output of the encoder to feed the decoder
        # Gets the ouputs as a displacement/velocity
        new_v = decoder(encoder.lstm_h[0].view(
            bs, -1), weighted_features.view(bs, -1), noise, device).view(bs, 2)
        # Deduces the predicted position
        new_p = new_v + last_obsv[:, :2]
        # The last prediction done will be new_p,new_v
        last_obsv = torch.cat([new_p, new_v], dim=1)
        # Keeps all the predictions
        pred_4ds.append(last_obsv)
        # Applies LSTM encoding to the last prediction
        # pred_4ds[-1]: batch_sizex4 tensor
        encoder(pred_4ds[-1], None)

    return torch.stack(pred_4ds, 1)


# ===================================================

def train(epoch):
    tic = time.process_time()
    # Evaluation metrics (ADE/FDE)
    train_ADE, train_FDE = 0, 0
    n_train_samples = 0

    # For all the training batches
    for count_, data_from_list in enumerate(train_data_list_of_list):
        train_data_list, train_size, scale = data_from_list
        if not scale_data or scale is None:
            scale_x = 1
        else:
            scale_x = scale.sx

        n_train_samples = n_train_samples + train_size

        for count, data in enumerate(train_data_list):
            (obsv, pred, obsv_4d, pred_4d, obsv_graph, bs, sub_batches) = data

            zeros = Variable(torch.zeros(
                bs, 1) + np.random.uniform(0, 0.1), requires_grad=False).to(device)
            ones = Variable(torch.ones(
                bs, 1) * np.random.uniform(0.9, 1.0), requires_grad=False).to(device)

            if new_criterion:
                zeros = Variable(torch.FloatTensor(bs, 1).fill_(0.0),
                                 requires_grad=False).to(device)
                ones = Variable(torch.FloatTensor(bs, 1).fill_(
                    1.0), requires_grad=False).to(device)

            noise = get_actual_noise(
                bs, noise_len, categorical_code_length=categorical_code_dim, continous_code_length=continous_code_dim, device=device, one_hot=one_hot)

            # ============== Train Discriminator ================
            for u in range(n_unrolling_steps + 1):
                # Zero the gradient buffers of all parameters
                D.zero_grad()
                with torch.no_grad():
                    pred_hat_4d = predict(
                        obsv_4d, noise, n_next, sub_batches, obsv_graph=obsv_graph)

                # classify fake samples
                fake_labels, (categ_code_hat, cont_code_hat) = D(
                    obsv_4d, pred_hat_4d)
                # Evaluate the MSE loss: the fake_labels should be close to zero

                d_loss_fake = None
                if new_criterion:
                    d_loss_fake = bce_loss(torch.sigmoid(fake_labels), zeros)
                else:
                    d_loss_fake = mse_loss(fake_labels, zeros)

                d_code_input = get_latent_codes_from_noise(
                    noise, categorical_code_dim, continous_code_dim)

                (categ_code, cont_code) = get_categ_and_cont_codes(
                    d_code_input, categorical_code_dim, continous_code_dim)

                d_loss_info = None

                if use_categ_loss and use_cont_loss:
                    category_code_loss = 0.0
                    if categorical_code_dim > 0:
                        category_code_loss = ce_loss(
                            categ_code_hat, categ_code)

                    continous_code_loss = None
                    if use_nll_loss:
                        var = torch.ones(bs, continous_code_dim,
                                         requires_grad=True)
                        continous_code_loss = nll_loss(
                            cont_code_hat, cont_code, var)
                    else:
                        continous_code_loss = mse_loss(
                            cont_code_hat, cont_code)

                    d_loss_info = loss_info_w * \
                        (continous_code_loss + category_code_loss)
                else:
                    # original had cont_code_hat.squeeze() applied for mse loss
                    continous_code_loss = mse_loss(
                        cont_code_hat, cont_code)
                    d_loss_info = continous_code_loss
                    d_loss_info = loss_info_w * d_loss_info

                # Evaluate the MSE loss: the real should be close to one
                # classify real samples
                real_labels, (categ_code_hat, cont_code_hat) = D(
                    obsv_4d, pred_4d)

                d_loss_real = None
                if new_criterion:
                    d_loss_real = bce_loss(torch.sigmoid(real_labels), ones)
                else:
                    d_loss_real = mse_loss(real_labels, ones)

                #  FIXME: which loss functinos to use for D?

                if new_criterion:
                    d_loss = (d_loss_fake + d_loss_real)
                    if use_info_loss and discriminator_use_info_loss:
                        d_loss += d_loss_info
                else:
                    d_loss = d_loss_fake + d_loss_real
                    if use_info_loss and discriminator_use_info_loss:
                        d_loss += d_loss_info

                d_loss.backward()  # update D
                D_optimizer.step()

                if u == 0 and n_unrolling_steps > 0:
                    backup = copy.deepcopy(D)

            # =============== Train Generator ================= #
            # Zero the gradient buffers of all the discriminator parameters
            D.zero_grad()
            # Zero the gradient buffers of all the generator parameters
            predictor_optimizer.zero_grad()
            # Applies a forward step of prediction
            pred_hat_4d = predict(obsv_4d, noise, n_next,
                                  sub_batches, obsv_graph=obsv_graph)

            # Classify the generated fake sample
            gen_labels, (categ_code_hat, cont_code_hat) = D(
                obsv_4d, pred_hat_4d)
            # L2 loss between the predicted paths and the true ones
            g_loss_l2 = mse_loss(pred_hat_4d[:, :, :2], pred)
            # Adversarial loss (classification labels should be close to one)
            g_loss_fooling = None
            if new_criterion:
                g_loss_fooling = bce_loss(torch.sigmoid(gen_labels), ones)
            else:
                g_loss_fooling = mse_loss(gen_labels, ones)

            # Information loss
            g_code_input = get_latent_codes_from_noise(
                noise, categorical_code_dim, continous_code_dim)

            (categ_code, cont_code) = get_categ_and_cont_codes(
                g_code_input, categorical_code_dim, continous_code_dim)

            g_loss_info = None

            if use_categ_loss and use_cont_loss:
                category_code_loss = 0.0
                if categorical_code_dim > 0:
                    category_code_loss = ce_loss(
                        categ_code_hat, categ_code)

                continous_code_loss = None
                if use_nll_loss:
                    var = torch.ones(bs, continous_code_dim,
                                     requires_grad=True)
                    continous_code_loss = nll_loss(
                        cont_code_hat, cont_code, var)
                else:
                    continous_code_loss = mse_loss(cont_code_hat, cont_code)

                g_loss_info = loss_info_w * \
                    (continous_code_loss + category_code_loss)
            else:
                # original had cont_code_hat.squeeze() applied for mse loss
                continous_code_loss = mse_loss(
                    cont_code_hat, cont_code)
                g_loss_info = continous_code_loss
                g_loss_info = loss_info_w * g_loss_info

            #  FIXME: which loss functions to use for G?
            #
            g_loss = g_loss_fooling
            # If using the info loss
            if use_info_loss:
                g_loss += g_loss_info

            # If using the L2 loss
            if use_l2_loss:
                g_loss += loss_l2_w * g_loss_l2
            if use_variety_loss:
                KV = 20
                all_20_losses = []
                for k in range(KV):
                    pred_hat_4d = predict(
                        obsv_4d, noise, n_next, sub_batches, obsv_graph=obsv_graph)
                    loss_l2_k = mse_loss(pred_hat_4d[k, :, :2], pred[k])
                all_20_losses.append(loss_l2_k.unsqueeze(0))
                all_20_losses = torch.cat(all_20_losses)
                variety_loss, _ = torch.min(all_20_losses, dim=0)
                g_loss += loss_l2_w * variety_loss

            g_loss.backward()
            predictor_optimizer.step()

            if n_unrolling_steps > 0:
                D.load(backup)
                del backup

            # calculate error
            with torch.no_grad():
                err_all = torch.pow(
                    (pred_hat_4d[:, :, :2] - pred) / scale_x, 2)
                err_all = err_all.sum(dim=2).sqrt()
                e = err_all.sum().item() / n_next
                train_ADE += e
                train_FDE += err_all[:, -1].sum().item()

    train_ADE /= n_train_samples
    train_FDE /= n_train_samples
    toc = time.process_time()
    print(f'Epc={epoch:4d}, Train ADE,FDE = ({train_ADE:.3f}, {train_FDE:.3f}) | time = {(toc - tic):.2f}',
          ": Total-Elapsed-Time", get_formatted_time(main_tic, toc))


def test(epoch, n_gen_samples=20, linear=False, write_to_file=None, just_one=False):
    tic = time.process_time()
    # =========== Test error ============
    ade_avg_12, fde_avg_12 = 0, 0
    ade_min_12, fde_min_12 = 0, 0
    n_test_samples = 0

    for count_, data_from_list in enumerate(test_data_list_of_list):
        test_data_list, test_size, scale = data_from_list
        if not scale_data or scale is None:
            scale_x = 1
        else:
            scale_x = scale.sx
        n_test_samples = n_test_samples + test_size

        for count, data in enumerate(test_data_list):
            (obsv, pred, obsv_4d, pred_4d, obsv_graph, bs, sub_batches) = data

            # FIXME - this is broken
            # current_t = dataset_t[batch_i[0]]
            current_t = count

            with torch.no_grad():
                all_20_errors = []
                all_20_preds = []

                linear_preds = predict_cv(obsv, n_next)
                if linear and not write_to_file:
                    all_20_preds.append(linear_preds.unsqueeze(0))
                    err_all = torch.pow(
                        (linear_preds[:, :, :2] - pred) / scale_x, 2).sum(dim=2, keepdim=True).sqrt()
                    all_20_errors.append(err_all.unsqueeze(0))
                else:
                    for kk in range(n_gen_samples):
                        noise = get_actual_noise(
                            bs, noise_len, categorical_code_length=categorical_code_dim, continous_code_length=continous_code_dim, device=device, one_hot=one_hot)
                        pred_hat_4d = predict(
                            obsv_4d, noise, n_next, obsv_graph=obsv_graph)
                        all_20_preds.append(pred_hat_4d.unsqueeze(0))
                        err_all = torch.pow(
                            (pred_hat_4d[:, :, :2] - pred) / scale_x, 2).sum(dim=2, keepdim=True).sqrt()
                        all_20_errors.append(err_all.unsqueeze(0))

                all_20_errors = torch.cat(all_20_errors)

                if write_to_file:
                    file_name = os.path.join(write_to_file, str(
                        epoch) + '-' + str(current_t) + '.npz')
                    print('saving to ', file_name)
                    np_obsvs = scale.denormalize(
                        obsv[:, :, :2].data.cpu().numpy())
                    np_preds_our = scale.denormalize(
                        torch.cat(all_20_preds)[:, :, :, :2].data.cpu().numpy())
                    np_preds_gtt = scale.denormalize(
                        pred[:, :, :2].data.cpu().numpy())
                    np_preds_lnr = scale.denormalize(
                        linear_preds[:, :, :2].data.cpu().numpy())
                    np.savez(file_name, timestamp=current_t,
                             obsvs=np_obsvs, preds_our=np_preds_our, preds_gtt=np_preds_gtt, preds_lnr=np_preds_lnr)

                # =============== Prediction Errors ================
                fde_min_12_i, _ = all_20_errors[:, :, -1].min(0, keepdim=True)
                ade_min_12_i, _ = all_20_errors.mean(2).min(0, keepdim=True)
                fde_min_12 += fde_min_12_i.sum().item()
                ade_min_12 += ade_min_12_i.sum().item()

                fde_avg_12 += all_20_errors[:, :, -
                                            1].mean(0, keepdim=True).sum().item()
                ade_avg_12 += all_20_errors.mean(2).mean(0,
                                                         keepdim=True).sum().item()
                # ==================================================
            if just_one:
                break

    if not just_one:
        ade_avg_12 /= n_test_samples
        fde_avg_12 /= n_test_samples
        ade_min_12 /= n_test_samples
        fde_min_12 /= n_test_samples

    toc = time.process_time()
    print(f'Avg ADE,FDE (12)= ({ade_avg_12:.3f}, {fde_avg_12:.3f}) | Min({n_gen_samples}) ADE,FDE (12)= ({ade_min_12:.3f}, {fde_min_12:.3f}) | time= {(toc - tic):.2f}',
          ": Total-Elapsed-Time", get_formatted_time(main_tic, toc))


def eval_codes(epoch, n_gen_samples=20, linear=False, write_to_file=None):
    tic = time.process_time()
    # =========== Test error ============
    ade_avg_12, fde_avg_12 = 0, 0
    ade_min_12, fde_min_12 = 0, 0
    n_test_samples = 0

    all_velocities = []

    all_categ1_pred_angle = []
    all_categ2_pred_angle = []

    all_cont1_pred_vel = []
    all_cont1_pred_slope = []
    all_cont1_pred_shift = []

    all_cont2_pred_vel = []
    all_cont2_pred_slope = []
    all_cont2_pred_shift = []

    angle_avg_diff = []

    whole_mean_angle = []
    whole_mean_direction = []
    whole_mean_speed = []
    whole_mean_shift = []

    for count_, data_from_list in enumerate(test_data_list_of_list):
        test_data_list, test_size, scale = data_from_list
        if not scale_data or scale is None:
            scale_x = 1
        else:
            scale_x = scale.sx
        n_test_samples = n_test_samples + test_size

        for count, data in enumerate(test_data_list):
            (obsv, pred, obsv_4d, pred_4d, obsv_graph, bs, sub_batches) = data

            with torch.no_grad():
                all_20_errors = []
                all_20_preds = []

                pred_velocity = calculateVelocityFor(pred, scale_x)
                mean_vel = pred_velocity.mean(2)
                all_velocities.append(mean_vel[0, 0])

                for kk in range(n_gen_samples):
                    noise = get_actual_noise(
                        bs, noise_len, categorical_code_length=categorical_code_dim, continous_code_length=continous_code_dim, device=device, one_hot=one_hot)

                    latent_code_size = categorical_code_dim + continous_code_dim
                    noise = noise[:, :-latent_code_size]

                    fixed_categ0 = 1.0
                    fixed_categ1 = 1.0
                    fixed_cont0 = 0.0
                    fixed_cont1 = 0.0

                    categ_step = 1
                    cont_step = 0.1

                    categ1_pred_angle = []
                    categ2_pred_angle = []

                    cont1_pred_vel = []
                    cont1_pred_slope = []
                    cont1_pred_shift = []
                    cont2_pred_vel = []
                    cont2_pred_slope = []
                    cont2_pred_shift = []

                    # first loop for categ variables
                    for i in range(0, 2):
                        index = 0
                        previousPred = None
                        for categ in np.arange(0, (9+categ_step), categ_step):
                            # for cont in np.arange(-1, 1, 0.1):
                            _noise = noise

                            # seems to be controlling speed
                            # seems to be controlling tilt/shift
                            a = round(categ, 1)
                            x = 0
                            y = 0

                            if i == 0:
                                x = a
                            elif i == 1:
                                y = a
                            else:
                                print("Should not be here!")
                                continue

                            categ_code = torch.FloatTensor([x, y]).to(device)
                            categ_code = torch.unsqueeze(
                                categ_code, 0).repeat(bs, 1)
                            _noise = torch.cat((_noise, categ_code), dim=1)

                            cont_code = torch.FloatTensor(
                                [fixed_cont0, fixed_cont1]).to(device)
                            cont_code = torch.unsqueeze(
                                cont_code, 0).repeat(bs, 1)
                            _noise = torch.cat((_noise, cont_code), dim=1)

                            pred_hat_4d = predict(
                                obsv_4d, _noise, n_next, obsv_graph=obsv_graph)

                            pred_angle = calculateAngleFor(
                                pred_hat_4d[:, :, :2], pred_4d[:, :, :2])
                            # mean_angle = pred_angle.mean(2)

                            if i == 0:
                                categ1_pred_angle.append(pred_angle[0, 0])
                                angle_avg_diff.append(
                                    pred_angle[0, 0].squeeze())

                            elif i == 1:
                                categ2_pred_angle.append(pred_angle[0, 0])

                    # categ1_diff = np.diff(categ1_pred_angle)
                    # all_categ1_pred_angle.append(categ1_diff.mean())

                    # categ2_diff = np.diff(categ2_pred_angle)
                    # all_categ2_pred_angle.append(categ2_diff.mean())

                    all_categ1_pred_angle.append(
                        sum(categ1_pred_angle) / len(categ1_pred_angle))

                    all_categ2_pred_angle.append(
                        sum(categ2_pred_angle) / len(categ2_pred_angle))

                    # second loop for cont variables
                    for i in range(0, 2):
                        index = 0
                        previousPred = None
                        for cont in np.arange(-1, (1+cont_step), cont_step):
                            # for cont in np.arange(-1, 1, 0.1):
                            _noise = noise

                            # seems to be controlling speed
                            # seems to be controlling tilt/shift
                            a = round(cont, 1)
                            x = 0
                            y = 0

                            if i == 0:
                                x = a
                            elif i == 1:
                                y = a
                            else:
                                print("Should not be here!")
                                continue

                            categ_code = torch.FloatTensor(
                                [fixed_categ0, fixed_categ1]).to(device)
                            categ_code = torch.unsqueeze(
                                categ_code, 0).repeat(bs, 1)
                            _noise = torch.cat((_noise, categ_code), dim=1)

                            cont_code = torch.FloatTensor([x, y]).to(device)
                            cont_code = torch.unsqueeze(
                                cont_code, 0).repeat(bs, 1)
                            _noise = torch.cat((_noise, cont_code), dim=1)

                            pred_hat_4d = predict(
                                obsv_4d, _noise, n_next, obsv_graph=obsv_graph)

                            pred_velocity = calculateVelocityFor(
                                pred_hat_4d[:, :, :2], scale_x)
                            mean_vel = pred_velocity.mean(2)

                            slope = calculateSlopeFor(
                                pred_hat_4d[:, :, :2], scale_x)
                            mean_slope = slope.mean(2)

                            predDistance = None

                            if index > 0:
                                predDistance = calculateShiftFor(
                                    pred_hat_4d[:, :, :2], previousPred)
                                predDistance = predDistance.mean(2)
                            else:
                                previousPred = pred_hat_4d[:, :, :2]

                            if i == 0:
                                cont1_pred_vel.append(mean_vel[0, 0])
                                cont1_pred_slope.append(mean_slope[0, 0])
                                if predDistance is not None:
                                    cont1_pred_shift.append(predDistance[0, 0])
                            elif i == 1:
                                cont2_pred_vel.append(mean_vel[0, 0])
                                cont2_pred_slope.append(mean_slope[0, 0])
                                if predDistance is not None:
                                    cont2_pred_shift.append(predDistance[0, 0])

                            index = index + 1

                    # first calculate the variation in speed (total differecnce then the mean)
                    xdiff = np.diff(cont1_pred_vel)
                    xdiff = xdiff.mean()  # xdiff = np.sum(xdiff) / len(xdiff)
                    all_cont1_pred_vel.append(xdiff)

                    ydiff = np.diff(cont2_pred_vel)
                    ydiff = ydiff.mean()  # ydiff = np.sum(ydiff) / len(ydiff)
                    all_cont2_pred_vel.append(ydiff)

                    # do same thing for slope as for the velocity
                    xdiff = np.diff(cont1_pred_slope)
                    xdiff = xdiff.mean()
                    all_cont1_pred_slope.append(xdiff)

                    ydiff = np.diff(cont2_pred_slope)
                    ydiff = ydiff.mean()
                    all_cont2_pred_slope.append(ydiff)

                    # do same thing for shift as for the velocity
                    xdiff = np.diff(cont1_pred_shift)
                    xdiff = xdiff.mean()
                    all_cont1_pred_shift.append(xdiff)

                    ydiff = np.diff(cont2_pred_shift)
                    ydiff = ydiff.mean()
                    all_cont2_pred_shift.append(ydiff)

                # again calculate average (speed) of all the samples
                total_xdiff_vel = sum(all_cont1_pred_vel) / \
                    len(all_cont1_pred_vel)
                total_xdiff_slope = sum(
                    all_cont1_pred_slope) / len(all_cont1_pred_slope)
                total_xdiff_shift = sum(all_cont1_pred_shift) / \
                    len(all_cont1_pred_shift)

                total_ydiff_vel = sum(all_cont2_pred_vel) / \
                    len(all_cont2_pred_vel)
                total_ydiff_slope = sum(
                    all_cont2_pred_slope) / len(all_cont2_pred_slope)
                total_ydiff_shift = sum(
                    all_cont2_pred_shift) / len(all_cont2_pred_shift)

                # print(
                #     f'(Velocity, slope, shift): Cont-1 (speed) = ({total_xdiff_vel:.6f}, {total_xdiff_slope:.6f}, {total_xdiff_shift:.6f}) | Cont-2 (shift) = ({total_ydiff_vel:.6f}, {total_ydiff_slope:.6f}, {total_ydiff_shift:.6f})')

                total_categ1_pred_angle = sum(
                    all_categ1_pred_angle) / len(all_categ1_pred_angle)
                total_categ2_pred_angle = sum(
                    all_categ2_pred_angle) / len(all_categ2_pred_angle)

                diff_degrees = math.degrees(
                    np.diff(np.array(angle_avg_diff)).mean())

                # print('Categ-1 (angle) = ', math.degrees(np.arccos(total_categ1_pred_angle)),
                #       '| Categ-2 (angle) = ', math.degrees(np.arccos(total_categ2_pred_angle)), 'categ-1 diff :', diff_degrees)

                whole_mean_angle.append(diff_degrees)
                whole_mean_direction.append(math.degrees(
                    np.arccos(total_categ2_pred_angle)))
                whole_mean_speed.append(total_xdiff_vel)
                whole_mean_shift.append(total_ydiff_shift)

    toc = time.process_time()

    # print(f'Avg Pred Vel =({mean_pred_velocity:.4f})')

    angle_array = np.array(whole_mean_angle)
    angle_array = angle_array[~np.isnan(angle_array)]

    ang_min = angle_array.min()
    ang_avg = angle_array.mean()
    ang_max = angle_array.max()

    dir_array = np.array(whole_mean_direction)
    dir_array = dir_array[~np.isnan(dir_array)]

    dir_min = dir_array.min()
    dir_avg = dir_array.mean()
    dir_max = dir_array.max()

    print('Categ-1 (angle) = (', ang_min, '|', ang_avg, '|', ang_max,
          ') | Categ-2 (dir) = ', dir_min, '|', dir_avg, '|', dir_max)

    print('cont-1 (speed) = (', np.array(whole_mean_speed).min(), '|', np.array(whole_mean_speed).mean(), '|', np.array(whole_mean_speed).max(),
          ') | cont-2 (shift) = ', np.array(whole_mean_shift).min(), '|', np.array(whole_mean_shift).mean(), '|', np.array(whole_mean_shift).max())

    print("Total-Elapsed-Time", get_formatted_time(main_tic, toc))

    # print(f'Avg ADE,FDE (12)= ({ade_avg_12:.3f}, {fde_avg_12:.3f}) | Min({n_gen_samples}) ADE,FDE (12)= ({ade_min_12:.3f}, {fde_min_12:.3f}) | time= {(toc - tic):.2f}',  ": Total-Elapsed-Time", get_formatted_time(main_tic, toc))

    pass


def perform_training(epochs=epochs, start_epoch=1):
    epochs_train_start_time = time.process_time()
    for epoch in trange(start_epoch, epochs + 1):
        # Main training function
        train(epoch=epoch)

        # ============== Save model on disk ===============
        if epoch % 50 == 0:  # FIXME : set the interval for running tests
            print('Saving model to file ...', model_file)
            torch.save({
                'run_time': run_date_time,
                'epoch': epoch,
                'attentioner_dict': attention.state_dict(),
                'feature_embedder_dict': feature_embedder.state_dict(),
                'encoder_dict': encoder.state_dict(),
                'decoder_dict': decoder.state_dict(),
                'pred_optimizer': predictor_optimizer.state_dict(),
                'D_dict': D.state_dict(),
                'D_optimizer': D_optimizer.state_dict()
            }, model_file)

            model_dir = model_file
            model_dir = model_file.replace('.pt', '')
            model_dir = model_dir + '/' + str(epoch) + '/' + model_file
            os.makedirs(os.path.dirname(
                model_dir), exist_ok=True)

            # save intermediate dicts too in a folder
            torch.save({
                'run_time': run_date_time,
                'epoch': epoch,
                'attentioner_dict': attention.state_dict(),
                'feature_embedder_dict': feature_embedder.state_dict(),
                'encoder_dict': encoder.state_dict(),
                'decoder_dict': decoder.state_dict(),
                'pred_optimizer': predictor_optimizer.state_dict(),
                'D_dict': D.state_dict(),
                'D_optimizer': D_optimizer.state_dict()
            }, model_dir)

        if epoch % 25 == 0:
            file_path = "medium"
            wr_dir = './TRAJ/trained_models/' + file_path + '/' + dataset_name + \
                '/' + model_name + '/' + str(epoch)

            wr_dir = None
            if wr_dir:
                os.makedirs(wr_dir, exist_ok=True)

            print('Performing validation on test data...')
            epochs_val_start_time = time.process_time()

            test(epoch=epoch, n_gen_samples=16,
                 write_to_file=wr_dir, just_one=False)

            epochs_val_end_time = time.process_time()
            time_print_util('Val Time:', epochs_val_start_time,
                            epochs_val_end_time)

    epochs_trian_end_time = time.process_time()

    time_print_util('Train Time:', epochs_train_start_time,
                    epochs_trian_end_time)


def perform_test(start_epoch=1):
    test_start_time = time.process_time()
    wr_dir = './TRAJ/trained_models/preds-iccv/' + \
        dataset_name + '/' + model_name + '/' + str(0000)
    wr_dir = None
    if wr_dir:
        os.makedirs(wr_dir, exist_ok=True)

    print('Performing test on data...')
    # test(epoch=epoch, n_gen_samples=4, write_to_file=wr_dir)
    test(epoch=start_epoch, n_gen_samples=192)
    # test(epoch=epoch, n_gen_samples=128, write_to_file=wr_dir, just_one=True)

    test_end_time = time.process_time()
    time_print_util('Test Time:', test_start_time,
                    test_end_time)


def perform_code_evaluation(start_epoch=1):
    eval_start_time = time.process_time()

    print('Performing Code Evaluation on data...')
    eval_codes(epoch=start_epoch, n_gen_samples=1, linear=False)

    eval_end_time = time.process_time()
    time_print_util('Eval Time:', eval_start_time,
                    eval_end_time)
    pass


def main():
    global main_tic
    global train_data_list_of_list, test_data_list_of_list
    global epochs

    main_tic = time.process_time()
    start_epoch = 1

    if os.path.isfile(model_file):
        print('Loading model from ' + model_file)
        checkpoint = torch.load(model_file)

        run_date_time = checkpoint['run_time']
        print('model trained on: ' + run_date_time)

        start_epoch = checkpoint['epoch'] + 1

        attention.load_state_dict(checkpoint['attentioner_dict'])
        feature_embedder.load_state_dict(checkpoint['feature_embedder_dict'])
        encoder.load_state_dict(checkpoint['encoder_dict'])
        decoder.load_state_dict(checkpoint['decoder_dict'])
        predictor_optimizer.load_state_dict(checkpoint['pred_optimizer'])

        D.load_state_dict(checkpoint['D_dict'])
        D_optimizer.load_state_dict(checkpoint['D_optimizer'])

    if start_epoch > epochs:
        _, test_data_list_of_list = get_train_test_data(
            dataset_name, device, batch_size=batch_size, use_graph=use_graph, leave_one_out=leave_one_out, isTest=True)

        if not code_eval:
            perform_test(start_epoch=start_epoch)
        else:
            perform_code_evaluation(start_epoch=start_epoch)
    else:
        if leave_one_out:
            train_data_list_of_list, _ = get_train_test_data(
                dataset_name, device, batch_size=batch_size, use_graph=use_graph, leave_one_out=leave_one_out)
            _, test_data_list_of_list = get_train_test_data(
                dataset_name, device, batch_size=batch_size, use_graph=use_graph, leave_one_out=leave_one_out, isTest=True)
        else:
            train_data_list_of_list, test_data_list_of_list = get_train_test_data(
                dataset_name, device, batch_size=batch_size, use_graph=use_graph, leave_one_out=leave_one_out)

        perform_training(epochs=epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    dict_args = vars(args)

    now = datetime.now()  # current date and time
    run_date_time = now.strftime("%d-%m-%y, %H:%M:%S")

    print("===== START -->> Data and Time:",
          run_date_time, "(", platform.system(), "-", os.name, ")", "=====")
    print("----- printing args -----")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dict_args)
    print("--x-- printing args --x--")

    model_file = get_model_file_name_from(
        args=args) + ".pt"
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    print("Model File For Training: ", model_file)

    main()

    old_now = now
    now = datetime.now()  # current date and time
    run_date_time = now.strftime("%d-%m-%y, %H:%M:%S")

    delta = now - old_now

    print("===== END -->> 'Data and Time:",
          run_date_time, "(", delta, ")", "=====")
