import argparse
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_model_file_name_from(args):
    epochs = args.epochs

    use_social = args.social_attention
    use_graph = args.graph
    new_criterion = args.new_criterion

    dataset_name = args.dataset
    model_name = args.model

    model_seperation_name = ""

    if use_graph:
        new_criterion = True
    else:
        new_criterion = False

    if use_social and use_graph:
        model_seperation_name = model_seperation_name + "_sg"
    elif not use_social and not use_graph:
        model_seperation_name = model_seperation_name + "_n"
    elif use_graph:
        model_seperation_name = model_seperation_name + "_g"
    elif use_social:
        model_seperation_name = model_seperation_name + "_s"

    if use_graph:
        model_seperation_name = model_seperation_name + "_" + \
            str(args.n_disc_code) + str(args.n_cont_code) + "_" + \
            str(args.n_social_features) + str(args.n_graph_attention_heads)

    model_seperation_name = model_seperation_name + "_" + str(epochs)

    n_unrolling_steps = args.unrolling_steps
    if n_unrolling_steps > 1:
        model_seperation_name = model_seperation_name + \
            "_u" + str(n_unrolling_steps)

    if new_criterion and use_graph:
        model_seperation_name = model_seperation_name + "-bce"
        model_seperation_name = model_seperation_name + \
            "-" + str(args.info_loss_weight)

    if args.lstm_decoder:
        model_seperation_name = model_seperation_name + '_dLSTM'
    if (args.n_lstm_layers > 1):
        model_seperation_name = model_seperation_name + \
            '_LSTM_' + str(args.n_lstm_layers)
    if not args.model_name_append == "":
        model_seperation_name = model_seperation_name + '_' + args.model_name_append

    if args.nll:
        model_seperation_name = model_seperation_name + '_' + 'nll'

    if use_graph:
        if args.c1h:
            model_seperation_name = model_seperation_name + '_' + '1H'
        else:
            # nec means new encoder categ code changes
            model_seperation_name = model_seperation_name + '_' + 'nec'

    if args.l1o:
        model_seperation_name = model_seperation_name + '_' + 'L1O'

    model_file = './TRAJ/trained_models/' + model_name + \
        "/" + dataset_name + model_seperation_name

    model_file = model_file
    return model_file

#  plotting utils


def plotable_trajectory_list():
    # tupple type: index, start, end, total
    trajectory_indexes = {
        'eth': [(90, 0, 2, 3)],
        'hotel': [(58, 4, None, 7),
                  (68, 4, None, 6)],
        'univ': [(68, 4, None, 6)],
        'zara01': [(68, 4, None, 6)],
        'zara02': [(68, 4, None, 6)],
    }

    return trajectory_indexes


def plot_trajectory_graph(G: nx.Graph):
    options = {
        "font_size": 8,
        "node_size": 420,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 2,
        "width": 2,
    }

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] == 1]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] != 1]

    pos0 = nx.spring_layout(G, seed=20)
    # pos_spectral = nx.spectral_layout(G)
    pos1 = nx.get_node_attributes(G, 'pos')
    pos = pos0
    # nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx(G, pos, **options)

    # # edges
    # nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    # nx.draw_networkx_edges(
    #     G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    # )

    # # node labels
    # nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

    # # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    edge_labels = dict([((u, v,), f"{d['weight']:.2f}")
                        for u, v, d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_trajectories(obsv, pred, pred_hat_4d):
    total_trajectories = np.shape((obsv.cpu().numpy()))
    total_trajectories = total_trajectories[0]

    print('total trajectories for thte index:', total_trajectories)
    fig, axs = plt.subplots(1, total_trajectories)
    fig.set_size_inches(18.5, 10.5, forward=True)

    for i in range(total_trajectories):
        # if i < 4:
        #     continue

        obsv_x = obsv[i, :, 0]
        obsv_y = obsv[i, :, 1]

        obsv_x = obsv_x.cpu().numpy()
        obsv_y = obsv_y.cpu().numpy()

        pred_x = pred[i, :, 0]
        pred_y = pred[i, :, 1]

        pred_x = pred_x.cpu().numpy()
        pred_y = pred_y.cpu().numpy()

        pred_x_hat = pred_hat_4d[i, :, 0:1]
        pred_y_hat = pred_hat_4d[i, :, 1:2]

        pred_x_hat = pred_x_hat.cpu().detach().numpy()
        pred_y_hat = pred_y_hat.cpu().detach().numpy()

        # if i >= 4:
        #     plotter_i = i - 4

        # if plotter_i > 1:
        #     break

        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        axs[i].plot(obsv_x, obsv_y, '-bo')
        axs[i].plot(pred_x, pred_y, '--ro')
        axs[i].plot(pred_x_hat, pred_y_hat, '-go')

    plt.subplots_adjust(wspace=0, hspace=0)


def plot_trajectories_sep(obsv, pred, pred_hat_4d):
    total_trajectories = np.shape((obsv.cpu().numpy()))
    total_trajectories = total_trajectories[0]
    for i in range(total_trajectories):
        obsv_x = obsv[i, :, 0]
        obsv_y = obsv[i, :, 1]

        obsv_x = obsv_x.cpu().numpy()
        obsv_y = obsv_y.cpu().numpy()

        pred_x = pred[i, :, 0]
        pred_y = pred[i, :, 1]

        pred_x = pred_x.cpu().numpy()
        pred_y = pred_y.cpu().numpy()

        pred_x_hat = pred_hat_4d[i, :, 0:1]
        pred_y_hat = pred_hat_4d[i, :, 1:2]

        pred_x_hat = pred_x_hat.cpu().detach().numpy()
        pred_y_hat = pred_y_hat.cpu().detach().numpy()

        plt.plot(obsv_x, obsv_y, '-bo')
        plt.plot(pred_x, pred_y, '--ro')
        plt.plot(pred_x_hat, pred_y_hat, '-go')
        plt.show()

# time related utils


def get_formatted_time(start_time, end_time):
    total_time = end_time - start_time

    (t_hour, rem) = divmod(total_time, 3600)
    (t_min, t_sec) = divmod(rem, 60)

    return '{} hour: {} min: {} sec'.format(
        math.floor(t_hour), math.floor(t_min), math.ceil(t_sec))


def time_print_util(message, start_time, end_time):
    print(message, get_formatted_time(start_time, end_time))


# Augment tensors of positions into positions+velocity

def get_traj_4d(obsv_p, pred_p):
    obsv_v = obsv_p[:, 1:] - obsv_p[:, :-1]
    obsv_v = torch.cat([obsv_v[:, 0].unsqueeze(1), obsv_v], dim=1)
    obsv_4d = torch.cat([obsv_p, obsv_v], dim=2)
    if len(pred_p) == 0:
        return obsv_4d
    pred_p_1 = torch.cat([obsv_p[:, -1].unsqueeze(1), pred_p[:, :-1]], dim=1)
    pred_v = pred_p - pred_p_1
    pred_4d = torch.cat([pred_p, pred_v], dim=2)
    return obsv_4d, pred_4d

# InfoGAN - Noise Utils


def get_noise(sample_size, noise_length, device):
    noise = torch.FloatTensor(torch.rand(sample_size, noise_length)).to(device)
    return noise


def get_categorial_code(sample_size, code_length, device, one_hot=False):
    # ! if you compute softmax after the output layer, you should use torch.nn.NLLLoss instead.
    # ! Remove the softmax if you want to use CrossEntropyLoss (https://stackoverflow.com/questions/63792353/pytorch-and-keras-different-behaviour)

    input = torch.tensor([1/10.]*10, dtype=torch.float).to(device)
    categ_code = torch.multinomial(input, code_length).to(device)
    categ_code = torch.unsqueeze(categ_code, 0)

    if one_hot:
        rand_code = torch.randint(
            low=0, high=code_length, size=(1,)).to(device)
        categ_code = torch.nn.functional.one_hot(rand_code, code_length)

    categ_code = categ_code.repeat(sample_size, 1)
    return categ_code


def get_categorial_code_new(sample_size, code_length, device, one_hot=False, independent=False):
    # ! if you compute softmax after the output layer, you should use torch.nn.NLLLoss instead.
    # ! Remove the softmax if you want to use CrossEntropyLoss (https://stackoverflow.com/questions/63792353/pytorch-and-keras-different-behaviour)

    categ_code = torch.empty(code_length, dtype=torch.uint8)

    input = torch.tensor([1/10.]*10, dtype=torch.float).to(device)
    if independent and not one_hot:
        for i in range(code_length):
            _categ_code = torch.multinomial(input, 1).to(device)
            _categ_code = torch.unsqueeze(_categ_code, 0)
            categ_code[i] = _categ_code
    elif not independent and not one_hot:
        categ_code = torch.multinomial(input, code_length).to(device)
        categ_code = torch.unsqueeze(categ_code, 0)
    elif one_hot:
        rand_code = torch.randint(
            low=0, high=code_length, size=(1,)).to(device)
        categ_code = torch.nn.functional.one_hot(rand_code, code_length)

    categ_code = categ_code.repeat(sample_size, 1)
    return categ_code


def get_continous_code(sample_size, code_length, device, from_=-1, to_=1):
    cont_noise = torch.rand(1, code_length)
    cont_noise = torch.rand(1, code_length).uniform_(from_, to_)
    cont_noise = torch.FloatTensor(cont_noise).to(device)

    cont_noise = cont_noise.repeat(sample_size, 1)
    return cont_noise


def get_latent_codes_from_noise(noise, categ_code_dim_size, cont_code_dim_size):
    latent_code_dim = categ_code_dim_size + cont_code_dim_size
    # original socialWays noise extraction seems wrong `code_input = noise[:, :latent_code_dim]` it should be `code_input = noise[:, -latent_code_dim:]` as the codes are appended at the end
    code_input = noise[:, -latent_code_dim:]
    return code_input


def get_categ_and_cont_codes(code_input, categ_code_dim, cont_code_dim):
    if categ_code_dim > 0 and cont_code_dim > 0:
        categ_code = code_input[:, :categ_code_dim]
        cont_code = code_input[:, -cont_code_dim:]
        return (categ_code, cont_code)
    else:
        return (None, code_input)


def get_actual_noise(sample_size, noise_length, categorical_code_length, continous_code_length, device, one_hot=False):
    ret_noise = None
    if categorical_code_length == 0 and continous_code_length == 0:
        ret_noise = get_noise(sample_size, noise_length, device)
    else:
        actual_noise_length = noise_length - \
            categorical_code_length - continous_code_length
        noise = get_noise(sample_size, actual_noise_length, device)

        ret_noise = noise

        categ_code = None
        cont_code = None
        if categorical_code_length > 0:
            categ_code = get_categorial_code(
                sample_size, categorical_code_length, device, one_hot=one_hot)

            ret_noise = torch.cat((ret_noise, categ_code), dim=1)

        if continous_code_length > 0:
            cont_code = get_continous_code(
                sample_size, continous_code_length, device)

            ret_noise = torch.cat((ret_noise, cont_code), dim=1)

    return ret_noise


# Evaluate the error between the model prediction and the true path
def calc_error(pred_hat, pred, scale=1):
    N = pred.size(0)
    T = pred.size(1)
    err_all = torch.pow((pred_hat - pred) / scale,
                        2).sum(dim=2).sqrt()  # N x T
    FDEs = err_all.sum(dim=0).item() / N
    ADEs = torch.cumsum(FDEs)
    for ii in range(T):
        ADEs[ii] /= (ii + 1)
    return ADEs.data.cpu().numpy(), FDEs.data().cpu().numpy()
