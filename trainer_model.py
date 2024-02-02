import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv

# modeling


def predict_cv(obsv, n_next):
    n_past = obsv.shape[1]
    if n_past > 2:
        my_vel = (obsv[:, -1] - obsv[:, -3]) / 2.
    else:
        my_vel = (obsv[:, -1] - obsv[:, -2])

    for si in range(n_next):
        pred_hat = obsv[:, -1] + my_vel
        obsv = torch.cat((obsv, pred_hat.unsqueeze(1)), dim=1)
    pred_hat = obsv[:, n_past:, :]
    return pred_hat


class AttentionPooling(nn.Module):
    def __init__(self, h_dim, f_dim):
        super(AttentionPooling, self).__init__()
        self.f_dim = f_dim
        self.h_dim = h_dim
        self.W = nn.Linear(h_dim, f_dim, bias=True)

    def forward(self, f, h, sub_batches):
        Wh = self.W(h)
        S = torch.zeros_like(h)
        for sb in sub_batches:
            N = sb[1] - sb[0]
            if N == 1:
                continue

            for ii in range(sb[0], sb[1]):
                fi = f[ii, sb[0]:sb[1]]
                sigma_i = torch.bmm(fi.unsqueeze(
                    1), Wh[sb[0]:sb[1]]. unsqueeze(2))
                sigma_i[ii-sb[0]] = -1000

                attentions = torch.softmax(sigma_i.squeeze(), dim=0)
                S[ii] = torch.mm(attentions.view(1, N), h[sb[0]:sb[1]])

        return S


class EmbedSocialFeatures(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EmbedSocialFeatures, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(),
                                nn.Linear(32, 64), nn.ReLU(),
                                nn.Linear(64, hidden_size))

    def forward(self, ftr_list, sub_batches):
        embedded_features = self.fc(ftr_list)
        return embedded_features


class GATConv(torch.nn.Module):
    def __init__(self, in_features=4, out_features=16, multihead_count=4, concat_multi_head_out=True,) -> None:
        super(GATConv,  self).__init__()
        self.gatconv = GATv2Conv(
            in_channels=in_features,
            out_channels=out_features,
            heads=multihead_count,
            concat=concat_multi_head_out,
            edge_dim=1)

    def forward(self, x, edge_index):
        x = self.gatconv(x, edge_index)
        return x


def DCA(xA_4d, xB_4d):
    dp = xA_4d[:2] - xB_4d[:2]
    dv = xA_4d[2:] - xB_4d[2:]
    ttca = torch.dot(-dp, dv) / (torch.norm(dv) ** 2 + 1E-6)
    # ttca = torch.max(ttca, 0)
    dca = torch.norm(dp + ttca * dv)
    return dca


def Bearing(xA_4d, xB_4d):
    dp = xA_4d[:2] - xB_4d[:2]
    v = xA_4d[2:]
    cos_theta = torch.dot(dp, v) / (torch.norm(dp) * torch.norm(v) + 1E-6)
    return cos_theta


def DCA_MTX(x_4d, D_4d):
    Dp = D_4d[:, :, :2]
    Dv = D_4d[:, :, 2:]
    DOT_Dp_Dv = torch.mul(Dp[:, :, 0], Dv[:, :, 0]) + \
        torch.mul(Dp[:, :, 1], Dv[:, :, 1])
    Dv_sq = torch.mul(Dv[:, :, 0], Dv[:, :, 0]) + \
        torch.mul(Dv[:, :, 1], Dv[:, :, 1]) + 1E-6
    TTCA = -torch.div(DOT_Dp_Dv, Dv_sq)
    DCA = torch.zeros_like(Dp)
    DCA[:, :, 0] = Dp[:, :, 0] + TTCA * Dv[:, :, 0]
    DCA[:, :, 1] = Dp[:, :, 1] + TTCA * Dv[:, :, 1]
    DCA = torch.norm(DCA, dim=2)
    return DCA


def BearingMTX(x_4d, D_4d):
    Dp = D_4d[:, :, :2]  # NxNx2
    v = x_4d[:, 2:].unsqueeze(1).repeat(1, x_4d.shape[0], 1)  # => NxNx2
    DOT_Dp_v = Dp[:, :, 0] * v[:, :, 0] + Dp[:, :, 1] * v[:, :, 1]
    COS_THETA = torch.div(DOT_Dp_v, torch.norm(Dp, dim=2)
                          * torch.norm(v, dim=2) + 1E-6)
    return COS_THETA


def SocialFeatures(x, sub_batches):
    N = x.shape[0]  # x is NxTx4 tensor

    x_ver_repeat = x[:, -1].unsqueeze(0).repeat(N, 1, 1)
    x_hor_repeat = x[:, -1].unsqueeze(1).repeat(1, N, 1)
    Dx_mat = x_hor_repeat - x_ver_repeat

    l2_dist_MTX = Dx_mat[:, :, :2].norm(dim=2)
    bearings_MTX = BearingMTX(x[:, -1], Dx_mat)
    dcas_MTX = DCA_MTX(x[:, -1], Dx_mat)
    sFeatures_MTX = torch.stack([l2_dist_MTX, bearings_MTX, dcas_MTX], dim=2)

    return sFeatures_MTX   # directly return the Social Features Matrix


class EncoderLstm(nn.Module):
    def __init__(self, hidden_size, num_layers=2, use_graph=False, graph_attention_heads=4, linear_after_gatconv: bool = True):
        # Dimension of the hidden state (h)
        self.hidden_size = hidden_size
        super(EncoderLstm, self).__init__()
        # Linear embedding 4xh
        self.embed = nn.Linear(4, self.hidden_size)

        self.use_graph = use_graph
        self.linear_after_gatconv = linear_after_gatconv
        self.obsv_len = 8
        _graph_attention_heads = graph_attention_heads
        _in_channels = 4
        _out_channels = 16
        _graph_attention_heads = 4
        _graph_attention_heads = graph_attention_heads
        _concat = True
        if _concat:
            self.interm_hidden_size = (_in_channels * _out_channels)
        else:
            self.interm_hidden_size = _out_channels

        self.gatconv = GATv2Conv(
            in_channels=_in_channels,
            out_channels=_out_channels,
            heads=_graph_attention_heads,
            concat=_concat,
            edge_dim=1)
        self.graph_embed = nn.Linear(self.interm_hidden_size, self.hidden_size)

        # The LSTM cell.
        # Input dimension (observations mapped through embedding) is the same as the output
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.lstm_h = []
        # init_weights(self)

    def init_lstm(self, h, c):
        # Initialization of the LSTM: hidden state and cell state
        self.lstm_h = (h, c)

    def forward(self, obsv, obsv_graph):
        # Batch size
        bs = obsv.shape[0]

        if self.use_graph and obsv_graph != None:
            obsv_graph = self.gatconv(obsv_graph.x, obsv_graph.edge_index)
            batch_shape = int(obsv_graph.shape[0]/self.obsv_len)
            obsv_graph = obsv_graph.reshape(
                batch_shape, self.obsv_len, self.interm_hidden_size)

            # embedding layer is needed for small dataset, like when only hotel dataset is used
            # for large dataset it worsen the results, this is needed after changing the encoder using more gat heads
            if self.linear_after_gatconv:
                obsv_graph = self.graph_embed(obsv_graph)

            # Reshape and applies LSTM over a whole sequence or over one single step
            obsv_view = obsv_graph.view(bs, -1, self.hidden_size)
            y, self.lstm_h = self.lstm(obsv_view, self.lstm_h)
            return y
        else:
            # Linear embedding
            obsv = self.embed(obsv)
            # Reshape and applies LSTM over a whole sequence or over one single step
            obsv_view = obsv.view(bs, -1, self.hidden_size)
            y, self.lstm_h = self.lstm(obsv_view, self.lstm_h)
            return y


class Generator(nn.Module):
    def __init__(self, device, use_social, use_graph, graph_head_count, hidden_size, lstm_decoder, n_lstm_layers, num_social_features, social_feature_size, noise_len):
        super(Generator, self).__init__()

        self.use_social = use_social
        self.device = device
        self.n_lstm_layers = n_lstm_layers

        self.encoder = EncoderLstm(hidden_size, n_lstm_layers,
                                   use_graph, graph_attention_heads=graph_head_count).to(device)
        self.feature_embedder = EmbedSocialFeatures(
            num_social_features, social_feature_size).to(device)
        self.attention = AttentionPooling(
            hidden_size, social_feature_size).to(device)

        # Decoder
        self.decoder = None
        if not lstm_decoder:
            self.decoder = DecoderFC(
                hidden_size + social_feature_size + noise_len).to(device)
        else:
            self.decoder = DecoderLstm(hidden_size + social_feature_size +
                                       noise_len, hidden_size, num_layers=n_lstm_layers).to(device)

    def forward(self, obsv_4d, noise, n_next, sub_batches=[], obsv_graph=None):
        # Batch size
        bs = obsv_4d.shape[0]

        # Adds the velocity component to the observations.
        # This makes of obsv_4d a batch_sizexTx4 tensor
        # obsv_4d = get_traj_4d(obsv_p, [])

        # Initial values for the hidden and cell states (zero)
        lstm_h_c = (torch.zeros((1*self.n_lstm_layers), bs, self.encoder.hidden_size).to(self.device),
                    torch.zeros((1*self.n_lstm_layers), bs, self.encoder.hidden_size).to(self.device))
        self.encoder.init_lstm(lstm_h_c[0], lstm_h_c[1])
        # Apply the encoder to the observed sequence
        # obsv_4d: batch_sizexTx4 tensor
        self.encoder(obsv_4d, obsv_graph)

        if len(sub_batches) == 0:
            sub_batches = [[0, obsv_4d.size(0)]]

        if self.use_social:
            features = SocialFeatures(obsv_4d, sub_batches)
            emb_features = self.feature_embedder(features, sub_batches)
            weighted_features = self.attention(
                emb_features, self.encoder.lstm_h[0].squeeze(), sub_batches)
        else:
            weighted_features = torch.zeros_like(
                self.encoder.lstm_h[0].squeeze())

        pred_4ds = []
        last_obsv = obsv_4d[:, -1]
        # For all the steps to predict, applies a step of the decoder
        for ii in range(n_next):
            # Takes the current output of the encoder to feed the decoder
            # Gets the ouputs as a displacement/velocity
            new_v = self.decoder(self.encoder.lstm_h[0].view(
                bs, -1), weighted_features.view(bs, -1), noise, self.device).view(bs, 2)
            # Deduces the predicted position
            new_p = new_v + last_obsv[:, :2]
            # The last prediction done will be new_p,new_v
            last_obsv = torch.cat([new_p, new_v], dim=1)
            # Keeps all the predictions
            pred_4ds.append(last_obsv)
            # Applies LSTM encoding to the last prediction
            # pred_4ds[-1]: batch_sizex4 tensor
            self.encoder(pred_4ds[-1], None)

        return torch.stack(pred_4ds, 1)


class Discriminator(nn.Module):
    def __init__(self, device, n_next, hidden_dim, n_disc_code, n_cont_code, use_aux_categ=False, use_aux_cont=True):
        super(Discriminator, self).__init__()
        self.lstm_dim = hidden_dim
        self.n_next = n_next

        self.device = device

        self.use_aux_categ = use_aux_categ
        self.use_aux_cont = use_aux_cont

        # LSTM Encoder for the observed part
        self.obsv_encoder_lstm = nn.LSTM(4, hidden_dim, batch_first=True)
        # FC sub-network: input is hidden_dim, output is hidden_dim//2. This ouput will be part of
        # the input of the classifier.
        self.obsv_encoder_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                             nn.Linear(hidden_dim // 2, hidden_dim // 2))
        # FC Encoder for the predicted part: input is n_next*4 (whole predicted trajectory), output is
        # hidden_dim//2. This ouput will also be part of the input of the classifier.
        self.pred_encoder = nn.Sequential(nn.Linear(n_next * 4, hidden_dim // 2), nn.LeakyReLU(0.2),
                                          nn.Linear(hidden_dim // 2, hidden_dim // 2))
        # Classifier: input is hidden_dim (concatenated encodings of observed and predicted trajectories), output is 1
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                        nn.Linear(hidden_dim // 2, 1))

        # Latent code inference: input is hidden_dim (concatenated encodings of observed and predicted trajectories), output is n_latent_code (distribution of latent codes)
        if self.use_aux_categ and self.use_aux_cont:
            self.latent_categ_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Softmax(dim=1),
                                                      nn.Linear(self.lstm_dim // 2, n_disc_code))
            self.latent_cont_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                                     nn.Linear(self.lstm_dim // 2, n_cont_code))
        else:
            self.latent_cont_decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.2),
                                                     nn.Linear(self.lstm_dim // 2, n_cont_code))

    def forward(self, obsv, pred):
        bs = obsv.size(0)
        lstm_h_c = (torch.zeros(1, bs, self.lstm_dim).to(self.device),
                    torch.zeros(1, bs, self.lstm_dim).to(self.device))
        # Encoding of the observed sequence trhough an LSTM cell
        obsv_code, lstm_h_c = self.obsv_encoder_lstm(obsv, lstm_h_c)
        # Further encoding through a FC layer
        obsv_code = self.obsv_encoder_fc(obsv_code[:, -1])
        # Encoding of the predicted/next part of the sequence through a FC layer
        pred_code = self.pred_encoder(pred.view(-1, self.n_next * 4))
        both_codes = torch.cat([obsv_code, pred_code], dim=1)
        # Applies classifier to the concatenation of the encodings of both parts
        label = self.classifier(both_codes)

        # Inference on the latent code
        categ_code_hat = None
        cont_code_hat = None
        if self.use_aux_categ and self.use_aux_cont:
            categ_code_hat = self.latent_categ_decoder(both_codes)
            cont_code_hat = self.latent_cont_decoder(both_codes)
        else:
            cont_code_hat = self.latent_cont_decoder(both_codes)

        return label, (categ_code_hat, cont_code_hat)

    def load(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()


# FC path decoding module
class DecoderFC(nn.Module):
    def __init__(self, hidden_dim, out_feature=2):
        super(DecoderFC, self).__init__()
        # Fully connected sub-network. Input is hidden_dim, output is 2.
        self.fc1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
                                 nn.Linear(hidden_dim, hidden_dim //
                                           2), nn.LeakyReLU(0.2),
                                 nn.Linear(hidden_dim // 2, hidden_dim // 4),
                                 nn.Linear(hidden_dim // 4, out_feature))

    def forward(self, h, s, z, device):
        # For each sample in the batch, concatenate h (hidden state), s (social term) and z (noise)
        inp = torch.cat([h, s, z], dim=1)
        # Applies the fully connected layer
        out = self.fc1(inp)
        return out


# LSTM path decoding module
class DecoderLstm(nn.Module):
    def __init__(self, input_size, hidden_size, out_feature=2, num_layers=2):
        super(DecoderLstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Decoding LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True)
        # Fully connected sub-network. Input is hidden_size, output is 2.
        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.2),
                                nn.Linear(hidden_size, hidden_size //
                                          2), nn.LeakyReLU(0.2),
                                nn.Linear(hidden_size // 2,
                                          hidden_size // 4),
                                nn.Linear(hidden_size // 4, out_feature))

        # init_weights(self)
        self.lstm_h = []

    def init_lstm(self, h, c):
        # Initialization of the LSTM: hidden state and cell state
        self.lstm_h = (h, c)

    def forward(self, h, s, z, device):
        # For each sample in the batch, concatenate h (hidden state), s (social term) and z (noise)
        inp = torch.cat([h, s, z], dim=1)

        lstm_h_c = (torch.zeros((1*self.num_layers), self.hidden_size).to(device),
                    torch.zeros((1*self.num_layers), self.hidden_size).to(device))
        self.init_lstm(lstm_h_c[0], lstm_h_c[1])

        # only for batched
        # inp = inp.unsqueeze(0)

        # Applies a forward step.
        out, self.lstm_h = self.lstm(inp, self.lstm_h)
        # Applies the fully connected layer to the LSTM output
        out = self.fc(out.squeeze())
        return out
