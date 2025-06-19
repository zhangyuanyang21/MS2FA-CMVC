from torch import nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import torch
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.enc_1 = nn.Linear(input_dim, 500)
        self.enc_2 = nn.Linear(500, 500)
        self.enc_3 = nn.Linear(500, 2000)
        self.enc_4 = nn.Linear(2000, feature_dim)
    def forward(self, x):
        enc_d1 = F.relu(self.enc_1(x))
        enc_d2 = F.relu(self.enc_2(enc_d1))
        enc_d3 = F.relu(self.enc_3(enc_d2))
        enc_d4 = self.enc_4(enc_d3)
        return enc_d1, enc_d2, enc_d3, enc_d4
class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.commonfeature_contrastive_module = nn.Sequential(
            nn.Linear(3*feature_dim*view, high_feature_dim),
        )
        self.commonlabel_contrastive_module = nn.Sequential(
            nn.Linear(3*feature_dim*view, class_num),
            nn.Softmax(dim=1)
        )
        self.class_num = class_num
        self.view = view
        # self.trans_enc = nn.TransformerEncoderLayer(d_model=feature_dim*view, nhead=1, dim_feedforward=256)
        # self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=1)
        # add share_space
        self.mk1 = torch.nn.Linear(feature_dim*view, 64, bias=False)
        self.mv1 = torch.nn.Linear(64, feature_dim*view, bias=False)
        self.mk2 = torch.nn.Linear(feature_dim * view, 64, bias=False)
        self.mv2 = torch.nn.Linear(64, feature_dim * view, bias=False)
        self.mk3 = torch.nn.Linear(feature_dim * view, 64, bias=False)
        self.mv3 = torch.nn.Linear(64, feature_dim * view, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        qs = []
        for v in range(self.view):
            x = xs[v]
            _, _, _, z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
            qs.append(q)
        return  xrs, zs, hs, qs

    def GCFAgg(self, xs):
        zs = []
        for v in range(self.view):
            x = xs[v]
            _, _, _, z = self.encoders[v](x)
            zs.append(z)
        catz = torch.cat(zs, 1)
        #out1
        attn = self.mk1(catz)
        attn = self.softmax(attn)
        out1 = self.mv1(attn)
        out1 = self.dropout(out1)
        commonz1 = out1 + catz
        # out2
        attn = self.mk2(catz)
        attn = self.softmax(attn)
        out2 = self.mv2(attn)
        out2 = self.dropout(out2)
        commonz2 = out2 + catz
        # out3
        attn = self.mk3(catz)
        attn = self.softmax(attn)
        out3 = self.mv3(attn)
        out3 = self.dropout(out3)
        commonz3 = out3 + catz

        commonz = torch.cat((commonz1,commonz2,commonz3), 1)
        commonq = self.commonlabel_contrastive_module(commonz)
        commonz = normalize(self.commonfeature_contrastive_module(commonz), dim=1)
        return catz, commonz, commonq
