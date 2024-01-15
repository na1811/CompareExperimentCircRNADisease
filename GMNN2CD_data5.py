import math
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sortscore
from MakeSimilarityMatrix import MakeSimilarityMatrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle

import time

from sklearn.preprocessing import minmax_scale
import pandas as pd

# import xlsxwriter

from models import GraphConv, AE, LP

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.') 
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Dimension of representations')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='Weight between lncRNA space and disease space')
parser.add_argument('--data', type=int, default=1, choices=[1, 2],
                    help='Dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
set_seed(args.seed, args.cuda)
# gdi, ldi, rnafeat, gl, gd = load_data(args.data, args.cuda)


class GNNq(nn.Module):
    def __init__(self):
        super(GNNq, self).__init__()
        self.gnnql = AE(rnafeat.shape[1], 256, args.hidden)
        self.gnnqd = AE(gdi.shape[0], 256, args.hidden)

    def forward(self, xl0, xd0):
        hl, stdl, xl = self.gnnql(gl, xl0)
        hd, stdd, xd = self.gnnqd(gd, xd0)
        return hl, stdl, xl, hd, stdd, xd


class GNNp(nn.Module):
    def __init__(self):
        super(GNNp, self).__init__()
        self.gnnpl = LP(args.hidden, ldi.shape[1])
        self.gnnpd = LP(args.hidden, ldi.shape[0])

    def forward(self, y0):
        yl, zl = self.gnnpl(gl, y0)
        yd, zd = self.gnnpd(gd, y0.t())
        return yl, zl, yd, zd


def GIP(circrna_disease_matrix):
    make_sim_matrix = MakeSimilarityMatrix(circrna_disease_matrix)
    return make_sim_matrix


def neighborhood(feat, k):
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C


def normalized(wmat):
    deg = np.diag(np.sum(wmat, axis=0))
    degpow = np.power(deg, -0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow, wmat), degpow)
    return W


def norm_adj(feat):
    C = neighborhood(feat.T, k=10)
    norm_adj = normalized(C.T * C + np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g

def load_data(rel_matrix, cuda):
    print("This is load_data...")
    make_sim_matrix = GIP(rel_matrix)
    # circ_gipsim_matrix, dis_gipsim_matrix = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix
    rnafeat, gdi = make_sim_matrix.circsimmatrix, make_sim_matrix.dissimmatrix
    ldi = rel_matrix.copy()


    rnafeat = minmax_scale(rnafeat, axis=0)
    gdit = torch.from_numpy(gdi).float()
    ldit = torch.from_numpy(ldi).float()
    rnafeatorch = torch.from_numpy(rnafeat).float()
    gl = norm_adj(rnafeat)
    gd = norm_adj(gdi.T)
    if cuda:
        gdit = gdit.cuda()
        ldit = ldit.cuda()
        rnafeatorch = rnafeatorch.cuda()
        gl = gl.cuda()
        gd = gd.cuda()

    return gdit, ldit, rnafeatorch, gl, gd

def criterion(output, target, msg, n_nodes, mu, logvar):
    if msg == 'disease':
        cost = F.binary_cross_entropy(output, target)
    else:
        cost = F.mse_loss(output, target)

    KL = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KL


def train(gnnq, gnnp, xl0, xd0, y0, epoch, alpha, i):
    losspl1 = []
    losspd1 = []
    lossp1 = []
    lossq1 = []
    beta0 = 1.0
    gamma0 = 1.0
    optp = torch.optim.Adam(gnnp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optq = torch.optim.Adam(gnnq.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for e in range(epoch):
        gnnq.train()

        gnnq.train()
        hl, stdl, xl, hd, stdd, xd = gnnq(xl0, xd0)
        lossql = criterion(xl, xl0,
                           "lncrna", gl.shape[0], hl, stdl)
        lossqd = criterion(xd, xd0,
                           "disease", gd.shape[0], hd, stdd)
        lossq = alpha * lossql + (1 - alpha) * lossqd + beta0 * e * F.mse_loss(
            torch.mm(hl, hd.t()), y0) / epoch
        optq.zero_grad()
        lossq1.append(lossq.item())
        lossq.backward()
        optq.step()
        gnnq.eval()
        with torch.no_grad():
            hl, _, _, hd, _, _ = gnnq(xl0, xd0)

        gnnp.train()
        yl, zl, yd, zd = gnnp(y0)
        losspl = F.binary_cross_entropy(yl, y0) + gamma0 * e * F.mse_loss(zl, hl) / epoch
        losspd = F.binary_cross_entropy(yd, y0.t()) + gamma0 * e * F.mse_loss(zd, hd) / epoch
        lossp = alpha * losspl + (1 - alpha) * losspd
        losspl1.append(losspl.item())
        losspd1.append(losspd.item())
        lossp1.append(lossp.item())
        optp.zero_grad()
        lossp.backward()
        optp.step()

        with torch.no_grad():
            yl, _, yd, _ = gnnp(y0)
        if e % 20 == 0:
            print('Epoch %d | Lossp: %.4f | Lossq: %.4f' % (e, lossp.item(), lossq.item()))
    return alpha * yl + (1 - alpha) * yd.t()


if __name__=="__main__":

    # with h5py.File('./Data/disease-circRNA.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_cancer/circRNA_cancer.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circRNA_disease_from_circRNADisease/association.h5', 'r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    # with h5py.File('./Data/circ2Traits/circRNA_disease.h5','r') as hf:
    #     circrna_disease_matrix = hf['infor'][:]

    with h5py.File('./Data/circad/circrna_disease.h5', 'r') as hf:
        circrna_disease_matrix = hf['infor'][:]
    
    file_name = "saved_variables_data5.pkl"
    with open(file_name, 'rb') as file:
        loaded_variables = pickle.load(file)
    index_tuple = loaded_variables["index_tuple"]
    one_list = loaded_variables["one_list"]
    split = loaded_variables["split"]

    all_tpr = []
    all_fpr = []
    all_recall = []
    all_precision = []
    all_accuracy = []
    all_F1 = []

    count = 1
    # 5-fold start
    for i in range(0, len(one_list), split):
        test_index = one_list[i:i + split]
        new_circrna_disease_matrix = circrna_disease_matrix.copy()
    
        for index in test_index:
            new_circrna_disease_matrix[index[0], index[1]] = 0
        roc_circrna_disease_matrix = new_circrna_disease_matrix + circrna_disease_matrix
        rel_matrix = new_circrna_disease_matrix
        circnum = rel_matrix.shape[0]
        disnum = rel_matrix.shape[1]

        gdi, ldi, rnafeat, gl, gd = load_data(rel_matrix, args.cuda)
        gnnq = GNNq()
        gnnp = GNNp()
        if args.cuda:
            gnnq = gnnq.cuda()
            gnnp = gnnp.cuda()
            
        rel_matrix_tensor = torch.tensor(np.array(rel_matrix).astype(np.float32))
        train(gnnq, gnnp, rnafeat, gdi.t(), rel_matrix_tensor, args.epochs, 0.8, i)
        gnnq.eval()
        gnnp.eval()
        yli, _, ydi, _ = gnnp(rel_matrix_tensor)
        resi = args.alpha * yli + (1 - args.alpha) * ydi.t()
        if args.cuda:
            ymat = resi.cpu().detach().numpy()
        else:
            ymat = resi.detach().numpy()

        S = ymat     
        prediction_matrix = S
    
        prediction_matrix_real = prediction_matrix.real
        result = pd.DataFrame(prediction_matrix_real)
        np.savetxt("./five_folds_prediction_output/Dataset5/GMNN2CD/GMNN2CD" + str(count) + ".csv", result, delimiter=",")
        count += 1