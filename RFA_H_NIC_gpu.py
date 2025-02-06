# Randomized Feature Aggregation
# Evaluated by node identity classification (NIC)

import torch
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score

import argparse
import time
import random
import pickle
from utils import *

rand_seed_gbl = 0

torch.cuda.set_device(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def NIC_eva(emb, gnd, train_ratio=0.2, n_splits=10, random_state=0):
    # ====================
    micro, macro = [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio, random_state=random_state)
    for trn_idx, tst_idx in shuffle.split(emb):
        emb_trn, emb_tst = emb[trn_idx], emb[tst_idx]
        gnd_trn, gnd_tst = np.array(gnd)[trn_idx], np.array(gnd)[tst_idx]
        clf = LogisticRegression(solver='lbfgs', max_iter=5000, random_state=random_state)
        clf.fit(emb_trn, gnd_trn)
        clf_res = clf.predict(emb_tst)
        mi = f1_score(gnd_tst, clf_res, average="micro", zero_division=1)
        ma = f1_score(gnd_tst, clf_res, average="macro", zero_division=1)
        micro.append(mi)
        macro.append(ma)
    # ==========
    mi_mean = np.mean(micro)
    mi_std = np.std(micro)
    ma_mean = np.mean(macro)
    ma_std = np.std(macro)
    # ==========
    print('NIC %d-FOLD TR %.1f MICRO-F1 %.2f~(%.2f) MACRO-F1 %.2f~(%.2f)'
          % (len(micro), train_ratio, mi_mean*100, mi_std*100, ma_mean*100, ma_std*100))

    return mi_mean, mi_std, ma_mean, ma_std

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='usa')
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--tau', type=float, default=20)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--act', type=str, default='exp') # exp, tanh
    parser.add_argument('--norm', type=int, default=1) # 0: n/a, 1: z-norm, 2: l2-norm
    parser.add_argument('--init', type=int, default=1) # 0: new rand init, 1: saved rand init
    args = parser.parse_args()
    # ==========
    data_name = args.data_name # Dataset name
    emb_dim = args.d # Embedding dimensionality
    tau = args.tau # Degree correction
    num_steps = args.K # Number of aggregation steps
    act_fun = args.act # activation function - exp, tanh
    norm_flag = args.norm # Normalization flag - 0, 1, 2
    init_flag = args.init # Initialization flag

    # ===================
    alpha = - 1.0
    tr_list = [0.2] # Ratio of training set

    # ====================
    # Read graph topology
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    # Read ground-truth for NIC
    pkl_file = open('data/%s_gnd.pickle' % (data_name), 'rb')
    gnd = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    num_nodes = len(gnd)
    num_edges = len(edges)
    num_clus = np.max(gnd) + 1
    print('DATA %s #NODES %d #EDGES %d #CLUS %d' % (data_name, num_nodes, num_edges, num_clus))

    # ====================
    degs = [0 for _ in range(num_nodes)]
    for (src, dst) in edges:
        degs[src] += 1
        degs[dst] += 1
    deg_max = np.max(degs)
    # ==========
    idxs = []
    vals = []
    for (src, dst) in edges:
        # ==========
        v = alpha / (np.sqrt(degs[src] + tau) * np.sqrt(degs[dst] + tau))
        # ==========
        idxs.append((src, dst))
        vals.append(v)
        # ==========
        idxs.append((dst, src))
        vals.append(v)
    for idx in range(num_nodes):
        idxs.append((idx, idx))
        vals.append(0.1)
    idxs_tnr = torch.LongTensor(idxs).to(device)
    vals_tnr = torch.FloatTensor(vals).to(device)
    sup_sp = torch.sparse.FloatTensor(idxs_tnr.t(), vals_tnr,
                                       torch.Size([num_nodes, num_nodes])).to(device)

    # ====================
    if init_flag == 1:
        pkl_file = open('rand_init/%s_%d_init.pickle' % (data_name, emb_dim), 'rb')
        emb = pickle.load(pkl_file)
        pkl_file.close()
    else:
        emb = get_rand_proj_mat(num_nodes, emb_dim, rand_seed=rand_seed_gbl) # Rand feat input
    emb = torch.FloatTensor(emb).to(device)

    # ====================
    time_s = time.time()
    for t in range(num_steps):
        emb = torch.spmm(sup_sp, emb)
        # ==========
        if act_fun == 'exp':
            emb = torch.exp(emb)
        elif act_fun == 'tanh':
            emb = torch.tanh(emb)
        # ==========
        if norm_flag == 1:
            for j in range(emb_dim):  # Column-wise z-normalization
                crt_mean = torch.mean(emb[:, j])
                crt_std = torch.std(emb[:, j])
                emb[:, j] = (emb[:, j] - crt_mean) / crt_std
        elif norm_flag == 2: # Row-wise l2-normalization
            emb = F.normalize(emb, dim=1, p=2)
    time_e = time.time()
    emb_time = time_e - time_s
    print('EMB TIME %f' % (time_e - time_s))

    # ====================
    if torch.cuda.is_available():
        emb = emb.cpu().data.numpy()
    else:
        emb = emb.data.numpy()
    for tr in tr_list:
        mi_mean, mi_std, ma_mean, ma_std = \
            NIC_eva(emb, gnd, train_ratio=tr, n_splits=10, random_state=rand_seed_gbl)
    print()
