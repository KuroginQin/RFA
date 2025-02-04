# Randomized Feature Aggregation
# Evaluated by node position classification (NIC)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score

import argparse
import time
import random
import pickle
from utils import *

rand_seed_gbl = 0

def construct_indicator(y_score, y):
    # rank the labels by the scores directly
    num_label = y.sum(axis=1, dtype=np.int32)
    num_label = np.reshape(num_label, (-1, 1)) # ???
    #num_label = np.sum(y, axis=1, dtype=np.int)
    y_sort = np.fliplr(np.argsort(y_score, axis=1))
    #y_pred = np.zeros_like(y_score, dtype=np.int32)
    row, col = [], []
    for i in range(y_score.shape[0]):
        row += [i]*num_label[i, 0]
        col += y_sort[i, :num_label[i, 0]].tolist()
        #for j in range(num_label[i, 0]):
        #    y_pred[i, y_sort[i, j]] = 1
    y_pred = sp.csr_matrix(
            ([1]*len(row), (row, col)),
            shape=y.shape, dtype=np.bool_)

    return y_pred

def NPC_eva(emb, gnd, train_ratio=0.2, n_splits=10, random_state=0, C=1.):
    # ====================
    micro, macro = [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio, random_state=random_state)
    for trn_idx, tst_idx in shuffle.split(emb):
        #print(trn_idx.shape, tst_idx.shape)
        emb_trn, emb_tst = emb[trn_idx], emb[tst_idx]
        gnd_trn, gnd_tst = gnd[trn_idx], gnd[tst_idx]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr",
                    max_iter=5000),
                n_jobs=-1)
        clf.fit(emb_trn, gnd_trn)
        score = clf.predict_proba(emb_tst)
        pred_tst = construct_indicator(score, gnd_tst)
        mi = f1_score(gnd_tst, pred_tst, average="micro")
        ma = f1_score(gnd_tst, pred_tst, average="macro")
        #print('MICRO-F1 %.4f MACRO-F1 %.4f' % (mi, ma))
        micro.append(mi)
        macro.append(ma)
    # ==========
    mi_mean = np.mean(micro)
    mi_std = np.std(micro)
    ma_mean = np.mean(macro)
    ma_std = np.std(macro)
    # ==========
    #print('%d-FOLD TR %.1f MICRO-F1 %.4f~%.4f MACRO-F1 %.4f~%.4f'
    #      % (len(micro), train_ratio, mi_mean, mi_std, ma_mean, ma_std))
    print('%d-FOLD TR %.1f MICRO-F1 %.2f~(%.2f) MACRO-F1 %.2f~(%.2f)'
          % (len(micro), train_ratio, mi_mean*100, mi_std*100, ma_mean*100, ma_std*100))

    return mi_mean, mi_std, ma_mean, ma_std

if __name__ == '__main__':
    # ====================
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='ppi')
    parser.add_argument('--d', type=int, default=256)
    parser.add_argument('--tau', type=float, default=20)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--act', type=str, default='tanh') # exp, tanh
    parser.add_argument('--norm', type=int, default=1) # 0: n/a, 1: z-norm, 2: l2-norm
    parser.add_argument('--init', type=str, default=1) # 0: new rand init, 1: saved rand init
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
    alpha = 1.0
    tr_list = [0.2]  # Ratio of training set

    # ====================
    # Read graph topology
    pkl_file = open('data/%s_edges.pickle' % (data_name), 'rb')
    edges = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    # Read ground-truth for NPC
    pkl_file = open('data/%s_gnd.pickle' % (data_name), 'rb')
    gnd_sp = pickle.load(pkl_file)
    pkl_file.close()
    # ==========
    node_set = set()
    clus_set = set()
    for (node_idx, clus_idx) in gnd_sp:
        if node_idx not in node_set:
            node_set.add(node_idx)
        if clus_idx not in clus_set:
            clus_set.add(clus_idx)
    num_nodes = max(node_set) + 1
    num_clus = max(clus_set) + 1
    num_edges = len(edges)
    # ==========
    gnd = np.zeros((num_nodes, num_clus))
    for (node_idx, clus_idx) in gnd_sp:
        gnd[node_idx, clus_idx] = 1.0
    print('DATA %s #NODES %d #EDGES %d #CLUS %d' % (data_name, num_nodes, num_edges, num_clus))

    # ====================
    degs = [0 for _ in range(num_nodes)]
    for (src, dst) in edges:
        degs[src] += 1
        degs[dst] += 1
    deg_max = np.max(degs)
    # ==========
    src_idxs = []
    dst_idxs = []
    vals = []
    for (src, dst) in edges:
        # ==========
        v = alpha / (np.sqrt(degs[src] + tau) * np.sqrt(degs[dst] + tau))
        # ==========
        src_idxs.append(src)
        dst_idxs.append(dst)
        vals.append(v)
        # ==========
        src_idxs.append(dst)
        dst_idxs.append(src)
        vals.append(v)
    for idx in range(num_nodes):
        src_idxs.append(idx)
        dst_idxs.append(idx)
        vals.append(0.1)
    sup_sp = sp.csr_matrix((vals, (src_idxs, dst_idxs)), shape=(num_nodes, num_nodes))

    # ====================
    if init_flag == 1:
        pkl_file = open('rand_init/%s_%d_init.pickle' % (data_name, emb_dim), 'rb')
        emb = pickle.load(pkl_file)
        pkl_file.close()
    else:
        emb = get_rand_proj_mat(num_nodes, emb_dim, rand_seed=rand_seed_gbl)  # Rand feat input

    # ==========
    time_s = time.time()
    for t in range(num_steps):
        emb = sup_sp.dot(emb)
        # ==========
        if act_fun == 'exp':
            emb = np.exp(emb)
        elif act_fun == 'tanh':
            emb = np.tanh(emb)
        # ==========
        if norm_flag == 1:
            for j in range(emb_dim):  # Column-wise z-normalization
                crt_mean = np.mean(emb[:, j])
                crt_std = np.std(emb[:, j])
                emb[:, j] = (emb[:, j] - crt_mean) / crt_std
        elif norm_flag == 2:  # Row-wise l2-normalization
            emb = normalize(emb, 'l2')
    time_e = time.time()
    emb_time = time_e - time_s
    print('EMB TIME %f' % (time_e - time_s))

    # ====================
    # Normalization before evaluation
    '''
    if norm_flag == 1:
        for j in range(emb_dim):  # Column-wise z-normalization
            crt_mean = np.mean(emb[:, j])
            crt_std = np.std(emb[:, j])
            emb[:, j] = (emb[:, j] - crt_mean) / crt_std
    elif norm_flag == 2:  # Row-wise l2-normalization
        emb = normalize(emb, 'l2')
    '''
    # ====================
    for tr in tr_list:
        mi_mean, mi_std, ma_mean, ma_std = \
            NPC_eva(emb, gnd, train_ratio=tr, n_splits=10, random_state=rand_seed_gbl)
    print()
