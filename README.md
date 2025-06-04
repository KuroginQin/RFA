# Efficient Identity and Position Graph Embedding via Spectral-Based Random Feature Aggregation

This repository provides a reference implementation of RFA introduced in paper "Efficient Identity and Position Graph Embedding via Spectral-Based Random Feature Aggregation", which has been accepted by ACM KDD 2025.

### Abstract
Graph neural networks (GNNs), which capture graph structures via a feature aggregation mechanism following the graph embedding framework, have demonstrated powerful ability to support various tasks. According to the topology properties (e.g., structural roles or community memberships of nodes) to be preserved, graph embedding can be categorized into identity and position embedding. However, it is unclear for most GNN-based methods which property they can capture. Some of them may also suffer from low efficiency and scalability caused by several time- and space-consuming procedures (e.g., feature extraction and training). From a perspective of graph signal processing, we find that high- and low-frequency information in the graph spectral domain may characterize node identities and positions, respectively. Based on this investigation, we propose random feature aggregation (RFA) for efficient identity and position embedding, serving as an extreme ablation study regarding GNN feature aggregation. RFA (i) adopts a spectral-based GNN without learnable parameters as its backbone, (ii) only uses random noises as inputs, and (iii) derives embeddings via just one feed-forward propagation (FFP). Inspired by degree-corrected spectral clustering, we further introduce a degree correction mechanism to the GNN backbone. Surprisingly, our experiments demonstrate that two variants of RFA with high- and low-pass filters can respectively derive informative identity and position embeddings via just one FFP (i.e., without any training). As a result, RFA can achieve a better trade-off between quality and efficiency for both identity and position embedding over various baselines.

### Citing
```
@article{qin2025efficient,
  title={Efficient Identity and Position Graph Embedding via Spectral-Based Random Feature Aggregation},
  author={Qin, Meng and Liu, Jiahong and King, Irwin},
  journal={arXiv preprint arXiv:2505.20992},
  year={2025}
}
```

If you have any questions regarding this repository, you can contact the author via [mengqin_az@foxmail.com].

### Requirements
* numpy
* scipy
* pytorch
* scikit-learn

### Usage
The pro-processed large datasets (i.e., Flickr, Youtube, and Orkut) can be downloaded via this [link](https://drive.google.com/file/d/1XsRByLfl6dijv-DvWN4DYW7Cn5Qav6i8/view?usp=drive_link). Please unzip the file and put datasets under ./data.

To run CPU version of **RFA(H)** on USA:
```
python RFA_H_NIC.py --data_name usa --d 64 --tau 20 --K 7 --act exp --norm 1 --init 1
```
To run GPU version of **RFA(H)** on USA:
```
python RFA_H_NIC_gpu.py --data_name usa --d 64 --tau 20 --K 7 --act exp --norm 1 --init 1
```
To run CPU version of **RFA(H)** on Europe:
```
python RFA_H_NIC.py --data_name europe --d 64 --tau 20 --K 3 --act exp --norm 0 --init 1
```
To run GPU version of **RFA(H)** on Europe:
```
python RFA_H_NIC_gpu.py --data_name europe --d 64 --tau 20 --K 3 --act exp --norm 0 --init 1
```
To run CPU version of **RFA(H)** on Reality-Call:
```
python RFA_H_NIC.py --data_name reality-call --d 128 --tau 20 --K 2 --act exp --norm 0 --init 1
```
To run GPU version of **RFA(H)** on Reality-Call:
```
python RFA_H_NIC_gpu.py --data_name reality-call --d 128 --tau 20 --K 2 --act exp --norm 0 --init 1
```
To run CPU version of **RFA(H)** on Actor:
```
python RFA_H_NIC.py --data_name actor --d 128 --tau 20 --K 2 --act exp --norm 0 --init 1
```
To run GPU version of **RFA(H)** on Actor:
```
python RFA_H_NIC_gpu.py --data_name actor --d 128 --tau 20 --K 2 --act exp --norm 0 --init 1
```
To run CPU version of **RFA(H)** on Film:
```
python RFA_H_NIC.py --data_name film --d 256 --tau 10 --K 12 --act tanh --norm 2 --init 1
```
To run GPU version of **RFA(H)** on Film:
```
python RFA_H_NIC_gpu.py --data_name film --d 256 --tau 10 --K 12 --act tanh --norm 2 --init 1
```

To run CPU version of **RFA(L)** on PPI:
```
python RFA_L_NPC.py --data_name ppi --d 256 --tau 20 --K 10 --act tanh --norm 1 --init 1
```
To run GPU version of **RFA(L)** on PPI:
```
python RFA_L_NPC_gpu.py --data_name ppi --d 256 --tau 20 --K 10 --act tanh --norm 1 --init 1
```
To run CPU version of **RFA(L)** on BlogCatalog:
```
python RFA_L_NPC.py --data_name blogcatalog --d 512 --tau 0 --K 9 --act tanh --norm 1 --init 1
```
To run GPU version of **RFA(L)** on BlogCatalog:
```
python RFA_L_NPC_gpu.py --data_name blogcatalog --d 512 --tau 0 --K 9 --act tanh --norm 1 --init 1
```
To run CPU version of **RFA(L)** on Flickr:
```
python RFA_L_NPC.py --data_name flickr --d 512 --tau 1 --K 7 --act tanh --norm 1 --init 1
```
To run GPU version of **RFA(L)** on Flickr:
```
python RFA_L_NPC_gpu.py --data_name flickr --d 512 --tau 1 --K 7 --act tanh --norm 1 --init 1
```
To run CPU version of **RFA(L)** on Youtube:
```
python RFA_L_NPC_lrg.py --data_name youtube --d 128 --tau 10 --K 14 --act tanh --norm 2 --init 1
```
To run GPU version of **RFA(L)** on Youtube:
```
python RFA_L_NPC_lrg_gpu.py --data_name youtube --d 128 --tau 10 --K 14 --act tanh --norm 2 --init 1
```
To run CPU version of **RFA(L)** on Orkut:
```
python RFA_L_NPC_lrg.py --data_name orkut --d 64 --tau 20 --K 8 --act tanh --norm 1 --init 1
```
To run GPU version of **RFA(L)** on Orkut:
```
python RFA_L_NPC_lrg_gpu.py --data_name orkut --d 64 --tau 20 --K 8 --act tanh --norm 1 --init 1
```

Please note that different environment setups (e.g., CPU, GPU, memory size, versions of libraries and packages, etc.) may result in different evaluation results regarding the inference time. When testing the inference time, please also make sure that there are no other processes with heavy resource requirements (e.g., GPUs and memory) running on the same server. Otherwise, the evaluated inference time may not be stable.