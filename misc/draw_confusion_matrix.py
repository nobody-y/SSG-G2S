import numpy as np
import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import matplotlib.font_manager
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
# from sklearn.metrics import confusion_matrix
print_type = 'graph'
filename = 'conf_graph_freq_train_sub.pdf' 
node_names = ["on","standing on", "sitting on", "holding","eating","looking at", "near","along"]
figsize = [100,100] 
prd_dist = np.load('rel_dis.npy')
#prd_dist[0] = np.inf
prd_dist = prd_dist[1:]
print('prd_dist shape: ',prd_dist.shape)
#prd_dist = prd_dist[1:]
VG_dict = json.load(open('/mnt/hdd2/****/datasets/visual_genome/data/genome/VG-SGG-dicts.json','r'))
conf_mat = np.load('conf_mat_freq_train.npy')  
conf_mat[:,0] = 0
conf_mat[0,:] = 0
conf_mat[0,0] = 1.0
# conf_mat = conf_mat * (1.0 - np.eye(conf_mat.shape[0]))
# conf_mat = conf_mat + 1.0
conf_mat = conf_mat[1:,1:]
pred_list = ['on', 'parked on', 'standing on','sitting on','near', 'along', 'in front of']
pred2ind  = VG_dict['predicate_to_idx']
prd_dict = ['']*(len(pred2ind))
for pred_i in pred2ind.keys():
    prd_dict[pred2ind[pred_i]-1] = pred_i
#prd_dict[0]='NULL'
prd_dict = np.array(prd_dict)

cm_sum = np.sum(conf_mat, axis=1)
conf_mat_nor = conf_mat / (cm_sum.astype(float)[:,None]+1e-8) #* 100


# cm_sum_nor = np.sum(conf_mat_nor, axis=1)
# conf_mat_nor = conf_mat_nor / (cm_sum_nor.astype(float)[:,None]) * 100 
# conf_mat = conf_mat + np.eye(conf_mat.shape[0]) * 1e-8
# cm_sum = np.sum(conf_mat, axis=0)
# conf_mat_nor = conf_mat / (cm_sum.astype(float)[None,:]) * 100 

prd_dist_sort = (0 - prd_dist).argsort()
conf_mat_nor_sort = conf_mat_nor[prd_dist_sort,:]
conf_mat_nor_sort = conf_mat_nor_sort[:,prd_dist_sort]
#print(conf_mat_nor_sort.sum(0))
print(conf_mat_nor_sort.sum(1))
conf_mat_sort = conf_mat[prd_dist_sort,:]
conf_mat_sort = conf_mat_sort[:,prd_dist_sort]
prd_dist = prd_dist[prd_dist_sort]
prd_dict = prd_dict[prd_dist_sort]
print("prd_dict: ", prd_dict)
cm_sum_sort = cm_sum[prd_dist_sort]
cm_sum = cm_sum_sort
cm = conf_mat_sort

cm_perc = conf_mat_nor_sort #conf_mat_nor_sort #/ cm_sum.astype(float)*100
annot = np.empty_like(cm).astype(str)
print(cm_perc.sum(1))
nrows, ncols = cm.shape

for i in range(nrows):
    for j in range(ncols):
        p = cm_perc[i, j]
        c = cm[i, j]
        s = cm_sum[i]
        annot[i, j] = '%.1f%%\n%.1f/%d' % (p, c, s)
        
def select_nodes(node_list):
    node_ids = []
    node_names = []
    for i in range(len(prd_dict)):
        if prd_dict[i] in node_list:
            node_ids.append(i)
            node_names.append(prd_dict[i])
    node_ids = np.array(node_ids)
    return node_ids, node_names
        
node_ids, node_names = select_nodes(node_names)
cm_perc = cm_perc[node_ids, :][:,node_ids]

prd_dist = prd_dist[node_ids]
prd_dict = node_names
annot = annot[node_ids, :][:,node_ids]
print(cm_perc.shape)
        # annot[i, j] = '%.1f' % (p)
cm = pd.DataFrame(cm_perc, index=prd_dict, columns=prd_dict)

if print_type == 'matrix':
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=8)
    res = sns.heatmap(cm, annot=None, fmt='', ax=None, annot_kws={},cbar=False)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 64, rotation=0, family = 'Times New Roman')
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 64, rotation=90, family = 'Times New Roman') 
    plt.xlabel('Output', fontsize = 64) # x-axis label with fontsize 15
    plt.ylabel('Annotation', fontsize = 64) 
    plt.savefig(filename, pad_inches = 0, bbox_inches='tight')
    
if print_type == 'graph':
    #G = nx.generators.directed.random_k_out_graph(10, 3, 0.5)
    # cm = cm * (1.0* (cm>0))
    # cm = cm[:10,:10]
    G = nx.from_pandas_adjacency(cm, create_using=nx.DiGraph)
    pos = nx.layout.circular_layout(G)
    prd_dist_nor = prd_dist / prd_dist.sum()
    node_sizes = [i*100 for i in prd_dist_nor]
    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="blue")
    edges = nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="->",
        arrowsize=10,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        width=2,
    )
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='r')
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)

    ax = plt.gca()
    ax.set_axis_off()
    plt.savefig(filename, pad_inches = 0, bbox_inches='tight')
