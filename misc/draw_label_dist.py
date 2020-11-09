import h5py
import numpy as np
import json
from collections import defaultdict
import matplotlib as mpl  
import matplotlib.pyplot as plt  
from scipy import stats 
import matplotlib.pylab as pylab
import seaborn as sns
from scipy import stats
import pandas as pd
vg_dict = json.load(open('/mnt/hdd2/guoyuyu/datasets/visual_genome/data/genome/VG-SGG-dicts-with-attri-info.json','r'))
pred_count = vg_dict['predicate_count']
pred_count_sort = sorted(pred_count.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)

name=['predicates','count']
test=pd.DataFrame(columns=name,data=pred_count_sort)
test.to_csv('predicate_dist.csv',encoding='gbk')
def draw_hist_from_dic(dict, name='None',step=5):
    fig_length = len(dict)
    params = {
        'axes.labelsize': '25',
        'xtick.labelsize': '45',
        'ytick.labelsize': '20',
        'lines.linewidth': '8',
        'legend.fontsize': '25',
        'figure.figsize': str(fig_length)+', 50'  # set figure size
    }
    pylab.rcParams.update(params)
    x = np.arange(len(dict))
    x_labels = []
    y_values = []
    plt.title(name)
    for i in dict:
        y_values.append(i[1])
        x_labels.append(i[0])
    plt.bar(x, y_values)
    plt.xticks(x, x_labels, rotation='vertical', weight=200)
    plt.savefig(name+'.pdf', dpi=200)
    plt.legend(loc='best')
    plt.close('all')
    return 0
if __name__ == "__main__":
    draw_hist_from_dic(dict=pred_count_sort, name='predicate_dist')
