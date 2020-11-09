import torch
import json
import h5py
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
import colorsys
import random
import os
from graphviz import Digraph
import matplotlib.pyplot as plt
from maskrcnn_benchmark.layers import nms as box_nms
model_name = 'transformer_predcls_dist20_2k_FixPModel_lr1e3_B16_FixCMatDot'
pred_model_name = 'transformer_predcls_float32_epoch16_batch16'
pred_list = ["standing on", "sitting on", "looking at", "riding","holding", "eating"]
project_dir = ''
image_file = json.load(open('./datasets/vg/image_data.json'))
vocab_file = json.load(open('./datasets/vg/VG-SGG-dicts.json'))
pred2idx = vocab_file['predicate_to_idx']
pred_idx_list = []
for i in pred_list:
    pred_idx_list.append(pred2idx[i])
data_file = h5py.File('./datasets/vg/VG-SGG.h5', 'r')
# remove invalid image
corrupted_ims = [1592, 1722, 4616, 4617]
tmp = []
for item in image_file:
    if int(item['image_id']) not in corrupted_ims:
        tmp.append(item)
image_file = tmp

# load detected results
detected_origin_path = './checkpoints_best/'+str(model_name)+'/inference/VG_stanford_filtered_with_attribute_test/'
pre_detected_origin_path = './checkpoints_best/'+str(pred_model_name)+'/inference/VG_stanford_filtered_with_attribute_test/'
main_path = detected_origin_path
pred_output_path = pre_detected_origin_path + '/visualization/'
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder  ---: ", path)
    else:
        print("---  There is this folder!  ---")
# mkdir(output_path)

detected_origin_result = torch.load(main_path + 'eval_results.pytorch')
detected_info = json.load(open(main_path + 'visual_info.json'))

# get image info by index

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def get_info_by_idx(idx, det_input, thres=0.5):
    groundtruth = det_input['groundtruths'][idx]
    prediction = det_input['predictions'][idx]
    # image path
    img_path = detected_info[idx]['img_file']
    idx2label = vocab_file['idx_to_label']
    # boxes
    boxes = prediction.bbox
    boxes = groundtruth.bbox
    labels = groundtruth.get_field('labels')
    scores = one_hot_embedding(labels, len(idx2label)+1)
    keep_gt_indices = box_nms(boxes, scores, 0.8)
    keep_gt_indices = keep_gt_indices.numpy()
    gt_old2new = {}
    for i, keep_i in enumerate(keep_gt_indices):
        gt_old2new[keep_i] = i
    boxes = boxes[keep_gt_indices]
    labels = labels[keep_gt_indices]
    # object labels


    #labels = ['{}-{}'.format(idx,idx2label[str(i)]) for idx, i in enumerate(groundtruth.get_field('labels').tolist())]

    # pred_label_num = {}
    # pred_labels_list = prediction.get_field('pred_labels').tolist() #
    # pred_labels_new = []
    # for i in pred_labels_list:
    #     pred_tmp = idx2label[str(int(i))]
    #     if pred_tmp not in pred_label_num:
    #         pred_labels_new.append(pred_tmp)
    #         pred_label_num[pred_tmp] = 1
    #     else:
    #         pred_labels_new.append(str(pred_label_num[pred_tmp])+'-'+pred_tmp)
    #         pred_label_num[pred_tmp] = pred_label_num[pred_tmp] + 1
    pred_labels_list = labels.tolist()
    pred_labels = pred_labels_list
    #print(pred_labels)
    #pred_labels = ['{}-{}'.format(idx,idx2label[str(int(i))]) for idx, i in enumerate(prediction.get_field('pred_labels').tolist())]
    pred_scores = prediction.get_field('pred_scores')#.tolist()
    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    #gt_rels = groundtruth.get_field('relation_tuple').tolist()
    #gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    gt_rels = None
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs')
    pred_rel_label = prediction.get_field('pred_rel_scores')
    pred_rel_label[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)

    # mask = pred_rel_score > thres
    # pred_rel_score = pred_rel_score[mask]
    # pred_rel_label = pred_rel_label[mask]
 
    pred_rel_pair_tmp = []
    pred_rel_label_tmp = []
    pred_rel_score_tmp = []
    old2new_idx = {}
    for i,j,k in zip(pred_rel_pair, pred_rel_label, pred_rel_score):
        if int(i[0]) in keep_gt_indices and int(i[1]) in keep_gt_indices:
            pred_rel_score_tmp.append(k)
            pred_rel_label_tmp.append(j)
            pred_rel_pair_tmp.append(i.numpy())
    pred_rel_label = np.array(pred_rel_label_tmp)
    pred_rel_score = np.array(pred_rel_score_tmp)
    pred_rel_pair = np.array(pred_rel_pair_tmp)
    
    keep_box_idx = []
    #
    #print(np.sort(-pred_rel_score))
    pred_rel_sort_ind = pred_rel_score > 0.3
    pred_rel_sort_ind = np.argsort(-pred_rel_score)[:int(len(pred_rel_score)/3)]
    pred_rel_score = pred_rel_score[pred_rel_sort_ind]
    pred_rel_label = pred_rel_label[pred_rel_sort_ind]
    pred_rel_pair = pred_rel_pair[pred_rel_sort_ind]
    for i in pred_rel_pair:
        gt_new_i0 = gt_old2new[int(i[0])]
        gt_new_i1 = gt_old2new[int(i[1])]
        if gt_new_i0 not in keep_box_idx:
            old2new_idx[int(i[0])] = len(keep_box_idx)
            keep_box_idx.append(gt_new_i0)
        if gt_new_i1 not in keep_box_idx:
            old2new_idx[int(i[1])] = len(keep_box_idx)
            keep_box_idx.append(gt_new_i1)
    keep_box_idx = np.array(keep_box_idx)
    keep_boxes = boxes[keep_box_idx]
    keep_pred_labels = []
    for i in keep_box_idx:
        keep_pred_labels.append(pred_labels[i])
        
    pred_label_num = {}
    pred_labels_new = []
    for i in keep_pred_labels:

        pred_tmp = idx2label[str(int(i))]
        if pred_tmp not in pred_label_num:
            pred_labels_new.append(pred_tmp)
            pred_label_num[pred_tmp] = 1
        else:
            pred_labels_new.append(str(pred_label_num[pred_tmp])+'-'+pred_tmp)
            pred_label_num[pred_tmp] = pred_label_num[pred_tmp] + 1

    pred_labels = pred_labels_new

    pred_scores = pred_scores[keep_box_idx]
    pred_scores = pred_scores.tolist()
    pred_rels = [(pred_labels[old2new_idx[int(i[0])]], idx2pred[str(j)], pred_labels[old2new_idx[int(i[1])]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]
    pred_rels_idx = [(old2new_idx[int(i[0])], idx2pred[str(j)], old2new_idx[int(i[1])]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]

    return img_path, keep_boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rels_idx, pred_rel_score, pred_rel_label

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)
    if draw_info:
        draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info)


def print_list(name, input_list, scores):
    for i, item in enumerate(input_list):
        if scores == None:
            print(name + ' ' + str(i) + ': ' + str(item))
        else:
            print(name + ' ' + str(i) + ': ' + str(item) + '; score: ' + str(scores[i].item()))


def draw_image(img_path, boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rel_score, pred_rel_label,
               print_img=True):
    pic = Image.open(img_path)
    num_obj = boxes.shape[0]
    for i in range(num_obj):
        info = pred_labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
    if print_img:
        display(pic)
    if print_img:
        print('*' * 50)
        print_list('gt_boxes', labels, None)
        print('*' * 50)
        print_list('gt_rels', gt_rels, None)
        print('*' * 50)
    print_list('pred_labels', pred_labels, pred_rel_score)
    print('*' * 50)
    print_list('pred_rels', pred_rels, pred_rel_score)
    print('*' * 50)

    return None


def draw_scene_graph(img_name, labels, pred_rels=None, pred_rels_idx=None):
    """
    draw a graphviz graph of the scene graph topology
    """
    viz_labels = labels
    viz_rels = pred_rels
    viz_rels_idx = pred_rels_idx
    #s,p,o

    return draw_graph(img_name, viz_labels, viz_rels, viz_rels_idx)


def draw_graph(img_name, labels, rels, rels_idx):
    u = Digraph('sg', filename=output_path+img_name+'_sg.gv')
    u.body.append('size="6,6"')
    u.body.append('rankdir="LR"')
    u.node_attr.update(style='filled')

    name_list = []
    for i, l in enumerate(labels):
        u.node(str(i), label=l, color='#CCCCFF', shape='box')

    for rel, rel_idx in zip(rels, rels_idx):
        edge_key = '%s_%s' % (rel_idx[0], rel_idx[2])
        u.node(edge_key, label=rel[1], color='#FFCCCC', shape='ellipse')
        u.edge(str(rel_idx[0]), edge_key)
        u.edge(edge_key, str(rel_idx[2]))
    #u.view()
    u.render(output_path+img_name+'_sg.gv',format='pdf')

def _viz_box(img_name, im, rois, labels):

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    # draw bounding boxes
    for i, bbox in enumerate(rois):
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        label_str = labels[i]
        ax.text(bbox[0], bbox[1] - 4,
                label_str,
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=28, color='white')
    ax.axis('off')
    fig.tight_layout()
    print("output: ", output_path)
    plt.savefig(output_path+img_name+'.pdf')
    plt.close()

def draw_boxes_img(img_name, img_path, rois, labels):
    """
    visualize a scene graph on an image
    """
    viz_rois = rois
    viz_labels = labels
    im = Image.open(img_path)
    return _viz_box(img_name, im, viz_rois, viz_labels)

def show_selected(idx_list):
    for select_idx in idx_list:
        print(f'Image {select_idx}:')
        img_path, boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rels_idx, \
        pred_rel_score, pred_rel_label = get_info_by_idx(
            select_idx, detected_origin_result)
        draw_boxes_img(img_name=str(select_idx), img_path=img_path, rois=boxes, labels=pred_labels)
        draw_scene_graph(img_name=str(select_idx),labels=pred_labels, pred_rels=pred_rels, pred_rels_idx=pred_rels_idx)


def show_all(start_idx, length):
    for cand_idx in range(start_idx, start_idx + length):
        print(f'Image {cand_idx}:')
        img_path, boxes, labels, pred_labels, pred_scores, gt_rels, pred_rels, pred_rels_idx, \
        pred_rel_score, pred_rel_label = get_info_by_idx(
            cand_idx, detected_origin_result)
        draw_boxes_img(img_name=str(cand_idx), img_path=img_path, rois=boxes, labels=pred_labels)
        draw_scene_graph(img_name=str(cand_idx),labels=pred_labels, pred_rels=pred_rels, pred_rels_idx=pred_rels_idx)
        # draw_image(img_path=img_path, boxes=boxes, labels=labels, pred_labels=pred_labels, pred_scores=pred_scores,
        #            gt_rels=gt_rels, pred_rels=pred_rels, pred_rel_score=pred_rel_score, pred_rel_label=pred_rel_label,
        #            print_img=True)

def select_from_score():
    eval_results = torch.load(detected_origin_path+'result_dict.pytorch')
    pre_eval_results = torch.load(pre_detected_origin_path + 'result_dict.pytorch')

    recall_list = eval_results['predcls_recall'][20]
    recall_np = np.array(recall_list)
    pre_recall_list = pre_eval_results['predcls_recall'][20]
    pre_recall_np = np.array(pre_recall_list)
    dff_mean_recall = recall_np - pre_recall_np
    # pred_out = []
    # for i in pred_idx_list:
    #     mean_recall_np = np.array(eval_results['predcls_mean_recall_collect'][20][i])
    #     pre_mean_recall_np = np.array(pre_eval_results['predcls_mean_recall_collect'][20][i])
    #     #flag_pos = pre_mean_recall_np > 0.5
    #     dff_mean_recall = mean_recall_np - pre_mean_recall_np
    #     pred_out.append(np.argsort(-dff_mean_recall)[:20])
    pred_out = np.argsort(-pre_recall_np)[:500]
    return pred_out

if __name__ == "__main__":
    idx_list = select_from_score()
    #idx_list = [24778,24721,10146,18305,3771,6169,21583]
    #idx_list = [24778,18305]
    global output_path
    # for i, j in zip(pred_list, idx_list):
        # sub_name = i.split(' ')[0]
        # if len(i.split(' ')) > 1:
        #     for k in range(len(i.split(' '))):
        #         if k + 1 >= len(i.split(' ')):
        #             break
        #         sub_name = sub_name + '_' + i.split(' ')[k+1]
        #
        # print(sub_name)
        #output_path = detected_origin_path+'/visualization/' + sub_name + '/'
    output_path = main_path + '/visualization_bad/' #visualization_bad
    mkdir(output_path)
    show_selected(idx_list)
    # show_selected([119, 967, 713, 5224, 19681, 25371])