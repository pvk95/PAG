import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import cc3d
import os
import sys

sys.path.append('./data/')
from data_utils import lab_info
from data_utils import alpha_cmap
import pickle

TH_MERGE_LABELS = 5
TH_MERGE_GT = 5
TH_HIT = 5
MIN_VOL_LAB = 25
TH_SEGMENT = 0.5


# ===========================================================================
# ============================= Data Loader  ================================
# ===========================================================================

def load_train_logs(save_dir):
    try:
        dirs = os.listdir(save_dir)
        model_name = None
        for f in dirs:
            if 'model' in f:
                model_name = f
                break

        train_logs = torch.load(os.path.join(save_dir, model_name), map_location='cpu')['train_logs']

    except KeyError:
        with open(os.path.join(save_dir, 'train_curves.pkl'), 'rb') as f:
            train_logs = pickle.load(f)

    return train_logs


def train_valid_test_idxs(save_dir):
    train_logs = load_train_logs(save_dir)

    train_idxs = train_logs['train_idxs']
    valid_idxs = train_logs['valid_idxs']
    test_idxs = train_logs['test_idxs']

    return train_idxs, valid_idxs, test_idxs


# ===========================================================================
# === Util functions for tumor detection by generation of bounding boxes  ===
# ===========================================================================


def get_bboxes(merge_dict, seg_predict):
    temp_dict = {}
    for k in merge_dict.keys():
        bb = {}
        points = merge_dict[k]

        bb['begin'] = np.min(points, axis=0)
        bb['end'] = 1 + np.max(points, axis=0)
        bb['center'] = np.mean(points, axis=0)
        bb['points'] = points

        vals = []
        for pp in points:
            vals.append(seg_predict[pp[0], pp[1], pp[2]])
        if len(vals) != 0:
            bb['confidence'] = np.mean(vals)
        else:
            bb['confidence'] = 0

        temp_dict[k] = bb

    return temp_dict


def merge_labels(image, pred=False):
    mask = image.copy()
    if pred:
        image = (image >= TH_SEGMENT).astype(np.int32)

    img_label = cc3d.connected_components(image.astype(np.int32))

    def get_dist(a, b):
        a = np.array(a)
        b = np.array(b)

        return np.sqrt(np.sum((a - b) ** 2))

    unique_labels = np.unique(img_label)
    unique_labels = unique_labels[1:]

    lab_pointer = {}
    merge_dict = {}

    center_positions = []
    for label in unique_labels:
        lab_points = np.where(img_label == label)
        center_positions.append([np.mean(z) for z in lab_points])
        merge_dict[label] = list(zip(*lab_points))
        lab_pointer[label] = label

    dist = np.zeros([len(center_positions)] * 2)
    equivalence = []
    for i in range(len(center_positions)):
        for j in range(i + 1, len(center_positions)):
            dist[i, j] = get_dist(center_positions[i], center_positions[j])
            dist[j, i] = dist[i, j]
            if dist[i, j] < TH_MERGE_LABELS and i != j:
                equivalence.append((unique_labels[i], unique_labels[j]))

    for eq in equivalence:
        idxs = (lab_pointer[eq[0]], lab_pointer[eq[1]])
        lab_modify = max(idxs)
        to_lab = min(idxs)
        lab_pointer[lab_modify] = to_lab

    for k in merge_dict.keys():
        if lab_pointer[k] != k:
            merge_dict[lab_pointer[k]] += merge_dict[k]
            # for p in points:
            #    img_label[p[0], p[1], p[2]] = to_lab

    for k in lab_pointer.keys():
        if lab_pointer[k] != k:
            merge_dict.pop(k, None)

    merge_dict = get_bboxes(merge_dict=merge_dict, seg_predict=mask)

    return merge_dict, equivalence, dist,


def get_gt_bbox(data_idx, meta_data, index_data):
    ex = index_data[data_idx]

    lab_points = meta_data[ex]['lab_points']
    label_names = meta_data[ex]['label_names']

    bboxes = {}

    for label in label_names:
        try:
            label_idx, tag, stage, comment = lab_info(label=label)
        except Exception:
            print(label_names)
            print(label)
            print(meta_data[index_data[data_idx]])
            continue
        try:
            points = lab_points[label_idx]
            if len(lab_points[label_idx]) > 0:
                bb = {'begin': np.min(points, axis=0), 'end': 1 + np.max(points, axis=0),
                      'center': np.mean(points, axis=0),
                      'label': label, 'tag': tag, 'points': points}
                bboxes[label] = bb
        except KeyError:
            print("{} label not found {}".format(data_idx, label))

    lab_pointer = {}

    for idx, k in enumerate(bboxes.keys()):
        lab_pointer[k] = idx

    dist = np.zeros([len(bboxes.keys())] * 2)

    for i, k1 in enumerate(bboxes.keys()):
        bb1 = bboxes[k1]
        for j, k2 in enumerate(bboxes.keys()):
            bb2 = bboxes[k2]
            d = get_dist_bb(bb1, bb2)
            dice = get_dice_coeff(bb1, bb2)
            dist[i, j] = d
            if (d < TH_MERGE_GT or dice > 0) and bb1['tag'] == bb2['tag']:
                val1 = lab_pointer[k1]
                val2 = lab_pointer[k2]

                lab_pointer[k1] = min(val1, val2)
                lab_pointer[k2] = min(val1, val2)

    group_labels = {}
    for k in lab_pointer.keys():
        idx = lab_pointer[k]
        if idx not in group_labels.keys():
            group_labels[idx] = [k]
        else:
            group_labels[idx].append(k)

    new_bboxes = {}
    for unique_idx, idx in enumerate(group_labels.keys()):
        labels = group_labels[idx]
        points = np.array([])

        unique_tag = None

        for label in labels:
            curr_points = bboxes[label]['points']
            tag = bboxes[label]['tag']

            assert curr_points.size > 0
            points = np.concatenate((points, curr_points)) if points.size else curr_points
            unique_tag = tag

        new_bboxes[unique_idx] = {'begin': np.min(points, axis=0), 'end': np.max(points, axis=0),
                                  'center': np.mean(points, axis=0),
                                  'labels': labels, 'tag': unique_tag}

    return new_bboxes


# ===========================================================================
# ========================== Analyze functions  =============================
# ===========================================================================


def get_dist_bb(bb1, bb2):
    return np.sqrt(np.sum((bb1['center'] - bb2['center']) ** 2))


def get_dice_coeff(bb1, bb2):
    for i in range(3):
        assert bb1['begin'][i] <= bb1['end'][i]
        assert bb2['begin'][i] <= bb2['end'][i]

    begin_coord = np.array([max(bb1['begin'][i], bb2['begin'][i]) for i in range(3)])
    end_coord = np.array([min(bb1['end'][i], bb2['end'][i]) for i in range(3)])

    if np.any(end_coord < begin_coord):
        return 0.0

    intersection_volume = np.prod(end_coord - begin_coord)

    bb1_volume = np.prod(bb1['end'] - bb1['begin'])
    bb2_volume = np.prod(bb2['end'] - bb2['begin'])

    iou = intersection_volume / (bb1_volume + bb2_volume - intersection_volume)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_precision_recall(true_positives, false_positives, false_negatives):
    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives) + 1e-10)
    recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives) + 1e-10)
    f1_score = 2 * recall * precision / (recall + precision + 1e-10)

    return precision, recall, f1_score


# ===========================================================================
# ========================== TP, FP, FN  ====================================
# ===========================================================================

def is_irrelevant(tag, labels, irrelevant_comments=('benign', 'other')):
    if tag == 'other-tag':
        return True
    elif tag == 'tumor-tag':
        for lb in labels:
            label_idx, tag, stage, comment = lab_info(label=lb)
            if comment in irrelevant_comments and len(labels) == 1:
                return True
    else:
        return False


def get_rates(bboxes, model_bboxes, threshold, tag_consider='all'):
    n_fp = 0
    n_fn = 0
    n_tp = 0

    indices_discard = []
    for k in model_bboxes.keys():
        bb = model_bboxes[k]
        bb['FP'] = None
        bb['discard'] = None
        if bb['confidence'] < threshold:
            indices_discard.append(k)
            bb['discard'] = True
            continue
        if len(bb['points']) < MIN_VOL_LAB:
            bb['discard'] = True
            continue
        if not len(bb['hit_by']):
            n_fp += 1
            bb['FP'] = True

    for k in bboxes.keys():
        bb1 = bboxes[k]
        bb1['TP'] = None
        bb1['FN'] = None
        bb1['discard'] = None

        if is_irrelevant(tag=bb1['tag'], labels=bb1['labels']):
            bb1['discard'] = True
            continue
        if tag_consider == 'all' or tag_consider == bb1['tag']:
            if not len(bb1['hit']):
                bb1['FN'] = True
                n_fn += 1
            elif np.all([aa in indices_discard for aa in list(zip(*bb1['hit']))[0]]):
                bb1['FN'] = True
                n_fn += 1
            else:
                bb1['TP'] = True
                n_tp += 1

    return n_tp, n_fp, n_fn


def get_hits(gt_cluster, model_cluster, data_indexes):
    temp_cluster = {}
    for data_idx in data_indexes:
        bboxes = gt_cluster[data_idx]
        model_bboxes = model_cluster[data_idx]
        for k in model_bboxes.keys():
            model_bboxes[k]['hit_by'] = []

        for k1 in bboxes.keys():
            bb1 = bboxes[k1]
            bb1['hit'] = []

            for k2 in model_bboxes.keys():
                bb2 = model_bboxes[k2]
                dice = get_dice_coeff(bb1, bb2)
                d = get_dist_bb(bb1=bb1, bb2=bb2)
                if dice > 0 or d < TH_HIT:
                    bb1['hit'].append((k2, dice, d))
                    bb2['hit_by'].append((k1, dice, d))
        '''            
        for k1 in bboxes.keys():
            bb1 = bboxes[k1]
            if not len(bb1['hit']):
                for k2 in model_bboxes.keys():
                    bb2 = model_bboxes[k2]
                    if bb2['hit_by'] is None:
                        d = get_dist_bb(bb1=bb1, bb2=bb2)
                        if d < TH_HIT:
                            bb1['hit'] = (k2, None, d)
                            bb2['hit_by'] = (k1, None, d)
        '''

        temp_cluster[data_idx] = model_bboxes

    for k in temp_cluster.keys():
        model_cluster[k] = temp_cluster[k]

    return gt_cluster, model_cluster


# ===========================================================================
# ================================ APIs  ====================================
# ===========================================================================

def discard(tag, comment):
    if tag == 'other-tag':
        return True
    elif tag != 'other-tag' and (comment == 'benign' or comment == 'other'):
        return True
    else:
        # print(tag, comment)
        return False


def reconstruct_mask(seg_predict, tag_consider, meta_data, index_data, data_indexes, data):
    img_shape = seg_predict.shape
    seg_mask = np.zeros(shape=img_shape, dtype=np.float32)

    for i, data_idx in enumerate(data_indexes):

        md = meta_data[index_data[data_idx]]
        label_names = md['label_names']
        lab_points = md['lab_points']

        for lb in label_names:
            label_idx, label_tag, label_stage, label_comment = lab_info(lb)
            if label_tag == tag_consider:
                if discard(label_tag, label_comment):
                    continue
                # print(label_idx, label_tag, label_stage, label_comment)
                points = lab_points[label_idx]
                if len(points) > 0:
                    for p in points:
                        seg_mask[i, 0, p[0], p[1], p[2]] = 1
        '''
        _, _, recall = dice_score(seg_predict=seg_predict[i, ...], seg_mask=data[i, :, :, :, :, -1])
        recall_this = tag_overlap_recall(seg_predict=seg_predict[i, ...], seg_mask=seg_mask[i, ...])
        try:
            # assert np.all(seg_mask[i, ...] == data[i, :, :, :, :, -1])
            assert np.allclose(recall, recall_this, atol=0.01)
        except AssertionError:
            # print("Assertion error for index: {}", data_idx)
            print("Data idx- {}: {}   {}".format(data_idx, recall, recall_this))
            assert not np.all(seg_mask[i, ...] == data[i, :, :, :, :, -1])
        '''

    tag_recall = tag_overlap_recall(seg_predict=seg_predict, seg_mask=seg_mask)
    return tag_recall


def tag_overlap_recall(seg_predict, seg_mask, eps=1e-10):
    seg_predict = np.array(seg_predict >= TH_SEGMENT, dtype=np.float)
    intersection = np.sum(seg_predict * seg_mask)
    den2 = np.sum(seg_mask)
    recall = intersection / (den2 + eps)

    return recall


def dice_score(seg_predict, seg_mask, eps=1e-10):
    seg_predict = np.array(seg_predict >= TH_SEGMENT, dtype=np.float)
    intersection = np.sum(seg_predict * seg_mask)
    den1 = np.sum(seg_predict)
    den2 = np.sum(seg_mask)
    dice = (2 * intersection) / (den1 + den2 + eps)

    tp = intersection

    fp = np.sum((1 - seg_mask) * seg_predict)

    fn = np.sum(seg_mask * (1 - seg_predict))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return dice, precision, recall


def sub_scores(tag_consider, threshold, data_indexes, gt_cluster, model_cluster):
    tps = []
    fps = []
    fns = []

    for data_idx in data_indexes:
        bboxes = gt_cluster[data_idx]
        model_bboxes = model_cluster[data_idx]

        tp, fp, fn = get_rates(model_bboxes=model_bboxes, bboxes=bboxes,
                               threshold=threshold, tag_consider=tag_consider)

        tps.append(tp)
        fps.append(fp)
        fns.append(fn)

    return np.array(tps), np.array(fps), np.array(fns)


def get_metrics(gt_cluster, model_cluster, data_indexes, tag='all', froc=False):
    fp_rate = []
    tp_rate = []

    precision_points = []
    recall_points = []

    thresholds = np.linspace(0.01, 0.99, 50)
    # thresholds = [model_cluster[k]['confidence'] for k in model_cluster.keys()]

    for th in thresholds:
        tps, fps, fns = sub_scores(tag_consider=tag, model_cluster=model_cluster,
                                   gt_cluster=gt_cluster, threshold=th, data_indexes=data_indexes)

        total_tumors = fns + tps

        if np.sum(total_tumors) == 0:
            continue

        tp_rate.append(np.mean(np.sum(tps) / np.sum(total_tumors)))

        fp_rate.append(np.mean(fps))

        pr, rc, _ = get_precision_recall(tps, fps, fns)
        precision_points.append(pr)
        recall_points.append(rc)

    fp_rate = np.array(fp_rate)
    tp_rate = np.array(tp_rate)

    x_idx = np.argsort(fp_rate)
    fp_rate = fp_rate[x_idx]
    tp_rate = tp_rate[x_idx]
    interp_vals = np.interp(x=[0.125, 0.25, 0.5, 1, 2, 4, 8], xp=fp_rate, fp=tp_rate)
    score = np.mean(interp_vals)

    precision_points = np.array(precision_points)
    recall_points = np.array(recall_points)
    idxs = np.where(recall_points != 0)[0]
    precision_points = precision_points[idxs]
    recall_points = recall_points[idxs]

    auprc = np.trapz(precision_points[::-1], recall_points[::-1])

    froc_points = None
    if froc:
        froc_points = (fp_rate, tp_rate)

    return score, auprc, precision_points, recall_points, froc_points


# ===========================================================================
# ========================== Plot functions  ================================
# ===========================================================================


def plot_training_curves(save_dir):
    train_logs = load_train_logs(save_dir)

    train_losses = train_logs['losses']['train_losses']
    valid_losses = train_logs['losses']['valid_losses']

    nrows = len(train_losses.keys())
    fig, axes = plt.subplots(nrows, 1, figsize=(18, 10), dpi=300)
    axes = axes.ravel()

    for i, k in enumerate(train_losses.keys()):
        ax = axes[i]
        ax.plot(train_losses[k], label='train')
        ax.plot(valid_losses[k], label='valid')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(k)
        ax.set_title(k)
        ax.legend()
        ax.grid()

    plt.suptitle('Loss functions')
    plt.savefig(save_dir + 'train_valid.png', dpi=300)
    plt.close()


def plot_froc(fig_name, x_points, y_points):
    plt.figure(figsize=(10, 5), dpi=300)
    plt.title('Sensitivity vs FP/scan')
    plt.plot(x_points, y_points)
    plt.xlabel('False Positives per scan')
    plt.ylabel('Sensitivity')
    plt.grid()
    plt.savefig(fig_name, dpi=300)
    plt.close()


def plot_prc(precision_points, recall_points, auprc, fig_name):
    plt.figure(figsize=(10, 5), dpi=300)
    plt.title('AUPRC: {}'.format(round(auprc, 3)))
    plt.plot(recall_points, precision_points)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.savefig(fig_name, dpi=300)


# ===========================================================================
# ========================== Render  ========================================
# ===========================================================================

def render(clusters, img, img_recon, pet_recon, seg_predict, data_idx, mov_name='mov.mp4'):
    def animate(i):
        ax1.set_data(img[0, :, :, i, 0])

        ax2.set_data(img[0, :, :, i, 0])
        ax2_2.set_data(img[0, :, :, i, 1])

        ax3.set_data(img[0, :, :, i, 0])
        ax3_3.set_data(img[0, :, :, i, 2])

        ax4.set_data(img_recon[0, :, :, i])

        # ax5.set_data(img[0, :, :, i, 0])
        ax5_5.set_data(pet_recon[0, :, :, i])

        ax6.set_data(img[0, :, :, i, 0])
        ax6_6.set_data(seg_predict[0, :, :, i])

        try:
            for j, k in enumerate(bboxes.keys()):
                bb = bboxes[k]
                x_begin, y_begin, z_begin = bb['begin']
                x_end, y_end, z_end = bb['end']

                if z_begin <= i <= z_end:
                    patches_rect[j].set_width(y_end - y_begin)
                    patches_rect[j].set_height(x_end - x_begin)
                    patches_rect[j].set_xy([y_begin, x_begin])

                    irrelevant = bb['discard']
                    irrelevant = True if irrelevant is not None and irrelevant else False

                    color = color_map[bb['tag']] if not irrelevant else 'w'
                    patches_rect[j].set_edgecolor(color=color)
                    if not irrelevant:
                        true_positive = True if bb['TP'] is not None and bb['TP'] else False
                        if true_positive:
                            assert bb['FN'] is None
                        else:
                            assert bb['FN'] is not None and bb['FN']
                        ls = '-' if true_positive else '--'
                        patches_rect[j].set_linestyle(ls=ls)

                else:
                    patches_rect[j].set_width(0)
                    patches_rect[j].set_height(0)
                    patches_rect[j].set_xy([0, 0])

            for j, k in enumerate(model_bboxes.keys()):
                bb = model_bboxes[k]
                x_begin, y_begin, z_begin = bb['begin']
                x_end, y_end, z_end = bb['end']

                if z_begin <= i <= z_end:
                    patches_rect_pred[j].set_width(y_end - y_begin)
                    patches_rect_pred[j].set_height(x_end - x_begin)
                    patches_rect_pred[j].set_xy([y_begin, x_begin])

                    if bb['discard'] is not None and bb['discard']:
                        color = 'w'
                    else:
                        fpc = True if bb['FP'] is not None and bb['FP'] else False
                        color = 'c' if not fpc else 'y'

                    patches_rect_pred[j].set_edgecolor(color=color)

                else:
                    patches_rect_pred[j].set_width(0)
                    patches_rect_pred[j].set_height(0)
                    patches_rect_pred[j].set_xy([0, 0])

        except IndexError:
            print("Something fishy!")
            pass

    seg_predict = np.array(seg_predict >= TH_SEGMENT, dtype=np.int)

    color_map = {'tumor-tag': 'r', 'node-tag': 'g', 'mets-tag': 'b', 'other-tag': 'w'}

    gt_cluster = clusters['gt_cluster']
    model_cluster = clusters['model_cluster']

    bboxes = gt_cluster[data_idx]
    model_bboxes = model_cluster[data_idx]

    tp, fp, fn = get_rates(model_bboxes=model_bboxes, bboxes=bboxes, threshold=TH_SEGMENT, tag_consider='all')

    fig, axes = plt.subplots(2, 3, figsize=(16, 5), dpi=100)
    plt.suptitle('TP: {}, FP: {}, FN: {}'.format(tp, fp, fn))
    axes = axes.ravel()

    for axz in axes:
        axz.axis('off')

    cmap, norm = alpha_cmap()

    axes[0].set_title('CT')
    ax1 = axes[0].imshow(img[0, :, :, 0, 0], cmap='gray')

    axes[1].set_title('CT + PET')
    ax2 = axes[1].imshow(img[0, :, :, 0, 0], cmap='gray')
    ax2_2 = axes[1].imshow(img[0, :, :, 0, 1], cmap='gist_heat', vmin=0, vmax=1, alpha=0.5)

    axes[2].set_title('CT + Mask')
    ax3 = axes[2].imshow(img[0, :, :, 0, 0], cmap='gray')
    ax3_3 = axes[2].imshow(img[0, :, :, 0, 0], cmap=cmap, norm=norm, alpha=0.5)

    axes[3].set_title('Image Recon')
    ax4 = axes[3].imshow(img[0, :, :, 0, 0], cmap='gray')

    axes[4].set_title('PET Recon')
    # ax5 = axes[4].imshow(img[0, :, :, 0, 0], cmap='gray')
    ax5_5 = axes[4].imshow(img[0, :, :, 0, 1], cmap='gist_heat', vmin=0, vmax=1)

    axes[5].set_title('CT + predicted mask')
    ax6 = axes[5].imshow(img[0, :, :, 0, 0], cmap='gray')
    ax6_6 = axes[5].imshow(img[0, :, :, 0, 0], cmap='gray', alpha=0.5)

    patches_rect = [patches.Rectangle((0, 0), 0, 0, facecolor='none', edgecolor='none', linewidth=2) for _ in
                    bboxes.keys()]
    patches_rect_pred = [patches.Rectangle((0, 0), 0, 0, facecolor='none', edgecolor='none', linewidth=2)
                         for _ in model_bboxes.keys()]

    [axes[2].add_patch(patch) for patch in patches_rect]
    [axes[5].add_patch(patch) for patch in patches_rect_pred]

    writer = animation.FFMpegWriter(fps=10, codec='libx264')
    ani = animation.FuncAnimation(fig, func=animate, frames=np.arange(1, seg_predict.shape[3]),
                                  interval=100, repeat_delay=1000)
    ani.save(mov_name, writer=writer)

    plt.close()

    return ani


'''
# data_path = '/'
data_path = './'

with open(os.path.join(data_path, 'meta_data.pkl'), 'rb') as my_dict:
    meta_data = pickle.load(my_dict)

with open(os.path.join(data_path, 'index_data.pkl'), 'rb') as my_dict:
    index_data = pickle.load(my_dict)

with open(os.path.join(data_path, 'clusters.pkl'), 'rb') as my_dict:
    clusters = pickle.load(my_dict)

model_cluster = clusters['model_cluster']
data_indexes = list(model_cluster.keys())
gt_cluster = get_gt_cluster(data_indexes)

gt_cluster, model_cluster = get_hits(gt_cluster=gt_cluster, model_cluster=model_cluster)

tag = 'all'

fp_rate = []
tp_rate = []

precision_points = []
recall_points = []

thresholds = np.linspace(0.5, 1, 50)

tp_arr = np.zeros(shape=(80, 50))
tt_arr = np.zeros(shape=(80, 50))

n_tumors = np.array([len(gt_cluster[k].keys()) for k in gt_cluster.keys()])

for ml, th in enumerate(thresholds):
    tps, fps, fns = sub_scores(tag_consider=tag, model_cluster=model_cluster,
                               gt_cluster=gt_cluster, threshold=th)

    total_tumors = fns + tps
    tp_arr[:, ml] = tps
    tt_arr[:, ml] = total_tumors

    if np.sum(total_tumors) == 0:
        continue

    tp_rate.append(np.mean(np.sum(tps) / np.sum(total_tumors)))

    fp_rate.append(np.mean(fps))

    pr, rc, _ = get_precision_recall(tps, fps, fns)
    precision_points.append(pr)
    recall_points.append(rc)

plt.plot(fp_rate, tp_rate)

precision_points = np.array(precision_points)
recall_points = np.array(recall_points)

idxs = np.where(recall_points != 0)[0]
precision_points = precision_points[idxs]
recall_points = recall_points[idxs]

plt.plot(recall_points, precision_points)

auprc = np.trapz(precision_points[::-1], recall_points[::-1])

auprc

tags_consider = ['all', 'tumor-tag', 'node-tag', 'mets-tag', 'other-tag']

metrics = np.zeros(shape=(len(data_indexes), 3, 5))
for i, t in enumerate(tags_consider):
    tps, fps, fns = sub_scores(tag_consider=t, model_cluster=model_cluster, gt_cluster=gt_cluster)
    metrics[:, 0, i] = tps
    metrics[:, 1, i] = fps
    metrics[:, 2, i] = fns

print(metrics.shape)

data_idx = -1
idx = data_indexes[data_idx]
model_bboxes = model_cluster[idx]
bboxes = gt_cluster[idx]

print(bboxes)
'''
