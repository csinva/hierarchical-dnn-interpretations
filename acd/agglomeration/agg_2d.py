import sys

sys.path.append('../scores')
sys.path.append('../util')
import numpy as np
import tiling
import cd
from skimage import measure  # for connected components
from math import ceil
from scipy.signal import convolve2d
from copy import deepcopy


def sum_interactions(lists):
    comps_list = lists['comps_list']
    cluster_scores = lists['comp_scores_raw_list']
    tmp_scores_list = lists['scores_orig_raw']
    img_size = lists['scores_list'][0].shape[0]
    num_classes = cluster_scores[1][1].shape[0]
    scores_list = np.zeros((img_size, img_size, num_classes))
    for i in range(img_size):
        scores_list[i] = tmp_scores_list[i * img_size:(i + 1) * img_size]

    int_sum = 0
    int_sums = np.zeros(len(comps_list))
    for idx in range(len(comps_list)):
        cluster_list = np.sort(np.unique(comps_list[idx]))[1:]
        cluster_sums = np.zeros((int(np.max(comps_list[idx])) + 1, num_classes))
        for old_cluster in np.unique(comps_list[idx - 1]):
            if old_cluster != 0:
                for new_cluster in cluster_list:
                    if np.sum((comps_list[idx] == new_cluster) * (comps_list[idx - 1] == old_cluster)) > 0:
                        cluster_sums[new_cluster] += cluster_scores[idx - 1][old_cluster]
        for cluster in cluster_list:
            cluster_sums[cluster] += np.sum((comps_list[idx] == cluster)[:, :, None] * scores_list, axis=(0, 1))
        for cluster in cluster_list:
            int_sums[idx] += np.sum(np.abs(cluster_scores[idx][cluster] - cluster_sums[cluster]))
    return np.sum(int_sums), int_sums


# return scores (higher is better)
def get_scores(model, method, ims, im_torch=None, pred_ims=None, layer='softmax', model_type='mnist'):
    scores = []
    if method == 'cd':
        for i in range(ims.shape[0]):  # can use tqdm here, need to use batches
            scores.append(cd.cd(np.expand_dims(ims[i], 0), im_torch, model, model_type)[0].data.cpu().numpy())
        scores = np.squeeze(np.array(scores))
    elif method == 'build_up':
        for i in range(ims.shape[0]):  # can use tqdm here, need to use batches
            scores.append(pred_ims(model, ims[i], layer)[0])
        scores = np.squeeze(np.array(scores))
    # scores = pred_ims(model, ims, layer)
    elif method == 'break_down':
        for i in range(ims.shape[0]):  # can use tqdm here, need to use batches
            scores.append(pred_ims(model, ims[i], layer)[0])
        scores = -1 * np.squeeze(np.array(scores))
    # scores = -1 * pred_ims(model, ims, layer)
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    return scores


# score doesn't have to just be prediction for label
def refine_scores(scores, lab_num):
    #     scores_non_lab = np.delete(scores, lab_num, axis=1)
    #     score_max2 = np.max(scores_non_lab, axis=1) # highest score for non-label class
    #     return scores[:, lab_num] - score_max2
    #     return np.abs(scores[:, lab_num])
    return scores[:, lab_num]


# higher scores are more likely to be picked
def threshold_scores(scores, percentile_include, method, use_abs=True):
    if use_abs:
        scores = np.abs(scores)
    X = scores

    # pick more when more is already picked
    num_picked = np.sum(np.isnan(scores))
    if num_picked > scores.size / 3:
        percentile_include -= 15

    thresh = np.nanpercentile(X, percentile_include)
    #     thresh = np.max(X) # pick only 1 pixel at a time
    im_thresh = np.logical_and(scores >= thresh, ~np.isnan(scores))
    # scores >= thresh #np.logical_and(scores >= thresh, scores != 0)

    # make sure we pick something
    while (np.sum(im_thresh) == 0):
        percentile_include -= 4
        thresh = np.nanpercentile(X, percentile_include)
        #     thresh = np.max(X) # pick only 1 pixel at a time
        im_thresh = np.logical_and(scores >= thresh, ~np.isnan(scores))
        # np.logical_and(scores >= thresh, scores != 0)
    return im_thresh


# if 3 sides of a pixel are selected, also select the pixel
filt = np.zeros((3, 3))
filt[:, 1] = 1  # middle column
filt[1, :] = 1  # middle row


def smooth_im_thresh(im_thresh_old, im_thresh):
    im = im_thresh_old + im_thresh
    im_count_neighbors = convolve2d(im, filt, mode='same')
    pixels_to_add = np.logical_and(np.logical_not(im), im_count_neighbors >= 3)
    return im + pixels_to_add


# establish correspondence between segs
def establish_correspondence(seg1, seg2):
    seg_out = np.zeros(seg1.shape, dtype='int64')
    new_counter = 0

    num_segs = int(np.max(seg2))
    remaining = list(range(1, 12))  # only have 10 colors though
    for i in range(1, num_segs + 1):
        seg = seg2 == i
        old_seg = seg1[seg]
        matches = np.unique(old_seg[old_seg != 0])
        num_matches = matches.size

        # new seg
        if num_matches == 0:
            new_counter -= 1
            seg_out[seg] = new_counter

        # 1 match
        elif num_matches == 1:
            seg_out[seg] = matches[0]
            remaining.remove(matches[0])

        # >1 matches (segs merged)
        else:
            seg_out[seg] = min(matches)
            remaining.remove(min(matches))

    # assign new segs    
    while new_counter < 0:
        seg_out[seg_out == new_counter] = min(remaining)
        remaining.remove(min(remaining))
        new_counter += 1

    return seg_out  # seg2


# agglomerate - black out selected pixels from before and resweep over the entire image
def agglomerate(model, pred_ims, percentile_include, method, sweep_dim,
                layer, im_orig, lab_num, use_abs, num_iters=5, im_torch=None, batch=False, model_type='mnist'):
    # set up shapes
    R = im_orig.shape[0]
    C = im_orig.shape[1]
    size_downsampled = (ceil(R / sweep_dim), ceil(C / sweep_dim))  # effectively downsampled

    # get scores
    if not batch:
        tiles = tiling.gen_tiles(im_orig, fill=0, method=method, sweep_dim=sweep_dim)
        scores_orig_raw = get_scores(model, method, ims=tiles, im_torch=im_torch,
                                     pred_ims=pred_ims, layer=layer, model_type=model_type)
        scores_track = np.copy(refine_scores(scores_orig_raw, lab_num)).reshape(
            size_downsampled)  # keep track of these scores
    else:
        scores_track = np.copy(batch_scores(model, im_torch, im_orig, lab_num))

    # threshold im
    im_thresh = threshold_scores(scores_track, percentile_include, method, use_abs)

    # initialize lists
    scores_list = [np.copy(scores_track)]
    im_thresh_list = [im_thresh]
    comps_list = []
    if not method == 'cd':
        comp_scores_raw_list = [{0: get_scores(model, 'build_up',
                                               ims=np.expand_dims(im_orig, 0),  # score for full image
                                               im_torch=im_torch, pred_ims=pred_ims, layer=layer,
                                               model_type=model_type)[0]}]
    else:
        comp_scores_raw_list = [{0: get_scores(model, method,
                                               ims=np.expand_dims(np.ones(im_orig.transpose().shape), 0),
                                               # score for full image
                                               im_torch=im_torch, pred_ims=pred_ims, layer=layer,
                                               model_type=model_type)[0]}]
    comp_scores_raw_combined_list = []

    # iterate
    for step in range(num_iters):
        # if already selected all pixels then break
        if np.sum(im_thresh_list[-1]) == R * C:
            break

        # find connected components for regions
        comps = np.copy(measure.label(im_thresh_list[-1], background=0, connectivity=2))

        # establish correspondence
        if step > 0:
            comps_orig = np.copy(comps)
            try:
                comps = establish_correspondence(comps_list[-1], comps_orig)
            except:
                comps = comps_orig
        # plt.imshow(comps)
        # plt.show()

        comp_tiles = {}  # stores tiles corresponding to each tile
        if not method == 'cd':
            comps_combined_tile = np.zeros(shape=im_orig.shape)  # stores all comp tiles combined
        else:
            comps_combined_tile = np.zeros(shape=(R, C))  # stores all comp tiles combined
        comp_surround_tiles = {}  # stores tiles around comp_tiles
        comp_surround_idxs = {}

        # make tiles
        comp_nums = np.unique(comps)
        comp_nums = comp_nums[comp_nums > 0]  # remove 0
        for comp_num in comp_nums:
            if comp_num > 0:
                # make component tile
                comp_tile_downsampled = (comps == comp_num)
                comp_tiles[comp_num] = tiling.gen_tile_from_comp(im_orig, comp_tile_downsampled,
                                                                 sweep_dim, method)  # this is full size
                comp_tile_binary = tiling.gen_tile_from_comp(im_orig, comp_tile_downsampled,
                                                             sweep_dim, 'cd')  # this is full size
                #             print('comps sizes', comps_combined_tile.shape, comp_tiles[comp_num].shape)
                comps_combined_tile += comp_tiles[comp_num]

                # generate tiles and corresponding idxs around component
                comp_surround_tiles[comp_num], comp_surround_idxs[comp_num] = \
                    tiling.gen_tiles_around_baseline(im_orig, comp_tile_binary, method=method, sweep_dim=sweep_dim)

        # predict for all tiles
        comp_scores_raw_dict = {}  # dictionary of {comp_num: comp_score}
        for comp_num in comp_nums:
            # calculate predictions
            #             print(comp_num, 'shapes before scoring', comp_tiles[comp_num].shape, comps_combined_tile.shape, comp_surround_tiles[comp_num].shape)
            #             plt.imshow(comp_tiles[comp_num])
            #             plt.show()
            #             for i in range(comp_surround_tiles[comp_num].shape[0]):
            #                 plt.imshow(comp_surround_tiles[comp_num][i])
            #                 plt.show()
            tiles = np.concatenate((np.expand_dims(comp_tiles[comp_num], 0),  # baseline tile at 0
                                    np.expand_dims(comps_combined_tile, 0),  # combined tile at 1
                                    comp_surround_tiles[comp_num]))  # all others afterwards
            scores_raw = get_scores(model, method, ims=tiles, im_torch=im_torch,
                                    pred_ims=pred_ims, layer=layer, model_type=model_type)

            # decipher scores
            score_comp = np.copy(refine_scores(scores_raw, lab_num)[0])
            scores_tiles = np.copy(refine_scores(scores_raw, lab_num)[2:])

            # store the predicted class scores
            comp_scores_raw_dict[comp_num] = np.copy(scores_raw[0])
            score_comps_raw_combined = np.copy(scores_raw[1])

            # update pixel scores
            tiles_idxs = comp_surround_idxs[comp_num]
            for i in range(len(scores_tiles)):
                (r, c) = tiles_idxs[i]
                scores_track[r, c] = np.max(scores_tiles[i] - score_comp)  # todo: subtract off previous comp / weight?

        # get class preds and thresholded image
        scores_track[im_thresh_list[-1]] = np.nan
        im_thresh = threshold_scores(scores_track, percentile_include, method, use_abs)
        im_thresh_smoothed = smooth_im_thresh(im_thresh_list[-1], im_thresh)

        # add to lists
        scores_list.append(np.copy(scores_track))
        im_thresh_list.append(im_thresh_smoothed)
        comps_list.append(comps)
        comp_scores_raw_list.append(comp_scores_raw_dict)
        comp_scores_raw_combined_list.append(score_comps_raw_combined)

    # pad first image
    comps_list = [np.zeros(im_orig.shape)] + comps_list

    lists = {'scores_list': scores_list,  # float arrs of scores tracked over time (NaN for already picked)
             'im_thresh_list': im_thresh_list,  # boolean array of selected pixels over time
             'comps_list': comps_list,  # numpy arrs (each component is a different number, 0 for background)
             'comp_scores_raw_list': comp_scores_raw_list,  # dicts, each key is a number corresponding to a component
             'comp_scores_raw_combined_list': comp_scores_raw_combined_list,
             # arrs representing scores for all current comps combined
             'scores_orig_raw': scores_orig_raw}  # one arr with original scores of pixels
    lists['num_before_final'] = len(im_thresh_list)
    lists = agglomerate_final(lists, model, pred_ims, percentile_include, method, sweep_dim,
                              layer, im_orig, lab_num, use_abs, num_iters=5, im_torch=im_torch,
                              batch=batch, model_type=model_type)

    return lists


# agglomerate the final blobs
def agglomerate_final(lists, model, pred_ims, percentile_include, method, sweep_dim,
                      layer, im_orig, lab_num, use_abs, num_iters=5, im_torch=None,
                      batch=False, model_type='mnist'):
    # while multiple types of blobs
    while (np.unique(lists['comps_list'][-1]).size > 2):
        #     for q in range(3):
        comps = np.copy(lists['comps_list'][-1])
        comp_scores_raw_dict = deepcopy(lists['comp_scores_raw_list'][-1])

        # todo: initially merge really small blobs with nearest big blobs
        # if q == 0:

        # make tiles by combining pairs in comps
        comp_tiles = {}  # stores tiles corresponding to each tile
        for comp_num in np.unique(comps):
            if comp_num > 0:
                # make component tile
                comp_tile_downsampled = (comps == comp_num)
                comp_tiles[comp_num] = tiling.gen_tile_from_comp(im_orig, comp_tile_downsampled,
                                                                 sweep_dim, method)  # this is full size

        # make combined tiles
        comp_tiles_comb = {}
        for comp_num1 in np.unique(comps):
            for comp_num2 in np.unique(comps):
                if 0 < comp_num1 < comp_num2:
                    comp_tiles_comb[(comp_num1, comp_num2)] = tiling.combine_tiles(comp_tiles[comp_num1],
                                                                                   comp_tiles[comp_num2], method)

        # predict for all tiles
        comp_max_score_diff = -1e10
        comp_max_key_pair = None
        comp_max_scores_raw = None
        for key in comp_tiles_comb.keys():
            # calculate scores
            tiles = 1.0 * np.expand_dims(comp_tiles_comb[key], 0)
            scores_raw = get_scores(model, method, ims=tiles, im_torch=im_torch,
                                    pred_ims=pred_ims, layer=layer, model_type=model_type)

            # refine scores for correct class - todo this doesn't work with refine_scores
            score_comp = np.copy(refine_scores(scores_raw, lab_num)[0])
            #             score_orig = np.max(refine_scores(np.expand_dims(comp_scores_raw_dict[key[0]], 0), lab_num)[0],
            #                                 refine_scores(np.expand_dims(comp_scores_raw_dict[key[1]], 0), lab_num)[0])
            score_orig = max(comp_scores_raw_dict[key[0]][lab_num], comp_scores_raw_dict[key[1]][lab_num])
            score_diff = score_comp - score_orig

            # find best score
            if score_diff > comp_max_score_diff:
                comp_max_score_diff = score_diff
                comp_max_key_pair = key
                comp_max_scores_raw = np.copy(scores_raw[0])  # store the predicted class scores

        # merge highest scoring blob pair
        comps[comps == comp_max_key_pair[1]] = comp_max_key_pair[0]

        # update highest scoring blob pair score
        comp_scores_raw_dict[comp_max_key_pair[0]] = comp_max_scores_raw
        comp_scores_raw_dict.pop(comp_max_key_pair[1])

        # add to lists
        lists['comps_list'].append(comps)
        lists['comp_scores_raw_list'].append(comp_scores_raw_dict)
        lists['scores_list'].append(lists['scores_list'][-1])
        lists['im_thresh_list'].append(lists['im_thresh_list'][-1])
        lists['comp_scores_raw_combined_list'].append(lists['comp_scores_raw_combined_list'][-1])

    return lists
