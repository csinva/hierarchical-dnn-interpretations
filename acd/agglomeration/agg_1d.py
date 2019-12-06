import numpy as np
from ..util import tiling_1d as tiling
import torch
from skimage import measure
from ..scores import score_funcs


# threshold scores at a specific percentile
def threshold_scores(scores, percentile_include, absolute):
    # pick based on abs value?
    if absolute:
        scores = np.absolute(scores)

    # last 5 always pick 2
    num_left = scores.size - np.sum(np.isnan(scores))
    if num_left <= 5:
        if num_left == 5:
            percentile_include = 59
        elif num_left == 4:
            percentile_include = 49
        elif num_left == 3:
            percentile_include = 59
        elif num_left == 2:
            percentile_include = 49
        elif num_left == 1:
            percentile_include = 0
    thresh = np.nanpercentile(scores, percentile_include)
    mask = scores >= thresh
    return mask


# agglomerative sweep - black out selected pixels from before and resweep over the entire image
def agglomerate(model, batch, percentile_include, method, sweep_dim,
                label, num_iters=5, subtract=True, absolute=True, device='cuda'):
    # get original text and score
    text_orig = batch.text.data.cpu().numpy()
    score_orig = score_funcs.get_scores_1d(batch, model, method, label, only_one=True,
                                           score_orig=None, text_orig=text_orig, subtract=subtract, device=device)[0]

    # get scores
    texts = tiling.gen_tiles(text_orig, method=method, sweep_dim=sweep_dim)
    texts = texts.transpose()
    batch.text.data = torch.LongTensor(texts).to(device)
    scores = score_funcs.get_scores_1d(batch, model, method, label, only_one=False,
                                       score_orig=score_orig, text_orig=text_orig, subtract=subtract, device=device)

    # threshold scores
    mask = threshold_scores(scores, percentile_include, absolute=absolute)

    # initialize lists
    scores_list = [np.copy(scores)]
    mask_list = [mask]
    comps_list = []
    comp_scores_list = [{0: score_orig}]

    # iterate
    for step in range(num_iters):
        # find connected components for regions
        comps = np.copy(measure.label(mask_list[-1], background=0, connectivity=1))

        # loop over components
        comp_scores_dict = {}
        for comp_num in range(1, np.max(comps) + 1):

            # make component tile
            comp_tile_bool = (comps == comp_num)
            comp_tile = tiling.gen_tile_from_comp(text_orig, comp_tile_bool, method)

            # make tiles around component
            border_tiles = tiling.gen_tiles_around_baseline(text_orig, comp_tile_bool,
                                                            method=method,
                                                            sweep_dim=sweep_dim)

            # predict for all tiles
            # format tiles into batch
            tiles_concat = np.hstack((comp_tile, np.squeeze(border_tiles[0]).transpose()))
            batch.text.data = torch.LongTensor(tiles_concat).to(device)

            # get scores (comp tile at 0, others afterwards)
            scores_all = score_funcs.get_scores_1d(batch, model, method, label, only_one=False,
                                                   score_orig=score_orig, text_orig=text_orig, subtract=subtract,
                                                   device=device)
            score_comp = np.copy(scores_all[0])
            scores_border_tiles = np.copy(scores_all[1:])

            # store the predicted class scores
            comp_scores_dict[comp_num] = np.copy(score_comp)

            # update pixel scores
            tiles_idxs = border_tiles[1]
            for i, idx in enumerate(tiles_idxs):
                scores[idx] = scores_border_tiles[i] - score_comp

        # get class preds and thresholded image
        scores[mask_list[-1]] = np.nan
        mask = threshold_scores(scores, percentile_include, absolute=absolute)

        # add to lists
        scores_list.append(np.copy(scores))
        mask_list.append(mask_list[-1] + mask)
        comps_list.append(comps)
        comp_scores_list.append(comp_scores_dict)

        if np.sum(mask) == 0:
            break

    # pad first image
    comps_list = [np.zeros(text_orig.size, dtype=np.int)] + comps_list

    return {'scores_list': scores_list,  # arrs of scores (nan for selected)
            'mask_list': mask_list,  # boolean arrs of selected
            'comps_list': comps_list,  # arrs of comps with diff number for each comp
            'comp_scores_list': comp_scores_list,  # dicts with score for each comp
            'score_orig': score_orig}  # original score


'''
{'scores_list': scores_list, # arrs of scores (nan for selected)
'mask_list': mask_list, # boolean arrs of selected
'comps_list': comps_list, # arrs of comps with diff number for each comp
'comp_scores_list': comp_scores_list, # dicts with score for each comp
'score_orig':score_orig} # original score
'''


def collapse_tree(lists):
    num_iters = len(lists['comps_list'])
    num_words = len(lists['comps_list'][0])

    # need to update comp_scores_list, comps_list
    comps_list = [np.zeros(num_words, dtype=np.int) for i in range(num_iters)]
    comp_scores_list = [{0: 0} for i in range(num_iters)]
    comp_levels_list = [{0: 0} for i in range(num_iters)]  # use this to determine what level to put things at

    # initialize first level
    comps_list[0] = np.arange(num_words)
    comp_levels_list[0] = {i: 0 for i in range(num_words)}

    # iterate over levels
    for i in range(1, num_iters):
        comps = lists['comps_list'][i]
        comps_old = lists['comps_list'][i - 1]
        comp_scores = lists['comp_scores_list'][i]

        for comp_num in range(1, np.max(comps) + 1):
            comp = comps == comp_num
            comp_size = np.sum(comp)
            if comp_size == 1:
                comp_levels_list[i][comp_num] = 0  # set level to 0
            else:
                # check for matches
                matches = np.unique(comps_old[comp])
                num_matches = matches.size

                # if 0 matches, level is 1
                if num_matches == 0:
                    level = 1
                    comp_levels_list[i][comp_num] = level  # set level to level 1

                # if 1 match, maintain level
                elif num_matches == 1:
                    level = comp_levels_list[i - 1][matches[0]]


                # if >1 match, take highest level + 1
                else:
                    level = np.max([comp_levels_list[i - 1][match] for match in matches]) + 1

                comp_levels_list[i][comp_num] = level
                new_comp_num = int(np.max(comps_list[level]) + 1)
                comps_list[level][comp] = new_comp_num  # update comp
                comp_scores_list[level][new_comp_num] = comp_scores[comp_num]  # update comp score

    # remove unnecessary iters
    num_iters = 0
    while np.sum(comps_list[num_iters] > 0) and num_iters < len(comps_list):
        num_iters += 1

    # populate lists
    lists['comps_list'] = comps_list[:num_iters]
    lists['comp_scores_list'] = comp_scores_list[:num_iters]
    return lists
