import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from cycler import cycler
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from skimage.transform import resize
import random


# Create an N-bin discrete colormap from the specified input map
def discrete_cmap(N, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    nums = np.linspace(1 / N, 1, N)
    random.Random(10).shuffle(
        nums)  # shuffle in place so colors aren't consecutive, 9 for imagenet figs, now set for mnist figs
    nums[0] = 0
    color_list = base(nums)
    cmap_name = base.name + str(N)
    return color_list, base.from_list(cmap_name, color_list, N)


# cmap
# cmap = matplotlib.cm.Greys
cmap = matplotlib.cm.get_cmap('RdBu')
cmap.set_bad(color='#60ff16')  # bright green
N_COLORS = 11
cmap_comp = discrete_cmap(N_COLORS, 'jet')[1]
cmap_comp.set_under(color='#ffffff')  # transparent for lowest value


def visualize_ims_tiled(ims_tiled):
    # plt.figure(figsize=(6, 30))
    num_ims = 25  # len(ims_tiled)
    D = 5
    for i in range(D * (num_ims // D)):
        plt.subplot(D, num_ims // D, 1 + i)
        plt.imshow(ims_tiled[i], cmap=cmap, interpolation='None')
        plt.axis('off')
    plt.subplots_adjust(wspace=None, hspace=None)


def visualize_preds(preds, num, N=28, prev_im=None, cbar=True, vabs=None, axis_off=True):
    N = int(math.sqrt(preds.shape[0]))
    preds = preds[:, num].reshape(N, N)
    if not prev_im is None:
        preds[prev_im] = np.nan

    ax = plt.gca()

    if vabs is None:
        vmin = np.nanmin(preds)
        vmax = np.nanmax(preds)
        vabs = max(abs(vmin), abs(vmax))
    p = plt.imshow(preds, cmap=cmap,
                   vmin=-1 * vabs, vmax=vabs, interpolation='None')
    if axis_off:
        plt.axis('off')

    # colorbar
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        plt.colorbar(p, cax=cax)

    return p


def visualize_batch_preds(preds, prev_im=None, N=28, im_num_start=0):
    preds_reshaped = np.zeros(N * N)
    preds_reshaped[im_num_start: im_num_start + preds.size] = preds
    preds_reshaped = preds_reshaped.reshape(N, N)
    #     accs_reshaped = accs[:, num].reshape(N, N)
    if not prev_im is None:
        preds_reshaped[prev_im] = np.nan
    plt.imshow(preds_reshaped)
    return preds_reshaped


def visualize_ims_list(ims_list, title='', cmap_new=None, subplot_row=None, subplot_rows=3, colorbar=True, im_orig=None,
                       plot_overlay=False, mturk=False, num_ims=None, comp_scores_raw=None, lab_num_correct=None,
                       skip_first=False, mnist=False):
    im_segs = []
    if subplot_row is None:
        plt.figure(figsize=(12, 2), facecolor='white')
        subplot_row = 1
    if num_ims is None:
        num_ims = len(ims_list)
    for i in range(num_ims):
        if i >= len(ims_list):
            break
        ax = plt.subplot(subplot_rows, num_ims, num_ims * subplot_row + i + 1 - mnist)
        if cmap_new == 'redwhiteblue':
            vmin = min([np.min(im[np.logical_not(np.isnan(im))]) for im in ims_list])
            vmax = max([np.max(im[np.logical_not(np.isnan(im))]) for im in ims_list])
            vabs = max(abs(vmin), abs(vmax))

            p = plt.imshow(ims_list[i], cmap=cmap,
                           vmin=-1 * vabs, vmax=vabs, interpolation='nearest')
        else:
            # color images
            if plot_overlay:
                if not mnist:
                    plt.imshow(im_orig)  # plot image as background
                # overlay component comps
                if i > 0 or skip_first:
                    if mturk:

                        # need to map this to values of comps not comp_num           
                        im_nums = np.copy(ims_list[i]).astype(np.float32)
                        comp_to_score = comp_scores_raw[i]

                        for r in range(im_nums.shape[0]):
                            for c in range(im_nums.shape[1]):
                                comp_num = int(im_nums[r, c])
                                if comp_num > 0:
                                    im_nums[r, c] = comp_to_score[comp_num][lab_num_correct]

                        im = cmap(im_nums)
                        for r in range(im.shape[0]):
                            for c in range(im.shape[1]):
                                if im[r, c, 1] == 0:
                                    im[r, c, 3] = 0

                        vmin = min([comp_to_score[comp_num][lab_num_correct]
                                    for comp_to_score in comp_scores_raw[1:]
                                    for comp_num in comp_to_score.keys()])
                        vmax = max([comp_to_score[comp_num][lab_num_correct]
                                    for comp_to_score in comp_scores_raw[1:]
                                    for comp_num in comp_to_score.keys()])
                        vabs = max(abs(vmin), abs(vmax))
                    else:
                        # renumber to maintain right colors
                        #                         if i > 1:
                        #                             im_seg = establish_correspondence(ims_list[i-1], ims_list[i])
                        #                             ims_list[i] = im_seg
                        #                         else:
                        #                             im_seg = ims_list[i]

                        im_seg = ims_list[i]
                        im = cmap_comp(im_seg)
                        for r in range(im.shape[0]):
                            for c in range(im.shape[1]):
                                if im_seg[r, c] == 0:
                                    im[r, c, 3] = 0
                    map_reshaped = resize(im, (224, 224, 4), mode='symmetric', order=0)
                    if mturk:
                        plt.imshow(map_reshaped, alpha=0.9, interpolation='None', vmin=-1 * vabs, vmax=vabs)
                    else:
                        plt.imshow(map_reshaped, alpha=0.7)
            # not color
            else:
                p = plt.imshow(ims_list[i],
                               cmap=discrete_cmap(N_COLORS,  # len(np.unique(ims_list[i])) + 1,
                                                  'jet')[1], vmin=0, vmax=N_COLORS, interpolation='None')
                #                 plt.imshow(ims_list[i])
        if i > 0 or mturk:
            plt.axis('off')
        else:
            plt.axis('off')
            #             plt.ylabel(title)
            #             plt.yticks([])
            #             plt.xticks([])

    # colorbar
    if colorbar:
        plt.colorbar()
    # ax = plt.gca()
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="10%", pad=0.05)
    #     plt.colorbar(p, cax=cax)

    plt.subplots_adjust(wspace=0, hspace=0)


def visualize_dict_list(dict_list, method='break-down / build-up',
                        subplot_row=None, subplot_rows=3, lab_num=None, bar_graph=False):
    # if passed lab_num, plot only lab_num
    if lab_num is not None:
        dict_list_temp = []
        for d in dict_list:
            d_new = {}
            for key in d:
                d_new[key] = np.array(d[key][lab_num])
            dict_list_temp.append(d_new)
        dict_list = dict_list_temp

    if subplot_row is None:
        plt.figure(figsize=(12, 2), facecolor='white')
        subplot_row = 1
    num_ims = len(dict_list)
    preds_orig = dict_list[0][0]

    #     try:
    vmin = min([np.min(d[key]) for d in dict_list[1:] for key in d])
    vmax = max([np.max(d[key]) for d in dict_list[1:] for key in d])
    if lab_num is None:
        vmin = min(vmin, np.min(preds_orig))
        vmax = max(vmax, np.max(preds_orig))

    # plot 1st preds
    plt.subplot(subplot_rows, num_ims, num_ims * subplot_row + 1)
    #     plt.plot(preds_orig, '_', color='black')

    if lab_num is None:
        plt.bar(range(preds_orig.size), preds_orig, color='black')
        plt.ylabel('raw score full image')
    else:
        plt.ylabel('cd blob scores')
    plt.ylim((vmin, vmax))
    for i in range(1, num_ims):
        p = plt.subplot(subplot_rows, num_ims, num_ims * subplot_row + i + 1)
        # num_components = len(dict_list[i].keys())
        p.set_prop_cycle(cycler('color', discrete_cmap(N_COLORS, 'jet')[0][1:]))

        if bar_graph:
            region_nums = sorted(dict_list[i])
            vals = [dict_list[i][region_num] for region_num in region_nums]
            plt.bar(region_nums, vals, color=discrete_cmap(N_COLORS, 'jet')[0][1:])

            plt.plot(region_nums, vals, '_', color='black')
            plt.ylim((vmin - 1, vmax + 1))
        else:

            for region_num in sorted(dict_list[i]):
                region_arr = dict_list[i][region_num]
                #             for class_num in range(10):
                #                 print(class_num, region_arr[class_num])
                plt.plot(region_arr, '_', markeredgewidth=1)
                plt.ylim((vmin, vmax))

        cur_axes = plt.gca()
        # if not i == 0 and not i == 1:
        cur_axes.yaxis.set_visible(False)
        if lab_num is None:
            cur_axes.xaxis.set_ticklabels(np.arange(0, 10, 2))
            cur_axes.xaxis.set_ticks(np.arange(0, 10, 2))
            cur_axes.xaxis.grid()
        else:
            cur_axes.xaxis.set_visible(False)
        if i == 0:
            plt.ylabel('raw comp scores for ' + method)
    plt.subplots_adjust(wspace=0, hspace=0)


#     except Exception as e:
#         print('some empty plots', e)

def visualize_arr_list(arr_list, method='break-down / build-up',
                       subplot_row=None, subplot_rows=3):
    if subplot_row is None:
        plt.figure(figsize=(12, 2), facecolor='white')
        subplot_row = 1
    num_ims = len(arr_list) + 1

    vmin = min([np.min(d) for d in arr_list])
    vmax = max([np.max(d) for d in arr_list])

    for i in range(1, num_ims):
        p = plt.subplot(subplot_rows, num_ims, num_ims * subplot_row + i + 1)
        arr = arr_list[i - 1]
        #         plt.plot(arr, '_', markeredgewidth=0, color='black')
        plt.bar(np.arange(arr.size), arr, color='black')
        plt.ylim((vmin, vmax))
        cur_axes = plt.gca()
        if not i == 1:
            cur_axes.yaxis.set_visible(False)
        cur_axes.xaxis.set_ticklabels(np.arange(0, 10, 2))
        cur_axes.xaxis.set_ticks(np.arange(0, 10, 2))
        cur_axes.xaxis.grid()
        if i == 0:
            plt.ylabel('raw combined score for ' + method)
    plt.subplots_adjust(wspace=0, hspace=0)


def visualize_original_preds(im_orig, lab_num, comp_scores_raw_list, scores_orig_raw,
                             subplot_rows=5, dset=None, mturk=False, tits=None):
    num_cols = 7 - mturk
    plt.subplot(subplot_rows, num_cols, 1)
    plt.imshow(im_orig)
    if not tits is None:
        plt.title(tits[0])
    else:
        plt.title(dset.lab_dict[lab_num].split(',')[0])
    plt.axis('off')

    num_top = 5
    preds = comp_scores_raw_list[0][0]
    ind = np.argpartition(preds, -num_top)[-num_top:]  # top-scoring indexes
    ind = ind[np.argsort(preds[ind])][::-1]  # sort the indexes
    labs = [dset.lab_dict[x][:12] for x in ind]
    vals = preds[ind]

    # plotting
    if not mturk:
        plt.subplot(subplot_rows, num_cols, 2)
        idxs = np.arange(num_top)
        plt.barh(idxs, vals, color='#2ea9e888', edgecolor='#2ea9e888', fill=True, linewidth=1)

        for i, (val) in enumerate(zip(idxs, vals)):
            lab = str(labs[i])
            if 'puck' in lab:
                lab = 'puck'
            plt.text(s=str(lab), x=1, y=i, color="black", verticalalignment="center", size=10)
        # plt.text(s=str(pr)+"%", x=pr-5, y=i, color="w",
        #                      verticalalignment="center", horizontalalignment="left", size=18)
        ax = plt.gca()
        #         ax.set_yticklabels(labs)
        #         ax.set_yticks(np.arange(num_top))
        #         plt.yticks(rotation='horizontal')
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        plt.title('prediction logits')

    vmin = min([np.nanmin(scores_orig_raw[:, x]) for x in ind])  # preds[:, num]
    vmax = max([np.nanmax(scores_orig_raw[:, x]) for x in ind])  # preds[:, num]
    vabs = max(abs(vmin), abs(vmax))

    for i, x in enumerate(ind):
        if i < num_top:
            plt.subplot(subplot_rows, num_cols, i + 3 - mturk)
            if mturk:
                visualize_preds(scores_orig_raw, num=x, cbar=False, vabs=vabs)
                plt.title(dset.lab_dict[x][:14] + '...')
            else:
                visualize_preds(scores_orig_raw, num=x, cbar=False, vabs=vabs)
                if tits is not None:
                    plt.title(tits[i + 2])
                else:
                    plt.title('CD (' + dset.lab_dict[x][:10] + ')')  # +'\n'+ str(preds[x]))

    return ind, labs


def visualize_dict_list_top(dict_list, method='break-down / build-up',
                            subplot_row=None, subplot_rows=3, lab_num=None,
                            ind=None, labs=None, num_top=5, dset=None, use_orig_top=True,
                            num_ims=None, skip_first=False, vmin=None, vmax=None):
    if subplot_row is None:
        plt.figure(figsize=(12, 2), facecolor='white')
        subplot_row = 1
    if num_ims is None:
        num_ims = len(dict_list)
    preds_orig = dict_list[0][0]

    if vmin is None:
        vmin = min([np.min(d[key]) for d in dict_list[1:num_ims + 1] for key in d]) - 1
        vmax = max([np.max(d[key]) for d in dict_list[1:num_ims + 1] for key in d]) + 1

    for i in range(1, num_ims + skip_first):
        if i >= len(dict_list):
            break
        p = plt.subplot(subplot_rows, num_ims, num_ims * subplot_row + i + 1 - skip_first)
        # num_components = len(dict_list[i].keys())
        p.set_prop_cycle(cycler('color', discrete_cmap(N_COLORS, 'jet')[0][1:]))
        #         print('keys', dict_list[i].keys())

        for region_num in range(1, max(dict_list[i].keys()) + 1):
            #         for region_num in sorted(dict_list[i]):
            #             print('dict_list[i]', dict_list[i])

            if region_num in dict_list[i]:  # check if present
                if use_orig_top:
                    #                     print(region_num)
                    region_arr = dict_list[i][region_num][ind]
                    plt.plot(region_arr, '_', markeredgewidth=2)
                    plt.xticks(np.arange(region_arr.size), labs, rotation='vertical')
                    plt.xlim((-1, region_arr.size))
                else:
                    if region_num == 1:
                        region_arr = dict_list[i][region_num]
                        ind = np.argpartition(region_arr, -num_top)[-num_top:]  # top-scoring indexes
                        ind = ind[np.argsort(region_arr[ind])][::-1]  # sort the indexes
                        labs = [dset.lab_dict[x][:12] for x in ind]
                        vals = region_arr[ind]
                        plt.plot(vals, '_', markeredgewidth=1)
                        plt.xticks(np.arange(ind.size), labs, rotation='vertical')
                        plt.xlim((-1, ind.size))
                plt.ylim((vmin, vmax))
            else:  # plot blank just to match with color cycle
                plt.plot(-1, 0)
                pass

        cur_axes = plt.gca()
        if not i == 1:
            cur_axes.yaxis.set_visible(False)

            if use_orig_top:
                cur_axes.xaxis.set_visible(False)
                #         if i == 5:
                #             plt.title('raw comp scores for ' + method)
        else:
            plt.ylabel('patch importance')
    plt.subplots_adjust(wspace=0, hspace=0)


def visualize_top_classes(model, dset, im_orig, scores_orig_raw):
    preds = dset.pred_ims(model, im_orig)
    ind = np.argpartition(preds, -8)[-8:]  # top-scoring indexes
    ind = ind[np.argsort(preds[ind])][::-1]  # sort the indexes

    plt.figure(figsize=(14, 4))
    for i, x in enumerate(ind):
        plt.subplot(1, 8, i + 1)
        visualize_preds(scores_orig_raw, num=x)
        plt.title(dset.lab_dict[x][:12] + '\n' + str(preds[x]))


def visualize_original_preds_mnist(im_orig, lab_num, comp_scores_raw_list, scores_orig_raw,
                                   subplot_rows=5, dset=None, mturk=False, use_vmax=True):
    num_cols = 7 - mturk
    plt.subplot(subplot_rows, num_cols, 1)
    plt.imshow(im_orig, interpolation='None', cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    num_top = 5
    preds = comp_scores_raw_list[0][0]
    ind = np.argpartition(preds, -num_top)[-num_top:]  # top-scoring indexes
    ind = ind[np.argsort(preds[ind])][::-1]  # sort the indexes
    labs = ind  # [dset.lab_dict[x][:12] for x in ind]
    vals = preds[ind]

    # plotting
    if not mturk:
        plt.subplot(subplot_rows, num_cols, 2)
        idxs = np.arange(num_top)
        plt.barh(idxs, vals, color='#2ea9e888', edgecolor='#2ea9e888', fill=False, linewidth=1)
        for i, (val) in enumerate(zip(idxs, vals)):
            plt.text(s=str(labs[i]), x=1, y=i, color="black", verticalalignment="center", size=10)
        # plt.text(s=str(pr)+"%", x=pr-5, y=i, color="w",
        #                      verticalalignment="center", horizontalalignment="left", size=18)
        ax = plt.gca()
        #         ax.set_yticklabels(labs)
        #         ax.set_yticks(np.arange(num_top))
        #         plt.yticks(rotation='horizontal')
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.get_yaxis().set_visible(False)
        plt.title('logits')

    vmin = min([np.nanmin(scores_orig_raw[:, x]) for x in ind])  # preds[:, num]
    vmax = max([np.nanmax(scores_orig_raw[:, x]) for x in ind])  # preds[:, num]
    vabs = max(abs(vmin), abs(vmax))

    for i, x in enumerate(ind):
        if i < num_top:
            plt.subplot(subplot_rows, num_cols, i + 3 - mturk)
            if mturk:
                if use_vmax:
                    visualize_preds(scores_orig_raw, num=x, cbar=False, vabs=vabs)
                else:
                    visualize_preds(scores_orig_raw, num=x, cbar=False)
                plt.title(x)
            else:
                visualize_preds(scores_orig_raw, num=x, cbar=False, vabs=vabs)
                plt.title(x)

    return ind, labs
