import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import join as oj
import math
import matplotlib.colors as colors


def visualize_scores(scores, label, text_orig, score_orig, sweep_dim=1, method='break_down'):
    plt.figure(figsize=(2, 10))
    try:
        p = scores.data.cpu().numpy()[:, label]
    except:
        p = scores

    # plot with labels
    text_orig = text_orig[sweep_dim - 1:]  # todo - don't do this, deal with edges better
    plt.barh(range(p.size), p[::-1], align='center', tick_label=text_orig[::-1])
    c = "pos" if label == 0 else "neg"
    plt.title(method + ' class ' + c + '\n(higher is more important)')  # pretty sure 1 is positive, 2 is negative


#     plt.show()

def print_scores(lists, text_orig, num_iters):
    text_orig = np.array(text_orig)
    print('score_orig', lists['score_orig'])

    print(text_orig)
    print(lists['scores_list'][0])

    # print out blobs and corresponding scores
    for i in range(1, num_iters):
        print('iter', i)
        comps = lists['comps_list'][i]
        comp_scores_list = lists['comp_scores_list'][i]

        # sort scores in decreasing order
        comps_with_scores = sorted(zip(range(1, np.max(comps) + 1),
                                       [comp_scores_list[i] for i in comp_scores_list.keys()]),
                                   key=lambda x: x[1], reverse=True)

        for comp_num, comp_score in comps_with_scores:
            print(comp_num, '\t%.3f, %s' % (comp_score, str(text_orig[comps == comp_num])))


def word_heatmap(text_orig, lists, label_pred, label, method, subtract=True, mturk=False, no_text=False, data=None):
    text_orig = np.array(text_orig)
    num_words = text_orig.size
    num_iters = len(lists['comps_list'])

    # populate data
    if data is None:
        data = np.empty(shape=(num_iters, num_words))
        data[:] = np.nan
        data[0, :] = lists['scores_list'][0]
        for i in range(1, num_iters):
            comps = lists['comps_list'][i]
            comp_scores_list = lists['comp_scores_list'][i]

            for comp_num in range(1, np.max(comps) + 1):
                idxs = comps == comp_num
                data[i][idxs] = comp_scores_list[comp_num]

    data[np.isnan(data)] = 0  # np.nanmin(data) - 0.001
    if num_iters == 1:
        plt.figure(figsize=(16, 1), dpi=300)
    else:
        plt.figure(figsize=(16, 3), dpi=300)

    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

            #     cmap = plt.get_cmap('RdBu') if label_pred == 0 else plt.get_cmap('RdBu_r')

    cmap = plt.get_cmap('RdBu')
    if label_pred == 1:
        data *= -1
    # cmap = matplotlib.cm.Greys
    # cmap.set_bad(color='black')
    #                    cmap='viridis')#'RdBu')
    abs_lim = max(abs(np.nanmax(data)), abs(np.nanmin(data)))

    c = plt.pcolor(data,
                   edgecolors='k',
                   linewidths=0,
                   norm=MidpointNormalize(vmin=abs_lim * -1, midpoint=0., vmax=abs_lim),
                   cmap=cmap)

    def show_values(pc, text_orig, data, fmt="%s", **kw):
        val_mean = np.nanmean(data)
        val_min = np.min(data)
        pc.update_scalarmappable()
        # ax = pc.get_axes()
        ax = pc.axes

        for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
            x, y = p.vertices[:-2, :].mean(0)
            # pick color for text
            if np.all(color[:3] > 0.5):  # value > val_mean: #value > val_mean: #
                color = (0.0, 0.0, 0.0)
            else:
                color = (1.0, 1.0, 1.0)
            x_ind = math.floor(x)
            y_ind = math.floor(y)

            # sometimes don't display text
            if y_ind == 0 or data[y_ind, x_ind] != 0:  # > val_min:
                ax.text(x, y, fmt % text_orig[x_ind],
                        ha="center", va="center",
                        color=color, fontsize=9, **kw)

    class_pred = 'pos' if label_pred == 0 else 'neg'
    class_actual = 'pos' if label == 0 else 'neg'

    show_values(c, text_orig, data)
    if not mturk:
        plt.title(method +
                  ' score_orig: ' + '{:.2f}'.format(lists['score_orig']) +
                  ' pred: ' + class_pred +
                  ' label: ' + class_actual +
                  ' subtract: ' + str(subtract))
        #         plt.ylabel('Tree level')
        plt.xlabel(' '.join(text_orig))
        cb = plt.colorbar(c, extend='both')  # fig.colorbar(pcm, ax=ax[0], extend='both')
        cb.outline.set_visible(False)
    plt.xlim((0, num_words))
    plt.ylim((0, num_iters))
    plt.yticks([])
    plt.plot([0, num_words], [1, 1], color='black')

    plt.xticks([])

    if no_text and not mturk:
        cb.ax.set_title('CD score')
        plt.title("")
        plt.xlabel("")
    # cb.set_ticks([])
    # clean up a lot of the viz
    if mturk:
        plt.title("")
        plt.ylabel("")
        plt.xlabel("")
