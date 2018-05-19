import numpy as np
from math import ceil


def combine_tiles(tile1, tile2, method='cd'):
    if not method == 'occlusion':
        return tile1 + tile2


def gen_tiles(image, fill=0, method='occlusion', prev_im=None,
              sweep_dim=1, num_ims=None, im_num_start=0):
    R = image.shape[0]
    C = image.shape[1]

    if image.ndim == 2:  # mnist case
        if num_ims is None:  # check if theres a limit on how many ims to have
            num_ims = ceil(R / sweep_dim) * ceil(C / sweep_dim)
        # print('sizes', R, C, num_ims)
        ims = np.empty((num_ims, R, C))
    else:  # imagenet case
        if num_ims is None:  # check if theres a limit on how many ims to have
            num_ims = ceil(R / sweep_dim) * ceil(C / sweep_dim)
        if method == 'cd':
            ims = np.empty((num_ims, R, C))
        else:
            ims = np.empty((num_ims, R, C, 3))

    i = 0
    # iterate over top, left indexes
    for rmin in range(0, R, sweep_dim):
        for cmin in range(0, C, sweep_dim):
            if im_num_start <= i < im_num_start + num_ims:

                # calculate bounds of box
                rmax = min(rmin + sweep_dim, R)
                cmax = min(cmin + sweep_dim, C)

                # create appropriate images
                if method == 'occlusion':
                    im = np.copy(image)
                    im[rmin:rmax, cmin:cmax] = fill  # image[r-1:r+1, c-1:c+1]
                    if not prev_im is None:
                        im[prev_im] = fill
                elif method == 'build_up':
                    im = np.zeros(image.shape)
                    im[rmin:rmax, cmin:cmax] = image[rmin:rmax, cmin:cmax]
                    if not prev_im is None:
                        im[prev_im] = image[prev_im]
                elif method == 'cd':
                    im = np.zeros((R, C))
                    im[rmin:rmax, cmin:cmax] = 1
                    if not prev_im is None:
                        im[prev_im] = 1
                ims[i - im_num_start] = np.copy(im)
            i += 1
    return ims


def gen_tiles_around_baseline(im_orig, comp_tile, fill=0,
                              method='occlusion', sweep_dim=3):
    R = im_orig.shape[0]
    C = im_orig.shape[1]
    dim_2 = (sweep_dim // 2)  # note the +1 for adjacent, but non-overlapping tiles
    ims, idxs = [], []
    # iterate over top, left indexes
    for r_downsampled, rmin in enumerate(range(0, R, sweep_dim)):
        for c_downsampled, cmin in enumerate(range(0, C, sweep_dim)):

            rmax = min(rmin + sweep_dim, R)
            cmax = min(cmin + sweep_dim, C)

            # calculate bounds of new block + boundaries
            rminus = max(rmin - sweep_dim, 0)
            cminus = max(cmin - sweep_dim, 0)
            rplus = min(rmin + sweep_dim, R - 1)
            cplus = min(cmin + sweep_dim, C - 1)

            # new block isn't in old block
            if not comp_tile[rmin, cmin]:
                # new block borders old block
                if comp_tile[rminus, cmin] or comp_tile[rmin, cminus] or comp_tile[rplus, cmin] or comp_tile[
                    rmin, cplus]:
                    if method == 'occlusion':
                        im = np.copy(im_orig)  # im_orig background
                        im[rmin:rmax, cmin:cmax] = fill  # black out new block
                        im[comp_tile] = fill  # black out comp_tile
                    elif method == 'build_up':
                        im = np.zeros(im_orig.shape)  # zero background
                        im[rmin:rmax, cmin:cmax] = im_orig[rmin:rmax, cmin:cmax]  # im_orig at new block
                        im[comp_tile] = im_orig[comp_tile]  # im_orig at comp_tile
                    elif method == 'cd':
                        im = np.zeros((R, C))  # zero background
                        im[rmin:rmax, cmin:cmax] = 1  # 1 at new block
                        im[comp_tile] = 1  # 1 at comp_tile
                    ims.append(im)
                    idxs.append((r_downsampled, c_downsampled))
    return np.array(ims), idxs


# generates full-size tile from comp which could be downsampled
# todo: upsample for things that aren't cd
def gen_tile_from_comp(im_orig, comp_tile_downsampled, sweep_dim, method, fill=0):
    R = im_orig.shape[0]
    C = im_orig.shape[1]
    if method == 'occlusion':
        im = np.copy(im_orig)
        #         im[comp_tile] = fill
        # fill in comp_tile with fill
        for r in range(comp_tile_downsampled.shape[0]):
            for c in range(comp_tile_downsampled.shape[1]):
                if comp_tile_downsampled[r, c]:
                    im[r * sweep_dim: (r + 1) * sweep_dim, c * sweep_dim: (c + 1) * sweep_dim] = fill

    elif method == 'build_up':
        im = np.zeros(im_orig.shape)
        #         im[comp_tile] = im_orig[comp_tile]
        # fill in comp_tile with im_orig
        for r in range(comp_tile_downsampled.shape[0]):
            for c in range(comp_tile_downsampled.shape[1]):
                if comp_tile_downsampled[r, c]:
                    im[r * sweep_dim: (r + 1) * sweep_dim, c * sweep_dim: (c + 1) * sweep_dim] = \
                        im_orig[r * sweep_dim: (r + 1) * sweep_dim, c * sweep_dim: (c + 1) * sweep_dim]

    elif method == 'cd':
        im = np.zeros((R, C), dtype=np.bool_)
        # fill in comp_tile with 1
        for r in range(comp_tile_downsampled.shape[0]):
            for c in range(comp_tile_downsampled.shape[1]):
                if comp_tile_downsampled[r, c]:
                    im[r * sweep_dim: (r + 1) * sweep_dim, c * sweep_dim: (c + 1) * sweep_dim] = 1
    return im
