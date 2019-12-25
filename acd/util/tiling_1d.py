import numpy as np


# pytorch needs to return each input as a column
def gen_tiles(text, fill=0,
              method='occlusion', prev_text=None, sweep_dim=1):
    '''
    Returns
    -------
    texts: np.ndarray
        batch_size x L
    '''
    L = text.shape[0]
    texts = np.zeros((L - sweep_dim + 1, L), dtype=np.int)
    for start in range(L - sweep_dim + 1):
        end = start + sweep_dim
        if method == 'occlusion':
            text_new = np.copy(text).flatten()
            text_new[start:end] = fill
        elif method == 'build_up' or method == 'cd':
            text_new = np.zeros(L)
            text_new[start:end] = text[start:end]
        texts[start] = np.copy(text_new)
    return texts

def gen_tile_from_comp(text_orig, comp_tile, method, fill=0):
    '''return tile representing component
    '''
    if method == 'occlusion':
        tile_new = np.copy(text_orig).flatten()
        tile_new[comp_tile] = fill
    elif method == 'build_up' or method == 'cd':
        tile_new = np.zeros(text_orig.shape)
        tile_new[comp_tile] = text_orig[comp_tile]
    return tile_new



def gen_tiles_around_baseline(text_orig, comp_tile, method='build_up', sweep_dim=1, fill=0):
    '''generate tiles around component
    '''
    L = text_orig.shape[0]
    left = 0
    right = L - 1
    while not comp_tile[left]:
        left += 1
    while not comp_tile[right]:
        right -= 1
    left = max(0, left - sweep_dim)
    right = min(L - 1, right + sweep_dim)
    tiles = []
    for x in [left, right]:
        if method == 'occlusion':
            tile_new = np.copy(text_orig).flatten()
            tile_new[comp_tile] = fill
            tile_new[x] = fill
        elif method == 'build_up' or method == 'cd':
            tile_new = np.zeros(text_orig.shape)
            tile_new[comp_tile] = text_orig[comp_tile]
            tile_new[x] = text_orig[x]
        tiles.append(tile_new)
    return np.array(tiles), [left, right]
