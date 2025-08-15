#4-color algorthm based on https://forum.image.sc/t/relabel-with-4-colors-like-map/33564 with extensions and improvements 

import numpy as np
from numba import njit
import scipy
from .format_labels import format_labels
# from . import format_labels

import fastremap

# import edt # does not yet support return_indices
from scipy.ndimage import distance_transform_edt


def is_sequential(labels):
    return np.all(np.diff(fastremap.unique(labels))==1)
    

def unique_nonzero(labels):
    """
    Get unique nonzero labels.
    """
    sub = labels[labels > 0].astype(int)
    if not sub.flags.writeable:
        sub = sub.copy()
    if sub.size:
        return fastremap.unique(sub)
    else:
        return np.array([], dtype=labels.dtype)
        

        
# def unique_nonzero(labels):
#     """
#     Get unique nonzero labels. 
#     """
#     if np.any(labels):
#         sub = labels[labels > 0]
#         if not sub.flags.writeable:
#             sub = sub.copy()
#         return fastremap.unique(sub)
#     else:
#         return np.array([0])


def label(lab, n=4, conn=2, max_depth=5, offset=0, expand=None, return_n=False, greedy=False, verbose=False):
    # needs to be in standard label form
    # but also needs to be in int32 data type to work properly; the formatting automatically
    # puts it into the smallest datatype to save space
    # if not is_sequential(lab):
    #     lab = format_labels(lab)
    pad = 1
    unpad = tuple([slice(pad,-pad)]*lab.ndim)
    mask = lab!=0
    unique = unique_nonzero(lab)
    if verbose: print('number of masks', unique.size)
    
    if unique.size==1:
        ncl = mask.astype(int)
        nc = 1
    else:
        # by default, 2D images should be expanded, 3D should not
        # this allows expand to override either with True or False
        if expand or (lab.squeeze().ndim==2 and expand is None):
            lab = expand_labels(lab)
        # lab = np.pad(format_labels(lab),pad)
        lab = format_labels(np.pad(lab,pad),background=0)
        lut = get_lut(lab,n,conn,max_depth,offset,greedy)
        
        ncl = lut[lab][unpad]*mask
        nc = np.max(lut)
    
    if return_n: 
        return ncl, nc
    else:    
        return ncl

def get_lut(lab, n=4, conn=2, max_depth=5, offset=0, greedy=False):
    # lab = format_labels(lab).astype(np.int32)
    lab = format_labels(lab).astype(np.int64) 
     
    idx = connect(lab, conn)
    idx = mapidx(idx)
    # Handle case where no masks touch one another.
    # When the connection map is empty, assign the same color to every label.
    if not idx:
        lut = np.ones(lab.max() + 1, dtype=np.uint8)
        lut[0] = 0
        return lut
    if greedy:
        colors = greedy_coloring(idx)
    else:
        colors = render_net(idx, n=n, rand=10, max_depth=max_depth, offset=offset)
        
        if colors is None:
            raise ValueError(f"Failed to color the labels with {n} colors. Try increasing n or max_depth.")
        
    lut = np.ones(lab.max()+1, dtype=np.uint8)
    for i in colors: lut[i] = colors[i]
    lut[0] = 0
    return lut

    
def neighbors(shape, conn=1):
    dim = len(shape)
    block = scipy.ndimage.generate_binary_structure(dim, conn)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

# @njit(fastmath=True, cache=True)
@njit(cache=True)
def search(img, nbs):
    line = img.ravel()
    total = len(line)
    D = len(nbs)

    # Worst case: every pixel connects to every neighbor
    rst = np.empty((total * D, 2), img.dtype)
    s = 0

    for i in range(total):
        vi = line[i]
        if vi == 0:
            continue
        for d in nbs:
            j = i + d
            # Guard out-of-bounds indices (NumPy negative wraps; disallow here)
            if j < 0 or j >= total:
                continue
            vj = line[j]
            if vj == 0:
                continue
            if vi == vj:
                continue
            rst[s, 0] = vi
            rst[s, 1] = vj
            s += 1

    return rst[:s]
                            
def connect(img, conn=1):
    buf = np.pad(img, 1, 'constant')
    nbs = neighbors(buf.shape, conn)
    rst = search(buf, nbs)
    if len(rst)<2:
        return rst
    rst.sort(axis=1)
    key = (rst[:,0]<<16) #zero out
    key += rst[:,1]
    order = np.argsort(key)
    key[:] = key[order]
    diff = key[:-1]!=key[1:]
    idx = np.where(diff)[0]+1
    idx = np.hstack(([0], idx))  
    return rst[order][idx]

# maybe replace this with fastremap
# import fastremap
# def mapidx(idx):
#     dic = {}
#     # for i in np.unique(idx): dic[i] = []
#     for i in fastremap.unique(idx): dic[i] = [] # marginally faster 
#     for i,j in idx:
#         dic[i].append(j)
#         dic[j].append(i)
#     return dic
    
def mapidx(idx):
    # Stack idx and its reversed version to account for both directions
    idx_rev = idx[:, [1, 0]]
    idx_all = np.vstack((idx, idx_rev))

    # Sort idx_all by the first column (i)
    order = np.argsort(idx_all[:, 0])
    idx_all_sorted = idx_all[order]

    i = idx_all_sorted[:, 0]
    j = idx_all_sorted[:, 1]

    # Find unique 'i's and the indices where they occur
    unique_i, indices = fastremap.unique(i, return_index=True)

    # Split 'j' into lists according to the indices
    splits = np.split(j, indices[1:])

    # Build the dictionary mapping each 'i' to its list of neighbors
    dic = dict(zip(unique_i, splits))
    return dic

# create a connection mapping 
def render_net(conmap, n=4, rand=12, depth=0, max_depth=5, offset=0):
    # LARGE_INT = len(conmap)+1 # minimal to work, doesn't look as good?
    LARGE_INT = len(conmap)*2 # get back to previous behavior
    thresh = LARGE_INT
    if depth<max_depth:
        nodes = list(conmap.keys())
        np.random.seed(depth+1+offset)
        np.random.shuffle(nodes)
        colors = dict(zip(nodes, [0]*len(nodes)))
        counter = dict(zip(nodes, [0]*len(nodes)))
        count = 0
        while len(nodes)>0 and count<thresh:
            count+=1
            k = nodes.pop(0)
            counter[k] += 1
            hist = [LARGE_INT] + [0] * n
            for p in conmap[k]:
                hist[colors[p]] += 1
            if min(hist)==0:
                colors[k] = hist.index(min(hist))
                counter[k] = 0
                continue
            hist[colors[k]] = LARGE_INT
            minc = hist.index(min(hist))
            if counter[k]==rand:
                counter[k] = 0
                np.random.seed(count)
                minc = np.random.randint(1,n+1)
                
            colors[k] = minc
            for p in conmap[k]:
                if colors[p] == minc:
                    nodes.append(p)
        if count==thresh:
            # print(n,'-color algorthm failed,trying again with',n+1,'colors. Depth',depth)
            colors = render_net(conmap,n+1,rand,depth+1,max_depth, offset)
        return colors
        


def greedy_coloring(conmap):
    # faster and uses fewer colors than render_net
    colors = {}
    for node in conmap:
        neighbor_colors = {colors.get(neigh) for neigh in conmap[node] if neigh in colors}
        for color in range(1, len(conmap) + 1):
            if color not in neighbor_colors:
                colors[node] = color
                break
    return colors
    

def expand_labels(label_image):
    """
    Sped-up version of the scikit-image function just by dropping the distance thresholding. 
    Here we expand the labels into every background pixel. Can be over 40% faster. 
    """
    nearest_label_coords = distance_transform_edt(label_image==0, 
                                                  return_distances=False, 
                                                  return_indices=True)
    return label_image[tuple(nearest_label_coords)]
    
    

# attempts to not use njit
    
# def search2(img, nbs):
#     line = img.ravel()
#     len_line = len(line)
#     nz_indices = np.flatnonzero(line)  # Indices where line is non-zero

#     N = len(nz_indices)     # Number of non-zero elements
#     D = len(nbs)            # Number of neighbor offsets

#     # Create repeated indices for 'i' and 'j'
#     i_repeat = np.repeat(nz_indices, D)  # Shape: (N * D,)
#     d_tile = np.tile(nbs, N)             # Shape: (N * D,)
#     j = i_repeat + d_tile                # Neighbor indices, Shape: (N * D,)

#     # Filter out invalid neighbor indices
#     valid_mask = (j >= 0) & (j < len_line)
#     i_valid = i_repeat[valid_mask]
#     j_valid = j[valid_mask]

#     # Get the labels at the valid indices
#     line_i = line[i_valid]
#     line_j = line[j_valid]

#     # Apply the conditions:
#     # - Neighbor is non-zero
#     # - Labels are different
#     mask = (line_j != 0) & (line_i != line_j)

#     # Collect the valid pairs
#     pairs = np.column_stack((line_i[mask], line_j[mask]))

#     return pairs
    
    
# def search2(img, conn=1):
#     coords = np.array(np.nonzero(img))  # Convert to a NumPy array
#     npix = coords.shape[1]  # Number of non-zero pixels
#     dim = img.ndim
#     shape = img.shape

#     # Define neighbor offsets
#     from scipy.ndimage import generate_binary_structure
#     structure = generate_binary_structure(dim, conn)
#     structure[tuple([1]*dim)] = 0  # Remove the center
#     neighbor_offsets = np.array(np.nonzero(structure)) - 1  # Offsets relative to center
#     n_neighbors = neighbor_offsets.shape[1]

#     # Compute neighbor coordinates
#     # Expand coords to shape (dim, npix, 1)
#     coords_expanded = coords[:, :, np.newaxis]  # Shape: (dim, npix, 1)
#     # Broadcast neighbor_offsets to (dim, 1, n_neighbors) and add
#     neighbor_coords = coords_expanded + neighbor_offsets[:, np.newaxis, :]  # Shape: (dim, npix, n_neighbors)

#     # Reshape to 2D arrays for easier indexing
#     neighbor_coords = neighbor_coords.reshape(dim, -1)  # Shape: (dim, npix * n_neighbors)
#     center_coords = np.repeat(coords_expanded, n_neighbors, axis=2).reshape(dim, -1)  # Shape: (dim, npix * n_neighbors)

#     # Handle out-of-bounds coordinates
#     valid_mask = np.all((neighbor_coords >= 0) & (neighbor_coords < np.array(shape)[:, np.newaxis]), axis=0)

#     # Filter valid neighbor coordinates
#     valid_neighbor_coords = neighbor_coords[:, valid_mask]
#     valid_center_coords = center_coords[:, valid_mask]

#     # Map coordinates to flat indices
#     neighbor_indices = np.ravel_multi_index(valid_neighbor_coords, shape)
#     center_indices = np.ravel_multi_index(valid_center_coords, shape)

#     # Get labels at indices
#     line = img.ravel()
#     labels_center = line[center_indices]
#     labels_neighbor = line[neighbor_indices]

#     # Filter valid pairs
#     valid_pairs_mask = (labels_neighbor != 0) & (labels_neighbor != labels_center)

#     # Collect valid label pairs
#     pairs = np.column_stack((labels_center[valid_pairs_mask], labels_neighbor[valid_pairs_mask]))

#     return pairs
    
    
    # import fastremap

# def connect(img, conn=1):
#     buf = np.pad(img, 1, 'constant')
#     rst = search2(buf, conn)
#     if len(rst) < 2:
#         return rst
#     # Remove duplicates and sort the pairs
#     rst = fastremap.unique(np.sort(rst, axis=1), axis=0)
#     return rst

# using fastremap is a lot slower?
# def connect(img, conn=1):
#     buf = np.pad(img, 1, 'constant')
#     nbs = neighbors(buf.shape, conn)
#     rst = search(buf, nbs)
#     if len(rst) < 2:
#         return rst
#     rst.sort(axis=1)
#     print(rst.shape)
#     # Use np.unique to find unique rows (label pairs)
#     rst_unique = fastremap.unique(rst, axis=0)
#     return rst_unique

# import sys
# sys.modules[__name__] = label