#4-color algorthm based on https://forum.image.sc/t/relabel-with-4-colors-like-map/33564 with extensions and improvements 

import numpy as np
from numba import njit
import scipy
from .format_labels import format_labels, is_sequential
from skimage.segmentation import expand_labels as skimage_expand_labels
import edt
from scipy.ndimage import distance_transform_edt


def label(lab, n=4, conn=2, max_depth=5, offset=0, expand=None, return_n=False):
    # needs to be in standard label form
    # but also needs to be in int32 data type to work properly; the formatting automatically
    # puts it into the smallest datatype to save space

    # if not is_sequential(lab):
    #     lab = format_labels(lab)
    pad = 1
    unpad = tuple([slice(pad,-pad)]*lab.ndim)
    mask = lab!=0
    
    # by default, 2D images should be expanded, 3D should not
    # this allows expand to override either with True or False
    if expand or (lab.squeeze().ndim==2 and expand is None):
        lab = expand_labels(lab)
    # lab = np.pad(format_labels(lab),pad)
    lab = format_labels(np.pad(lab,pad),background=0)
    lut = get_lut(lab,n,conn,max_depth,offset,return_n)
    
    nc = lut[lab][unpad]*mask
    
    if return_n: 
        return nc, np.max(lut)
    else:    
        return nc

def get_lut(lab, n=4, conn=2, max_depth=5, offset=0, return_n=False):
    lab = format_labels(lab).astype(np.int32) 
    idx = connect(lab, conn)
    idx = mapidx(idx)
    colors = render_net(idx, n=n, rand=10, max_depth=max_depth, offset=offset)
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

@njit(fastmath=True)
def search(img, nbs):
    s, line = 0, img.ravel()
    rst = np.zeros((len(line),2), img.dtype)
    for i in range(len(line)):
        if line[i]==0: continue
        for d in nbs:
            if line[i+d]==0: continue
            if line[i]==line[i+d]: continue
            rst[s,0] = line[i]
            rst[s,1] = line[i+d]
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
def mapidx(idx):
    dic = {}
    for i in np.unique(idx): dic[i] = []
    for i,j in idx:
        dic[i].append(j)
        dic[j].append(i)
    return dic

# create a connection mapping 
def render_net(conmap, n=4, rand=12, depth=0, max_depth=5, offset=0):
    thresh = 1e4
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
            hist = [1e4] + [0] * n
            for p in conmap[k]:
                hist[colors[p]] += 1
            if min(hist)==0:
                colors[k] = hist.index(min(hist))
                counter[k] = 0
                continue
            hist[colors[k]] = 1e4
            minc = hist.index(min(hist))
            if counter[k]==rand:
                counter[k] = 0
                np.random.seed(count)
                minc = np.random.randint(1,4)
            colors[k] = minc
            for p in conmap[k]:
                if colors[p] == minc:
                    nodes.append(p)
        if count==thresh:
            print(n,'-color algorthm failed,trying again with',n+1,'colors. Depth',depth)
            colors = render_net(conmap,n+1,rand,depth+1,max_depth, offset)
        return colors
    else:
        print('N-color algorthm exceeded max depth of',max_depth)
        return None
    
def expand_labels(label_image):
    """
    Sped-up version of the scikit-image function just by dropping the distance thresholding. 
    Here we expand the labels into every background pixel. Can be over 40% faster. 
    """
    nearest_label_coords = distance_transform_edt(label_image==0, 
                                                  return_distances=False, 
                                                  return_indices=True)
    return label_image[tuple(nearest_label_coords)]