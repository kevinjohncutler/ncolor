#4-color algorthm based on https://forum.image.sc/t/relabel-with-4-colors-like-map/33564 with extensions and improvements 

import numpy as np
from numba import njit
import scipy 
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import edt 
import random
import fastremap
from skimage import measure

def label(lab,n=4,conn=2):
    # needs to be in standard label form
    # but also needs to be in int32 data type to work properly; the formatting automatically
    # puts it into the smallest datatype to save space 
    lab = format_labels(lab).astype(np.int32) 
    idx = connect(lab, conn)
    idx = mapidx(idx)
    colors = render_net(idx, n=n, rand=10)
    lut = np.ones(lab.max()+1, dtype=np.uint8)
    for i in colors: lut[i] = colors[i]
    lut[0] = 0
    return lut[lab]

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
        if line[i]==0:continue
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
    key = (rst[:,0]<<16)
    key += rst[:,1]
    order = np.argsort(key)
    key[:] = key[order]
    diff = key[:-1]!=key[1:]
    idx = np.where(diff)[0]+1
    idx = np.hstack(([0], idx))
    return rst[order][idx]

def mapidx(idx):
    dic = {}
    for i in np.unique(idx): dic[i] = []
    for i,j in idx:
        dic[i].append(j)
        dic[j].append(i)
    return dic

# create a connection mapping 
def render_net(conmap, n=4, rand=12, depth=0, max_depth=5):
    thresh = 1e4
    if depth<max_depth:
        nodes = list(conmap.keys())
        np.random.seed(depth+1)
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
            colors = render_net(conmap,n+1,rand,depth+1,max_depth)
        return colors
    else:
        print('N-color algorthm exceeded max depth of',max_depth)
        return None
    
# Should work for 3D too. Could put into usigned integer form at the end... 
# Also could use some parallelization 
def format_labels(labels, clean=False, min_area=9):
    """
    Puts labels into 'standard form', i.e. background=0 and cells 1,2,3,...,N-1,N.
    Optional clean flag: disconnect and disjoint masks and discard small masks beflow min_area. 
    min_area default is 9px. 
    """
    
    # Labels are stored as a part of a float array in Cellpose, so it must be cast back here.
    # some people also use -1 as background, so we must cast to the signed integar class. We
    # can safely assume no 2D or 3D image will have more than 2^31 cells. Finally, cv2 does not
    # play well with unsigned integers (saves to default uint8), so we cast to uint32. 
    labels = labels.astype('int32') 
    labels -= np.min(labels) 
    labels = labels.astype('uint32') 
    
    # optional cleanup 
    if clean:
        inds = np.unique(labels)
        for j in inds[inds>0]:
            mask = labels==j
            lbl = measure.label(mask)                       
            regions = measure.regionprops(lbl)
            regions.sort(key=lambda x: x.area, reverse=True)
            if len(regions) > 1:
                print('Warning - found mask with disjoint label.')
                for rg in regions[1:]:
                    if rg.area <= min_area:
                        labels[rg.coords[:,0], rg.coords[:,1]] = 0
                        print('secondary disjoint part smaller than min_area. Removing it.')
                    else:
                        print('secondary disjoint part bigger than min_area, relabeling. Area:',rg.area, 
                                'Label value:',np.unique(labels[rg.coords[:,0], rg.coords[:,1]]))
                        labels[rg.coords[:,0], rg.coords[:,1]] = np.max(labels)+1

            rg0 = regions[0]
            if rg0.area <= min_area:
                labels[rg0.coords[:,0], rg0.coords[:,1]] = 0
                print('Warning - found mask area less than', min_area)
                print('Removing it.')
        
    fastremap.renumber(labels,in_place=True) # convenient to have unit increments from 1 to N cells
    labels = fastremap.refit(labels) # put into smaller data type if possible 
    return labels