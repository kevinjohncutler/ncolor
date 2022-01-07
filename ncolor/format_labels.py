import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage import measure
import fastremap


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
