import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import fastremap


try:
    from skimage import measure
    SKIMAGE_ENABLED = True 
except:
    SKIMAGE_ENABLED = False
# # Should work for 3D too. Could put into usigned integer form at the end... 
# # Also could use some parallelization 
# def format_labels(labels, clean=False, min_area=9):
#     """
#     Puts labels into 'standard form', i.e. background=0 and cells 1,2,3,...,N-1,N.
#     Optional clean flag: disconnect and disjoint masks and discard small masks beflow min_area. 
#     min_area default is 9px. 
#     """
    
#     # Labels are stored as a part of a float array in Cellpose, so it must be cast back here.
#     # some people also use -1 as background, so we must cast to the signed integar class. We
#     # can safely assume no 2D or 3D image will have more than 2^31 cells. Finally, cv2 does not
#     # play well with unsigned integers (saves to default uint8), so we cast to uint32. 
#     labels = labels.astype('int32') 
#     labels -= np.min(labels) 
#     labels = labels.astype('uint32') 
    
#     # optional cleanup 
#     if clean:
#         inds = np.unique(labels)
#         for j in inds[inds>0]:
#             mask = labels==j
#             lbl = measure.label(mask)                       
#             regions = measure.regionprops(lbl)
#             regions.sort(key=lambda x: x.area, reverse=True)
#             if len(regions) > 1:
#                 print('Warning - found mask with disjoint label.')
#                 for rg in regions[1:]:
#                     if rg.area <= min_area:
#                         labels[rg.coords[:,0], rg.coords[:,1]] = 0
#                         print('secondary disjoint part smaller than min_area. Removing it.')
#                     else:
#                         print('secondary disjoint part bigger than min_area, relabeling. Area:',rg.area, 
#                                 'Label value:',np.unique(labels[rg.coords[:,0], rg.coords[:,1]]))
#                         labels[rg.coords[:,0], rg.coords[:,1]] = np.max(labels)+1

#             rg0 = regions[0]
#             if rg0.area <= min_area:
#                 labels[rg0.coords[:,0], rg0.coords[:,1]] = 0
#                 print('Warning - found mask area less than', min_area)
#                 print('Removing it.')
    
#     if np.any(labels):
#         fastremap.renumber(labels,in_place=True) # convenient to have unit increments from 1 to N cells
#         labels = fastremap.refit(labels) # put into smaller data type if possible 
#     return labels


def format_labels(labels, clean=False, min_area=9, despur=False, verbose=False, ignore=False):
    """
    Puts labels into 'standard form', i.e. background=0 and cells 1,2,3,...,N-1,N.
    Optional clean flag: disconnect and disjoint masks and discard small masks beflow min_area. 
    min_area default is 9px. 
    Optional ignore flag: 0 is now 'ignore' and 1 is background. We do not want to shift 1->0 in that case. 
    """
    
    # Labels are stored as a part of a float array in Cellpose, so it must be cast back here.
    # some people also use -1 as background, so we must cast to the signed integar class. We
    # can safely assume no 2D or 3D image will have more than 2^31 cells. Finally, cv2 does not
    # play well with unsigned integers (saves to default uint8), so we cast to uint32. 
    labels = labels.astype('int32')
    background = np.min(labels) 
    if not ignore:
        if verbose:
            print('minimum value is {}, shifting to 0'.format(background))
        labels -= background
        background = 0
    labels = labels.astype('uint32') 
    
    # optional cleanup 
    if clean:
        inds = np.unique(labels)
        for j in inds[inds>background]:
            mask = labels==j
            if despur:
                mask = delete_spurs(mask) #needs updating for ND 
            
            if SKIMAGE_ENABLED:
                lbl = measure.label(mask)                       
                regions = measure.regionprops(lbl)
                regions.sort(key=lambda x: x.area, reverse=True)
                if len(regions) > 1:
                    if verbose:
                        print('Warning - found mask with disjoint label.')
                    for rg in regions[1:]:
                        if rg.area <= min_area:
                            labels[tuple(rg.coords.T)] = background
                            if verbose:
                                print('secondary disjoint part smaller than min_area. Removing it.')
                        else:
                            if verbose:
                                print('secondary disjoint part bigger than min_area, relabeling. Area:',rg.area, 
                                        'Label value:',np.unique(labels[tuple(rg.coords.T)]))
                            labels[tuple(rg.coords.T)] = np.max(labels)+1
                            
                rg0 = regions[0]
                if rg0.area <= min_area:
                    labels[tuple(rg0.coords.T)] = background
                    if verbose:
                        print('Warning - found mask area less than', min_area)
                        print('Removing it.')
            else:
                connectivity_shape = np.array([3 for i in range(mask.ndim)])
                lbl = label(mask, connectivity=np.ones(connectivity_shape))[0]
                labels = lbl
        
    fastremap.renumber(labels,in_place=True) # convenient to have unit increments from 1 to N cells
    labels = fastremap.refit(labels) # put into smaller data type if possible 
    return labels
