import numpy as np
import fastremap

# Lazy imports for heavy dependencies - only import when functions are called
def _lazy_import_skimage_measure():
    from skimage import measure
    return measure

def _lazy_import_skimage_morphology():
    from skimage.morphology import remove_small_holes
    return remove_small_holes


# Module-level ExpandEngine singleton. Constructing one per call adds
# ~5-10 ms of pool-spinup overhead — matches the rationale in color.py's
# _SOLVER. Engine is thread-safe for sequential format_labels calls.
_FORMAT_ENGINE = None


def _get_format_engine():
    global _FORMAT_ENGINE
    if _FORMAT_ENGINE is None:
        from ._backend import ExpandEngine
        _FORMAT_ENGINE = ExpandEngine(-1)
    return _FORMAT_ENGINE


def format_labels(labels, clean=False, min_area=9, despur=False,
                  verbose=False, background=None, ignore=False,
                  first_seen=False):
    """
    Puts labels into 'standard form', i.e. background=0 and cells 1,2,3,...,N-1,N.
    Optional clean flag: disconnect and disjoint masks and discard small masks below min_area.
    min_area default is 9px.
    Optional ignore flag: 0 is now 'ignore' and 1 is background. We do not want to shift 1->0 in that case.

    first_seen (default False): when False, the new label assigned to a
    source value is its rank among present values (ascending-source,
    parallel build, fast). When True, labels are assigned in input scan
    order — matches ``fastremap.renumber`` bit-for-bit, serial build,
    ~2× slower. Set to True only if downstream code depends on the
    legacy fastremap ordering.
    """
    # Hot path: simple compaction (no cleanup, default min-shift) ->
    # cpp engine. Skips the numpy.copy + astype + np.min + fastremap
    # round-trip that this function used to do per call.
    if (not clean and not ignore and background is None and not verbose):
        try:
            eng = _get_format_engine()
        except ImportError:
            pass  # fall through to legacy path
        else:
            # Pass the original-dtype array straight to cpp. ExpandEngine
            # casts to int32 in parallel inside the released-GIL block via
            # cast_to_int32 — no numpy.astype + .copy() round-trip outside
            # the GIL release. Output uses ascending-source numbering: the
            # new label assigned to source value k is its rank among
            # present values. Differs from fastremap.renumber's
            # input-order numbering by a permutation, but both produce a
            # valid 1..N compaction; downstream callers treat label
            # values as opaque identifiers.
            arr = np.ascontiguousarray(labels)
            out, _n = eng.format_labels(arr, first_seen=bool(first_seen))
            return out

    # Legacy path: clean=True / ignore / custom background / verbose.
    # Labels are stored as a part of a float array in Cellpose, so it must be cast back here.
    # some people also use -1 as background, so we must cast to the signed integar class. We
    # can safely assume no 2D or 3D image will have more than 2^31 cells. Finally, cv2 does not
    # play well with unsigned integers (saves to default uint8), so we cast to uint32.
    labels = labels.copy()
    labels = labels.astype('int32') #uint vs int
    if background is None:
        background = np.min(labels)
    else:
        background = 0

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
                labels[mask] = 0 #clear old label
                mask = delete_spurs(mask) #needs updating for ND 
                labels[mask] = j # put label back in
                
            measure = _lazy_import_skimage_measure()
            lbl = measure.label(mask)
            regions = measure.regionprops(lbl)
            regions.sort(key=lambda x: x.area, reverse=True)
            
            # If no regions were found (e.g. the mask was completely removed by the
            # cleaning/despur operations), skip further processing for this label.
            if len(regions) == 0:
                continue
            
            if len(regions) > 1:
                if verbose:
                    print('Warning - found mask with disjoint label.')
                for rg in regions[1:]:
                    if rg.area < min_area:
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
    
    fastremap.renumber(labels,in_place=True) # convenient to have unit increments from 1 to N cells
    labels = fastremap.refit(labels) # put into smaller data type if possible 
    return labels

def delete_spurs(mask):
    pad = 1
    #must fill single holes in image to avoid cusps causing issues. Will limit to holes of size ___
    remove_small_holes = _lazy_import_skimage_morphology()
    skel = remove_small_holes(np.pad(mask,pad,mode='constant'),5)
    nbad = 1
    niter = 0
    while (nbad > 0):
        bad_points = endpoints(skel) 
        skel = np.logical_and(skel,np.logical_not(bad_points))
        nbad = np.sum(bad_points)
        niter+=1
    
    unpad =  tuple([slice(pad,-pad)]*skel.ndim)
    skel = skel[unpad] #unpad

    return skel

def endpoints_nd(skel):
    """
    Detect endpoints in an N-dimensional skeleton.

    An endpoint is a foreground pixel that has exactly one foreground neighbor
    in its connectivity neighborhood.

    For 2D: Uses 8-connectivity (includes diagonal neighbors) to match original behavior
    For 3D+: Uses face connectivity (more appropriate for higher dimensions)

    Parameters
    ----------
    skel : ndarray
        Binary skeleton image of any dimensionality

    Returns
    -------
    endpoints : ndarray
        Binary image with endpoints marked as True
    """
    from scipy import ndimage

    ndim = skel.ndim

    # Choose connectivity based on dimensionality
    # For 2D, use 8-connectivity to match the original mahotas implementation
    # For higher dimensions, use face connectivity
    if ndim == 2:
        connectivity = ndimage.generate_binary_structure(ndim, 2)  # 8-connectivity
    else:
        connectivity = ndimage.generate_binary_structure(ndim, 1)  # face connectivity

    # Create kernel (exclude center pixel)
    kernel = connectivity.astype(np.float32)
    center_idx = tuple(np.array(kernel.shape) // 2)
    kernel[center_idx] = 0

    # Count neighbors
    neighbor_count = ndimage.convolve(skel.astype(np.float32), kernel,
                                     mode='constant', cval=0)

    # Endpoints are foreground pixels with exactly 1 neighbor
    endpoints = (skel > 0) & (neighbor_count == 1)

    return endpoints

def endpoints_nd_alternative(skel):  # pragma: no cover
    """
    Alternative implementation using labeled components approach.
    This might be more accurate for complex skeleton structures.
    """
    from scipy import ndimage

    # Create connectivity structure
    ndim = skel.ndim
    connectivity = ndimage.generate_binary_structure(ndim, 1)

    # For each foreground pixel, count connected foreground neighbors
    # We'll use a more sophisticated approach with binary_dilation

    # Dilate by 1 pixel to get neighborhood
    dilated = ndimage.binary_dilation(skel, connectivity)

    # The difference gives us the boundary
    boundary = dilated & ~skel

    # For each foreground pixel, count how many boundary pixels it touches
    # This is equivalent to counting neighbors

    # Use distance transform to find pixels with minimal connectivity
    # Actually, let's use a simpler approach with convolution

    # Create a kernel that counts neighbors
    kernel = connectivity.astype(np.float32)
    kernel[tuple(np.array(kernel.shape) // 2)] = 0  # Don't count center

    # Count neighbors
    neighbor_count = ndimage.convolve(skel.astype(np.float32), kernel, mode='constant', cval=0)

    # Endpoints have exactly 1 neighbor
    endpoints = (skel > 0) & (neighbor_count == 1)

    return endpoints

# Current endpoints function - uses the ND version
def endpoints(skel):
    """
    Detect endpoints in a skeleton. Now uses the N-dimensional implementation.
    """
    return endpoints_nd(skel)

# Keep the original function for comparison (renamed)
def endpoints_original_2d(skel):  # pragma: no cover
    pad = 1 # appears to require padding to work properly....
    skel = np.pad(skel,pad)
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])
    
    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])
    
    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])
    
    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])
    
    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])
    
    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])
    
    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])
    
    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])
    
    import mahotas as mh
    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    unpad =  tuple([slice(pad,-pad)]*ep.ndim)
    ep = ep[unpad]
    return ep
    

# import sys
# sys.modules[__name__] = format_labels