[![PyPI version](https://badge.fury.io/py/ncolor.svg)](https://badge.fury.io/py/ncolor)
[![Downloads](https://static.pepy.tech/personalized-badge/ncolor?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/ncolor)

# ncolor <img src="https://github.com/kevinjohncutler/ncolor/blob/main/logo.png?raw=true" width="400" title="bacteria" alt="bacteria" align="right" vspace = "0">

Fast remapping of instance labels 1,2,3,...,M to a smaller set of repeating, disjoint labels, 1,2,...,N. The four-color-theorem guarantees that at most four colors are required for any 2D segmentation/map, but this algorithm will opt for 5 or 6 to give an acceptable result if it fails to find a 4-color mapping quickly. Also works for 3D labels (&lt;8 colors typically required) and perhaps higher dimensions as well.

### Usage
If you have an integer array called `masks`, you may transform it into an N-color representation as follows:

```python
import ncolor 
ncolor_masks = ncolor.label(masks)
```

If you need the number of unique labels returned:
```python
ncolor_masks, num_labels = ncolor.label(masks,return_n=True)

```
If you need to convert back to `0,...,N` object labels:
```python
labels = ncolor.format_labels(ncolor_masks,clean=True)

```

Note that `format_labels` with ```clean=True``` will also remove small labels (<9px) by default. This behavior can be changed with the `min_area` parameter. 

    
The integer array `ncolor_masks` can be visualized using any color map you prefer. The example in this README uses the viridis colormap. See `example.ipynb` for more details.

Thanks to Ryan Peters ([@ryanirl](https://github.com/ryanirl)) for suggesting the `expand_labels` function. This is applied by default to 2D images (optionally for 3D images with `expand=True`, but this can give bad results since objects in 3D have a lot more wiggle room to make contact when expanded). This preprocessing step eliminates cases where close (but not touching) or dispersed objects previously received the same label. I dug a layer back to use `ndimage.distance_transform_edt` for a speed boost. If undesired for 2D images, use `expand=False`. 
