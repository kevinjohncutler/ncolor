[![PyPI version](https://badge.fury.io/py/ncolor.svg)](https://badge.fury.io/py/ncolor)
<!-- [![Downloads](https://pepy.tech/badge/ncolor/day)](https://pepy.tech/project/ncolor) -->

# ncolor <img src="https://github.com/kevinjohncutler/ncolor/blob/main/logo.png?raw=true" width="400" title="bacteria" alt="bacteria" align="right" vspace = "0">

Fast remapping of instance labels 1,2,3,...,M to a smaller set of repeating, disjoint labels, 1,2,...,N. The four-color-theorem guarantees that at most four colors are required for any 2D segmentation/map, but this algorithm will opt for 5 or 6 to give an acceptable result if it fails to find a 4-color mapping quickly. Also works for 3D labels (&lt;8 colors typically required) and perhaps higher dimensions as well.
