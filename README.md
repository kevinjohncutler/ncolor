# ncolor
Fast remapping of intance labels 1,2,3,...,M to a smaller set of repeating, disjoint labels, 1,2,...,N. The four-color-theorem guarantees that at most four colors are required for any 2D segmentation/map, but this algorithm will opt for 5 or 6 to give an acceptable result quickly. Also works for 3D labels (&lt;8 colors typically required) and perhaps higher dimensions as well.
