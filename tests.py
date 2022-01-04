import pytest
import os, sys, shutil
from pathlib import Path
import skimage.io
import ncolor.ncolor as ncolor


def test_ncolor():
    masks_dir = Path.home().joinpath('ncolor')
    print('masks_dir')
    masks = skimage.io.imread(masks_dir)
    ncolor_masks = ncolor.label(masks)