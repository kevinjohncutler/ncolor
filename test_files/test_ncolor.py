import os
from pathlib import Path
import skimage.io
import ncolor

def test_ncolor():
    # rootdir = Path(ncolor.__file__).parent.parent.parent
    # masks_dir = os.path.join(rootdir,'test_files')
    masks_dir = os.path.dirname(os.path.abspath(__file__))
    print('masks dir', masks_dir)
    masks = skimage.io.imread(os.path.join(masks_dir,'example.png'))
    ncolor_masks = ncolor.label(masks)
    
if __name__ == '__main__':
    test_ncolor()