import os
from pathlib import Path
import skimage.io
import ncolor

def test_ncolor():
    masks_dir = Path(os.path.dirname(ncolor.__file__)).parent.parent.absolute()
    print('masks dir', masks_dir)
    masks = skimage.io.imread(os.path.join(masks_dir,'example.png'))
    ncolor_masks = ncolor.label(masks)
    
if __name__ == '__main__':
    test_ncolor()