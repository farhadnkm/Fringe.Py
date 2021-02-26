import numpy as np
from skimage import io


# Import / export / convert_type methods.


def import_image(path, preprocessor=None):
    """
    :param path: Directory of the image
    :param preprocessor: An arbitrary class having a function 'process' which is performed on the images on import
    :return: a numpy array containing the image
    """
    img = io.imread(path)

    print("Image imported from:", path)
    if preprocessor is not None:
        if hasattr(preprocessor, '__iter__'):
            for pp in preprocessor:
                img = pp.process(img=img)
        else:
            img = preprocessor.process(img=img)

    return img


def import_image_seq(paths, preprocessor=None):
    """
    :param paths: Directories of the images
    :param preprocessor: An arbitrary class having a function 'process' which is performed on the images on import
    :return: A list of images as numpy arrays
    """
    imgs = []
    for path in paths:
        img = import_image(path, preprocessor)
        imgs.append(img)
    return imgs


def export_image(img, path, dtype='uint16'):
    assert dtype in ['uint8', 'uint16']
    _img = img.copy()
    _img[_img > 1] = 1
    _img[_img < 0] = 0

    if dtype == 'uint8':
        _img *= 2 ** 8 - 1
        _img = np.uint8(_img)
    elif dtype == 'uint16':
        _img *= 2 ** 16 - 1
        _img = np.uint16(_img)

    io.imsave(path, _img)
    print("Image exported to:", path)
