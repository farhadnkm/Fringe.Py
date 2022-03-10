import numpy as np
from skimage import io

# Import / export / convert_type methods.


def import_image(path, preprocessors=None, *args, **kwargs):
    """
    :param path: Directory of the image
    :param preprocessors: list of classes with a process method.
    :return: a numpy array containing the image
    """
    img = io.imread(path)

    if kwargs.get('Verbose', True):
        print("Image imported from:", path)

    if preprocessors is not None:
        if hasattr(preprocessors, '__iter__'):
            for pp in preprocessors:
                img = pp.process(img=img, *args, **kwargs)

        else:
            img = preprocessors.process(img=img, *args, **kwargs)

    return img


def import_image_seq(paths, preprocessor=None, *args, **kwargs):
    """
    :param paths: Directories of the images
    :param preprocessor: An arbitrary class having a function 'solvers' which is performed on the images on import
    :return: A list of images as numpy arrays
    """
    imgs = []
    for path in paths:
        img = import_image(path, preprocessor, *args, **kwargs)
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
