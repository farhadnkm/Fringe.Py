import numpy as np
from skimage import io


def import_image(path, modifiers=None, verbose=False, *args, **kwargs):
    """
    Imports an image specified by path.

    Parameters:
    -----------
        path:
            string
            Image path.

        modifiers:
            iterable[class(modifiers.Modifier)]
            list of 'Modifier' classes to apply operations on import.

    Returns:
    ----------
        The imported image with a type of ndarray.
    """
    img = io.imread(path)

    if verbose:
        print("Image imported from:", path)

    if modifiers is not None:
        if hasattr(modifiers, '__iter__'):
            for m in modifiers:
                img = m.process(img=img, *args, **kwargs)

        else:
            img = modifiers.process(img=img, *args, **kwargs)

    return img


def import_image_seq(paths, modifiers=None, verbose=False, *args, **kwargs):
    """
    Imports a sequence of images for a given list of paths.

    Parameters:
    -----------
        paths:
            iterable, list, tuple
            Paths of the images each given as a string.

        modifiers:
            iterable[class(modifiers.Modifier)]
            list of 'Modifier' classes to apply operations on images on import.

    Returns:
    ----------
        A list of imported images with the order of paths of type ndarray.
    """
    imgs = []
    for path in paths:
        img = import_image(path, modifiers, verbose, *args, **kwargs)
        imgs.append(img)
    return imgs


def export_image(image, path, dtype='uint8', verbose=True):
    """
    Exports a given image array to the specified path.

    Parameters:
    -----------
        image:
            ndarray
            the image array.
        path:
            string, path
            output path including file name and extension.
        dtype:
            string, dtype
            data type of the output image. Default: 'uint16'
            Note that some image formats do not support 16 bits color depth.

    Note:
    -----------
        Image arrays get flattened between 0 and 1 before export. So values higher than 1 and lower than 0
        will be replaced with 1 and 0.
    """
    assert dtype in ['uint8', 'uint16']
    _img = image.copy()
    _img[_img > 1] = 1
    _img[_img < 0] = 0

    if dtype == 'uint8':
        _img *= 2 ** 8 - 1
        _img = np.uint8(_img)
    elif dtype == 'uint16':
        _img *= 2 ** 16 - 1
        _img = np.uint16(_img)

    io.imsave(path, _img, check_contrast=False)
    if verbose:
        print("Image exported to:", path)
