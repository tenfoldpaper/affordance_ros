import cv2
import numpy as np
from ..assertions import assert_object_type
from ..logging import log_important, log_warning, log_detail

CV_INTERPOLATIONS = {'nearest': cv2.INTER_NEAREST, 'bilinear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC}


def scale_to_bound(tensor_shape, bounds, interpret_as_max_bound=False):

    sf = bounds[0] / tensor_shape[0], bounds[1] / tensor_shape[1]

    lt = sf[0] < sf[1] if interpret_as_max_bound else sf[1] < sf[0]
    target_size = (bounds[0], int(tensor_shape[1] * sf[0])) if lt else (int(tensor_shape[0] * sf[1]), bounds[1])

    # make sure do not violate the bounds
    if interpret_as_max_bound:
        target_size = min(target_size[0], bounds[0]), min(target_size[1], bounds[1])
    else:
        target_size = max(target_size[0], bounds[0]), max(target_size[1], bounds[1])

    return target_size


def tensor_resize(tensor, target_size_or_scale_factor, interpret_as_min_bound=False, interpret_as_max_bound=False,
                  channel_dim=None, interpolation='bilinear', autoconvert=False, keep_channels_last=False):
    """
    Resizes a tensor (e.g. an image) along two dimensions while dimension `channel_dim` remains constant.
    if `target_size_or_scale_factor` is
     - a tuple (int, int) it specifies the target size. One dimension can be set to None if interpret_as_bound is not set.
     - a float it specifies the scale factor

    Best performance is obtained when `channel_index` is 2.
    """

    assert_object_type(tensor, np.ndarray)

    assert not (tensor.dtype.name == 'uint8' and tensor.ndim > 2 and tensor.shape[2] > 3), 'For uint8 more than three channels give wrong results. Maybe a bug on OpenCV?'

    if autoconvert:
        if tensor.dtype.name in {'int64', 'int32'}:
            log_important('Data is converted from ' + tensor.dtype.name + ' to int16, information might be lost.')
            tensor = tensor.astype('int16')
        elif tensor.dtype.name in {'float16'}:
            log_important('Data is converted from ' + tensor.dtype.name + ' to float32.')
            tensor = tensor.astype('float32')
        elif tensor.dtype.name in {'bool'}:
            log_important('Data is converted from ' + tensor.dtype.name + ' to uint8.')
            tensor = tensor.astype('uint8')
    else:
        if tensor.dtype.name not in {'uint8', 'int16', 'uint16', 'float32', 'float64'}:
            raise TypeError('unsupported datatype (by opencv): ' + tensor.dtype.name)

    if len(tensor.shape) == 2 and channel_dim is not None:
        log_warning('A 2d array is passed to tensor_resize, specifying channel_dim has no effect.')
        channel_dim = None

    # if channel index is not 2 then transpose such that it is.
    if channel_dim == 0:
        tensor = tensor.transpose([1, 2, 0])
    elif channel_dim == 1:
        tensor = tensor.transpose([0, 2, 1])

    if type(target_size_or_scale_factor) == tuple:
        # scale_factor = None

        if interpret_as_max_bound or interpret_as_min_bound:
            assert not interpret_as_min_bound or not interpret_as_max_bound

            target_size = scale_to_bound(tensor.shape, bounds=target_size_or_scale_factor,
                                         interpret_as_max_bound=interpret_as_max_bound)

            # sf = bounds[0] / tensor.shape[0], bounds[1] / tensor.shape[1]
            #
            # lt = sf[0] < sf[1] if interpret_as_max_bound else sf[1] < sf[0]
            # target_size = (bounds[0], int(tensor.shape[1] * sf[0])) if lt else (int(tensor.shape[0] * sf[1]), bounds[1])
            #
            # # make sure do not violate the bounds
            # if interpret_as_max_bound:
            #     target_size = min(target_size[0], bounds[0]), min(target_size[1], bounds[1])
            #
            # if interpret_as_min_bound:
            #     target_size = max(target_size[0], bounds[0]), max(target_size[1], bounds[1])

        else:
            if target_size_or_scale_factor[0] is None:
                target_size = int(tensor.shape[0] * target_size_or_scale_factor[1] / tensor.shape[1]), target_size_or_scale_factor[1]
            elif target_size_or_scale_factor[1] is None:
                target_size = target_size_or_scale_factor[0], int(tensor.shape[1] * target_size_or_scale_factor[0] / tensor.shape[0])
                # scale_factor = target_size_or_scale_factor[0] / tensor.shape[0]
            #
            # if scale_factor is not None:
            #     target_size = int(tensor.shape[0] * scale_factor), int(tensor.shape[1] * scale_factor)
            else:
                target_size = target_size_or_scale_factor
    elif type(target_size_or_scale_factor) == float:
        scale_factor = target_size_or_scale_factor
        target_size = int(tensor.shape[0] * scale_factor), int(tensor.shape[1] * scale_factor)
    else:
        raise ValueError('target_size must be either a int, float or a tuple of int.')

    log_detail('Resize tensor of shape', tensor.shape, 'and type', tensor.dtype, 'to target size', target_size)
    tensor = cv2.resize(tensor, (target_size[1], target_size[0]), interpolation=CV_INTERPOLATIONS[interpolation])

    if not keep_channels_last:
        if channel_dim == 0:
            tensor = tensor.transpose([2, 0, 1])
        elif channel_dim == 1:
            tensor = tensor.transpose([0, 2, 1])

    return tensor
