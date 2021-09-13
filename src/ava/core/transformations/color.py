import cv2
import numpy as np


def get_gamma_offset_lut(gamma, offset):
    if offset is None:
        if type(gamma) in {float, int}:
            offset = 0
        else:
            offset = np.zeros_like(gamma)

    if type(gamma) in {float, int}:
        inv_gamma = 1 / gamma
        lut = np.arange(0, 1, 1/256)
        lut = (lut ** inv_gamma) * 256 + offset
        lut = np.clip(lut, 0, 255)

    elif len(gamma) == 3:
        inv_gamma = 1 / np.array(gamma)

        lut = np.arange(0, 1, 1 / 256)
        lut = (lut[:, np.newaxis] ** inv_gamma) * 255 + offset
        lut = np.clip(lut, 0, 255)
        lut = lut.reshape((256, 1, 3))
    else:
        raise ValueError('gamma must be either an array or list of three gamma values or a single float value')

    return np.array(lut).astype('uint8')


def apply_gamma_offset(img, gamma=None, offset=None, lut=None, channel_dim=2, keep_channels_last=False):
    assert img.dtype == 'uint8'

    assert (gamma is not None) or (lut is not None), 'either gamma/offset or lut must be provided'

    if gamma is not None:
        lut = get_gamma_offset_lut(gamma, offset)

    # if channel index is not 2 then transpose such that it is.
    if channel_dim == 0:
        img = img.transpose([1, 2, 0])
    elif channel_dim == 1:
        img = img.transpose([0, 2, 1])

    img = cv2.LUT(img, lut)

    if not keep_channels_last:
        if channel_dim == 0:
            img = img.transpose([2, 0, 1])
        elif channel_dim == 1:
            img = img.transpose([0, 2, 1])

    return img