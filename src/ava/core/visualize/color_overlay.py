import numpy as np
from skimage.color import hsv2rgb, gray2rgb, rgb2gray


def color_overlay(img, maps, h_shift, indices, one_hot, n_colors=None, start_color=0, intensities=None):
    n_colors = len(indices) if n_colors is None else n_colors

    color_palette = hsv2rgb(np.dstack([
        np.arange(h_shift, 1, 1 / n_colors),
        0.7 * np.ones(n_colors),
        0.3 * np.ones(n_colors)
    ]))[0]

    out = gray2rgb(rgb2gray(img))  # desaturate
    out *= 0.5

    for i, idx in enumerate(indices):
        m = maps == idx if one_hot else maps[:, :, idx]
        # m = np.clip(m, 0, 1)
        col = color_palette[start_color + i]

        if intensities is not None:
            col = col * intensities[i]

        out = np.clip(out + col * m[:, :, None], 0, 1)

    out = (255 * out).astype('uint8')
    return out