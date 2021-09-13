import cv2
import numpy as np
from ..assertions import assert_equal_shape


def crop_from_image(img, y_min, y_max, x_min, x_max, min_extent=80):
    # object crop = max_entent around center
    half_max_extent = max(int(min_extent/2), int(0.5 * max(y_max - y_min, x_max - x_min)))
    center_y, center_x = int(y_min + 0.5 * (y_max - y_min)), int(x_min + 0.5 * (x_max - x_min))

    # img[center_y - 2:center_y + 2, center_x - 2:center_x + 2, 0] = 255
    # img[center_y - 2:center_y + 2, center_x - 2:center_x + 2, 1] = 255
    # img[center_y - 2:center_y + 2, center_x - 2:center_x + 2, 2] = 0

    img_crop = img[
        max(0, center_y - half_max_extent): min(center_y + half_max_extent, img.shape[0]),
        max(0, center_x - half_max_extent): min(center_x + half_max_extent, img.shape[1])
    ]

    return img_crop


def random_crop_slices(origin_size, target_size):
    """
    Gets slices of a random crop
    """
    assert origin_size[0] >= target_size[0] and origin_size[1] >= target_size[1]

    offset_y = np.random.randint(0, origin_size[0] - target_size[0] + 1)  # range: 0 <= value < high
    offset_x = np.random.randint(0, origin_size[1] - target_size[1] + 1)

    return slice(offset_y, offset_y + target_size[0]), slice(offset_x, offset_x + target_size[1])


def random_crop(tensor, target_size, image_dimensions=(0, 1)):
    """
    Randomly samples a crop of size `target_size` from `tensor` along `image_dimensions`
    """
    assert len(image_dimensions) == 2 and type(image_dimensions[0]) == int and type(image_dimensions[1]) == int

    # slices = random_crop_slices(tensor, target_size, image_dimensions)
    origin_size = tensor.shape[image_dimensions[0]], tensor.shape[image_dimensions[1]]
    slices_y, slices_x = random_crop_slices(origin_size, target_size)

    slices = [slice(0, None) for _ in range(len(tensor.shape))]
    slices[image_dimensions[0]] = slices_y
    slices[image_dimensions[1]] = slices_x
    slices = tuple(slices)
    return tensor[slices]



def patch_sampling(img, segmentation, target_size, importance_map_index=None,
                   importance_map_size=(100, 100), random_scaling=True,
                   random_shift=True, label_input=False):
    """
    Takes an image `img` (HxWx3) and a segmentation tensor `segmentation` (HxWxC or HxW, if label_input) and randomly samples a
    smaller (`target size`) patch by considering a segmentation channel or index (`importance_map_index`) as
    a probability density.
    If random_scaling is True, before cropping, the image is scaled.
    If random shift is True, the patch is slightly shifted from the center.
    Returns:
        img_crop: cropped image
        seg_crop: cropped segmentation aligned with img_crop
        img: resized image if random_scaling, else None
    """

    msg = 'If label_input is True, segmentation must have two dimensions. Actual number of dimensions: '
    msg += str(segmentation.ndim)
    assert (segmentation.ndim == 3 and not label_input) or (segmentation.ndim == 2 and label_input), msg

    if random_scaling:
        # resize the input images randomly
        max_scale = min(img.shape[0] / target_size[0], img.shape[1] / target_size[1])

        scale_range = np.arange(min(1, max_scale), max_scale, 0.03)

        if len(scale_range) == 0:
            scale = 1
        else:
            scale_prob = 1.1 * max_scale - scale_range
            scale_prob = scale_prob / scale_prob.sum()

            # assert abs(0 - scale_prob.min()) < 0.0001, 'scale probabilities must be larger than zero'
            assert abs(1 - scale_prob.sum()) < 0.001, 'actual sum of scale_prob: ' + str(scale_prob.sum())
            scale = np.random.choice(scale_range, p=scale_prob)

        if scale != 1:
            scaled_size = (int(round(img.shape[1] / scale)), int(round(img.shape[0] / scale)))
            img = cv2.resize(img, scaled_size)
            if not label_input:
                segmentation = np.dstack(
                    [cv2.resize(segmentation[:, :, s].astype('float32'), scaled_size, interpolation=cv2.INTER_NEAREST)
                     for s in range(segmentation.shape[2])])
            else:
                segmentation = cv2.resize(segmentation, scaled_size, interpolation=cv2.INTER_NEAREST)

            # print('after resize', img.shape, aff.shape)

    assert img.shape[:2] == segmentation.shape[:2]

    if importance_map_index:
        if not label_input:
            importance_map = segmentation[:, :, importance_map_index].astype('float32')
        else:
            importance_map = (segmentation == importance_map_index).astype('float32')
    else:
        importance_map = None

    if importance_map is None:
        slice_indices = random_crop_slices(img.shape, target_size)
    else:
        slice_indices = sample_random_patch(img.shape, target_size, importance_map, random_shift=random_shift)

    img_crop = img[slice_indices]
    aff_crop = segmentation[slice_indices]

    assert_equal_shape(img_crop.shape[:2], target_size)
    assert_equal_shape(aff_crop.shape[:2], target_size)

    return img_crop, aff_crop, img if random_scaling else None


def sample_random_patch(img_shape, target_size, importance_map=None, random_shift=True, rescale=None):
    """
    Takes an image `img` (HxWx3)  and randomly samples a
    smaller (`target size`) patch according to the probability density `importance_map`.
    If random_scaling is True, before cropping, the image is scaled.
    If random shift is True, the patch is slightly shifted from the center.
    importance_map_size can speed up the sampling process by using a smaller map than the image.
    Returns:
        numpy slice object to be applied on the image dimensions
    """

    assert target_size[0] <= img_shape[0] and target_size[1] <= img_shape[1]

    if importance_map is not None:

        if rescale:
            assert type(rescale) == tuple and len(rescale) == 2

            importance_map_small = cv2.resize(importance_map, rescale)
            # importance_map_small = importance_map_small.flatten()
            importance_density = importance_map_small
        else:
            importance_density = importance_map

        # importance_density = np.zeros_like(importance_density)
        # importance_density[50:60, 70:80] = 1
        # idx = np.random.choice(importance_density.shape[0] * importance_density.shape[1],
        #                        p=(importance_density / importance_density.sum()).flatten())
        # iy, ix = np.unravel_index(idx, importance_density.shape)
        # idx = (iy, ix) if not rescale else (int(iy * sy), int(ix * sx))

        assert importance_density.max() == 1
        choices = np.where(importance_density == 1)
        idx = np.random.choice(len(choices[0]))

        idx = choices[0][idx], choices[1][idx]

        if rescale:
            sy, sx = img_shape[0] / rescale[0], img_shape[1] / rescale[1]
            idx = int(idx[0] * sy), int(idx[1] * sx)

    else:
        importance_density = None
        idx = np.random.choice(target_size[0] * target_size[1], p=importance_density)
        idx = np.unravel_index(idx, target_size)

    # shift_y = np.random.randint(-2 * int(sy), 2 * int(sy)) if int(sy) > 0 and random_shift else 0
    # shift_x = np.random.randint(-2 * int(sx), 2 * int(sx)) if int(sx) > 0 and random_shift else 0

    shift_y = np.random.randint(target_size[0] // 2 - 3) if random_shift else 0
    shift_x = np.random.randint(target_size[1] // 2 - 3) if random_shift else 0

    # shift_y, shift_x = 0, 0
    # top_left = max(0, idx[0] - target_size[0] // 2 + shift_y), max(0, idx[1] - target_size[1] // 2 + shift_x)
    top_left = idx[0] - target_size[0] // 2 + shift_y, idx[1] - target_size[1] // 2 + shift_x

    # make sure we do not cut
    top_left = min(top_left[0], img_shape[0] - target_size[0] - 1), min(top_left[1], img_shape[1] - target_size[1] - 1)
    top_left = max(top_left[0], 0), max(top_left[1], 0)

    slices = np.s_[top_left[0]:top_left[0] + target_size[0], top_left[1]: top_left[1] + target_size[1]]

    # print(idx, top_left, slices, img_shape, target_size)
    # return numpy slice object
    return slices