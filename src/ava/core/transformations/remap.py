import numpy as np
from itertools import product

from ..logging import log_warning


def remap(orig_labels, occurring_labels, label_remapping, label_remapping_mask, no_exclusive_labels):


    n_properties = len(label_remapping[list(label_remapping.keys())[0]])
    seg_new = np.zeros((orig_labels[0].shape[0], orig_labels[0].shape[1], n_properties), 'float32')  # +1 because remapped labels start with 1
    mask_new = np.zeros_like(seg_new) if label_remapping_mask is not None else None

    # first use direct matches (without hierarchy)
    for object_set, label_map in [(occurring_labels[0], orig_labels[0]), (occurring_labels[1], orig_labels[1])]:
        for o_idx in object_set:
            if o_idx in label_remapping:
                matching_indices = np.where(label_map == o_idx)
                seg_new[matching_indices] = label_remapping[o_idx]
                if label_remapping_mask is not None:
                    mask_new[matching_indices] = label_remapping_mask[o_idx]

    # now hierarchical labels, i.e. object/part style entries
    for a, b in product(occurring_labels[0], occurring_labels[1]):

        found = None
        if (a, b) in label_remapping:
            found = (a, b)
        elif ('*', b) in label_remapping:
            found = ('*', b)

        if found is not None:

            matching_indices = np.where((orig_labels[0] == a) * (orig_labels[1] == b))
            # matching_indices = np.where((orig_labels[0] == a) * (orig_labels[1] == b))

            # print('hit2: ', a, b, objects[a], objects[b], matching_indices[0].shape)
            # print('found', self.objects[a], self.objects[b], cond)
            seg_new[matching_indices] = label_remapping[found]

            if label_remapping_mask is not None:
                mask_new[matching_indices] = label_remapping_mask[found]

    for a, b, c in product(occurring_labels[0], occurring_labels[1], occurring_labels[2]):
        found = None
        if (a, b, c) in label_remapping:
            found = (a, b, c)
        elif ('*', b, c) in label_remapping:
            found = ('*', b, c)

        if found is not None:
            matching_indices = np.where(((orig_labels[0] == a) * (orig_labels[1] == b) * (orig_labels[2] == c)))

            # print('hit3: ', a, b, c, objects[a], objects[b], objects[c], matching_indices[0].shape)
            # print('found', self.objects[a], self.objects[b], cond)
            seg_new[matching_indices] = label_remapping[found]

            if label_remapping_mask is not None:
                mask_new[matching_indices] = label_remapping_mask[found]

    mask = None
    if label_remapping_mask is not None:
        mask = mask_new

    if no_exclusive_labels:
        seg = seg_new
    else:
        if seg_new.sum(2).max() > 1:
            # raise ValueError('The remapping produced an overlapping segmentation')
            log_warning('The remapping produced an overlapping segmentation')

        seg = seg_new.argmax(2).astype('uint16')
        if mask is not None:
            mask = np.clip(mask.sum(2), 0, 1)

    if mask is not None:
        mask = mask.astype('bool')

    return seg, mask