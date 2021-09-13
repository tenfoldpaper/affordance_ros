import time

import numpy as np
from sklearn.metrics import average_precision_score

from .base import BaseMetric


class MAPMetric(BaseMetric):

    def __init__(self, binary=False, **kwargs):
        super().__init__(('mAP_cw', 'mAP_cw_mean'), **kwargs)
        self.binary = binary
        self.gts = []
        self.preds = []
        self.masks = []

    def add(self, vars_x, vars_y):

        predictions = vars_x[0]
        ground_truth = vars_y[0]
        mask = vars_y[1] if len(vars_y) == 2 else None

        y_pred = predictions.detach().cpu().numpy()
        y_gt = ground_truth.detach().cpu().numpy().astype('int32')
        if mask is not None:
            mask = mask.detach().cpu().numpy()
            mask = mask.transpose([0, 2, 3, 1]).reshape((mask.shape[0] * mask.shape[2] * mask.shape[3], mask.shape[1]))
            mask = mask.astype('bool')

        self.gts += [y_gt.transpose([0, 2, 3, 1]).reshape((y_gt.shape[0] * y_gt.shape[2] * y_gt.shape[3], y_gt.shape[1])) > 0.5]
        self.preds += [y_pred.transpose([0, 2, 3, 1]).reshape((y_pred.shape[0] * y_pred.shape[2] * y_pred.shape[3], y_pred.shape[1]))]
        self.masks += [mask]

    def value(self):
        y_gt_samples = np.concatenate(self.gts, axis=0)
        y_pred_samples = np.concatenate(self.preds, axis=0)
        mask_samples = np.concatenate(self.masks, axis=0) if self.masks[0] is not None else None

        t_start = time.time()
        avg_prec = []

        for i in range(y_gt_samples.shape[1]):
            prec = average_precision_score(y_gt_samples[:, i], np.round(y_pred_samples[:, i], 3),
                                           sample_weight=mask_samples[:, i] if mask_samples is not None else None)
            avg_prec += [prec if not np.isnan(prec) else 0]

        avg_prec = np.array(avg_prec)
        print(avg_prec.shape)
        print('mAP computation took {}s'.format(time.time() - t_start))
        return avg_prec, float(avg_prec.mean())
