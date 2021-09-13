import numpy as np

from .base import BaseMetric


class IoUMetric(BaseMetric):
    """
    Computes intersection over union.
    `binary`
    `threshold`: value at which activations are considered to be on (only if `binary=true`)
    `model_ref`: reference to model (if `threshold=None`)
    `binary`
    """

    def __init__(self, n_classes, binary=False, model_ref=None, threshold=None):
        thres = '@{:.2f}'.format(threshold) if threshold is not None else ''
        super().__init__(('IoU_cw' + thres, 'ioU_cw_mean' + thres))
        self.threshold = threshold
        self.n_classes = n_classes
        self.binary = binary
        self.model_ref = model_ref
        self.intersections = []
        self.unions = []

    def add(self, pred, gt):
        predictions = pred[0]
        ground_truth = gt[0]
        mask = gt[1] if len(gt) == 2 else None

        predictions = predictions.detach().cpu().numpy()
        ground_truth = ground_truth.detach().cpu().numpy().astype('int32')
        if mask is not None:
            mask = mask.detach().cpu().numpy()

        if self.binary:
            if self.threshold is None:
                a = predictions > self.model_ref.thresholds.detach().cpu().numpy().reshape((1, -1, 1, 1))
            else:
                a = predictions > self.threshold

            b = ground_truth > 0.5
            intersection = (a * b).astype('float32')
            union = np.clip(a + b, 0, 1).astype('float32')

            if mask is not None:
                intersection = intersection * mask
                union = union * mask

            intersection = intersection.sum(3).sum(2).sum(0)
            union = union.sum(3).sum(2).sum(0)

            self.intersections += [intersection]
            self.unions += [union]
        else:
            assert mask is None or mask.sum() == (mask.size(0) * mask.size(1) * mask.size(2) * mask.size(3))
            intersection, union = intersection_union(predictions.argmax(1), ground_truth, self.n_classes)
            self.intersections += [intersection]
            self.unions += [union]

    def value(self):
        intersections = np.array(self.intersections).sum(0)
        unions = np.array(self.unions).sum(0)
        classwise_iou = np.divide(intersections, unions, where=intersections > 0, out=np.zeros(intersections.shape))
        return classwise_iou, np.mean(classwise_iou)


def intersection_union(prediction, ground_truth, n_classes):
    """
    Computes the class-wise intersection and union and returns them as separate arrays.
    """

    assert type(prediction) == np.ndarray
    assert type(ground_truth) == np.ndarray

    err_msg = 'int data type required (no unsigned because of ignore index). Actual type: {}'
    assert prediction.dtype.name in {'int8', 'int16', 'int32', 'int64'}, err_msg.format(prediction.dtype)
    assert ground_truth.dtype.name in {'int8', 'int16', 'int32', 'int64'}, err_msg.format(prediction.dtype)

    prediction += 1
    ground_truth += 1

    # this will ignore all values that are zero (i.e. were -1)
    prediction = (ground_truth > 0) * prediction
    intersection = np.bincount((prediction * (prediction == ground_truth)).flatten(), minlength=n_classes+1)
    pred = np.bincount(prediction.flatten(), minlength=n_classes+1)
    gt = np.bincount(ground_truth.flatten(), minlength=n_classes+1)

    union = pred + gt - intersection

    return intersection[1:], union[1:]


def intersection_over_union(prediction, ground_truth, n_classes):
    intersection, union = intersection_union(prediction, ground_truth, n_classes)
    return intersection / union