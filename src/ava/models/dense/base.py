from functools import partial

import torch
import torch.nn.functional as nnf
import numpy as np
from torch.nn import Parameter
from ava.core.logging import *
from ava.core.metrics import IoUMetric, MAPMetric

imagenet_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
imagenet_std = np.array([0.229, 0.224, 0.225], dtype='float32')


def binary_cross_entropy_masked(input, target, channel_weights, mask, class_mean=False):

    eps = 10e-6
    loss = -0.5 * (target * torch.log(eps + input) + (1-target) * torch.log(eps + 1-input))

    if channel_weights is not None:
        mask *= channel_weights.view(1, -1, 1, 1)

    loss = loss * mask

    if class_mean:
        # mean over classes
        loss = loss.sum(dim=3).sum(dim=2).sum(dim=0) / (1 + mask.sum(dim=3).sum(dim=2).sum(dim=0))
        loss = loss.mean()
        loss = loss / mask.size(0)  # to normalize sum over batch dimension
    else:
        # loss over image-dimensions
        loss = loss.sum(dim=3).sum(dim=2).sum(dim=1) / (1 + mask.sum(dim=3).sum(dim=2).sum(dim=1))
        loss = loss.mean()

    return loss


class _ModelBase(torch.nn.Module):
    """
    Abstract base class for all network models.
    """

    def __init__(self, prediction_names, metrics=None, visualization_modes=None):
        super().__init__()

        self.prediction_names = prediction_names
        self._metrics = metrics if metrics is not None else tuple()
        if visualization_modes is None:
            self.visualization_modes = tuple(None for _ in prediction_names)

        self._defaults = dict()
        self.input_range, self.input_mean, self.input_std = None, None, None
        self.parameter_variables = []

    @staticmethod
    def is_baseline():
        return False

    def metrics(self):
        return self._metrics

    def metric_names(self):
        return [m for metric in self.metrics() for m in metric().names()]

    def forward(self, x):
        """
        Forward pass. See details in pytorch's documentation.
        """
        raise NotImplementedError

    # def prepare_sample(self, *args):
    #     """
    #     Receives a tuple of arrays and converts them to pytorch tensors. The base implementation should
    #     work in most cases but can be overridden if necessary.
    #     """
    #     out_args = []
    #     for a in args:
    #         if a.dtype in {'int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8'}:
    #             a = Variable(LongTensor(a))
    #         elif a.dtype in {'float64', 'float32', 'float16'}:
    #             a = Variable(FloatTensor(a))
    #
    #         if CONFIG['CUDA']:
    #             a = a.cuda()
    #
    #         out_args += [a]
    #
    #     return tuple(out_args)

    def loss(self, y_pred, y_gt):
        """
        Takes predictions and ground truth batches and computes a scalar loss.
        """
        raise NotImplementedError

    def accumulative_scores(self, y_pred, y_gt):
        """
        Takes predictions and ground truth batches and returns a tuple of integers, float or
        2d numpy arrays representing scores. The arrays must have size BATCH x K, where K can be an arbitrary integer
        (e.g. the number of classes for class-wise scores) or just one. The length of the returned tuple must
        match the number of arguments of score_accumulated.
        """
        raise NotImplementedError

    def score_accumulated(self, *acc_inputs):
        """
        Takes the intermediate objects computed by `accumulate_scores` and returns a tuple of integers, float or
        2d numpy arrays representing scores. The arrays must have size BATCH x K, where K can be an arbitrary integer
        (e.g. the number of classes for class-wise scores) or just one.
        """
        raise NotImplementedError

    def score(self, y_pred, y_gt):
        """
        Takes predictions and ground truth batches and returns a tuple of integers, float or
        2d numpy arrays representing scores. The arrays must have size BATCH x K, where K can be an arbitrary integer
        (e.g. the number of classes for class-wise scores) or just one.
        """
        raise NotImplementedError

    # def metrics(self):
    #     metrics = list(self.metric_names)
    #     metrics += list(self.acc_metric_names) if hasattr(self, 'acc_metric_names') else []
    #     return tuple(metrics)

    def name(self):
        return class_config_str(self.__class__.__name__, self.parameter_variables)

        # return self.__class__.__name__

    def on_epoch_end(self, epoch):
        """ Hook, that is called after a training epoch is complete. Can be used to set loss_weights. """
        pass

    def set_default_optimizer(self, optimizer, learning_rate, weight_decay=0):
        self._defaults['OPT'] = optimizer
        self._defaults['LR'] = learning_rate
        self._defaults['WEIGHT_DECAY'] = weight_decay

    def get_default(self, key):
        if key in self._defaults:
            return self._defaults[key]
        else:
            return None

    # def save_checkpoint(self, training_details, training_state):
    #     checkpoint = {'state_dict': self.state_dict()}
    #     checkpoint.update(training_details)
    #
    #     checkpoint.update(training_details)
    #     torch.save(checkpoint, training_details['checkpoint_name'])

    def set_normalization(self, in_range, mean, std, dims=4, color_dim=1):
        sizes = tuple(1 if i != color_dim else 3 for i in range(dims))
        self.input_range = Parameter(torch.FloatTensor([in_range]).view(*(1 for _ in range(dims))), requires_grad=False)
        self.input_mean = Parameter(torch.FloatTensor(mean).view(*sizes), requires_grad=False)
        self.input_std = Parameter(torch.FloatTensor(std).view(*sizes), requires_grad=False)

    def normalize(self, x):
        x = x.float()
        x = x * (self.input_range / 255.0)
        x = x - self.input_mean
        x = x / self.input_std
        # print(float(x.min()), float(x.max()))
        return x


class _DenseBase(_ModelBase):
    """
    This abstract class conducts the common initialization and  defines loss and metrics for dense prediction
    settings, e.g. semantic segmentation.

    `channel_range`:
    defines the channels to be used to compute the class-wise metrics (e.g. IoU). It can be used to
    exclude the background.
    """

    def __init__(self, in_channels=3, out_channels=10, out_channel_weights=None, binary=False, with_mask=False,
                 channel_range=None, class_mean=False, transfer_mode=False):

        metrics = [
            partial(IoUMetric, n_classes=out_channels, binary=binary, model_ref=self, threshold=0.1),
            partial(IoUMetric, n_classes=out_channels, binary=binary, model_ref=self, threshold=0.5),
        ]
        if binary:
            metrics += [partial(MAPMetric, binary=True, eval_intermediate=False, eval_validation=False)]

        super().__init__(('predicted_segmentation',), metrics)

        self.transfer_mode = transfer_mode
        self.class_mean = class_mean
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.binary = binary

        if channel_range is None:
            self.channel_range = range(self.out_channels)
        else:
            assert type(channel_range) == tuple and len(channel_range) == 2
            self.channel_range = range(channel_range[0], channel_range[1])

        self.thresholds = Parameter(0.6*torch.ones(out_channels), requires_grad=False)

        self.loss_weights = None

        if type(out_channel_weights) in {tuple, list}:
            out_channel_weights = torch.FloatTensor(out_channel_weights)
            out_channel_weights /= out_channel_weights.sum()
            self.out_channel_weights = Parameter(out_channel_weights, requires_grad=False)
            log_important('Using channel weights', out_channel_weights)
        elif type(out_channel_weights) == int:
            w = out_channel_weights
            self.out_channel_weights = Parameter(torch.FloatTensor([w for _ in range(out_channels)]), requires_grad=False)
            log_important('Using channel weights', out_channel_weights)
        else:
            self.out_channel_weights = None

        # self.metric_names = 'pixel accuracy', 'accuracy classwise', 'mean accuracy classwise',

        if self.binary:
            # self.acc_metric_names = 'iou_classwise', 'iou_classwise_mean', 'mAP_classwise', 'mAP_classwise_mean'
            self.visualization_modes = ('ImageSlice', {'maps_dim': 0}),
        else:
            # self.acc_metric_names = 'iou_classwise', 'iou_classwise_mean'
            self.visualization_modes = ('ImageSlice', {'maps_dim': 0, 'normalize': True}),



        self.with_mask = with_mask
        self.input_range = Parameter(torch.FloatTensor([1]).view(1, 1, 1, 1), requires_grad=False)
        self.input_mean = Parameter(torch.FloatTensor(imagenet_mean).view(1, 3, 1, 1), requires_grad=False)
        self.input_std = Parameter(torch.FloatTensor(imagenet_std).view(1, 3, 1, 1), requires_grad=False)

    def set_thresholds(self, thresholds):
        """ Special method that allows to change the threshold of an initialized model. """

        thresholds = torch.FloatTensor(thresholds)

        if next(self.parameters()).is_cuda:
            thresholds = thresholds.cuda()

        self.thresholds = Parameter(thresholds, requires_grad=False)

    def set_mask(self, use_mask):
        if use_mask:
            self.with_mask = True
        else:
            self.with_mask = False

    def loss(self, y_pred, y_gt):
        """
        Compute the loss given y_pred and y_gt. Here we expect y_pred to be a tuple. If there is more than on element
        we assume the network has multiple predictions
        """
        #y_pred, = y_pred  # predictions need to be tuples, even with one element
        # y_gt, mask = y_gt

        mask = y_gt[-1]
        y_gt = y_gt[:-1]

        assert len(y_pred) == len(y_gt), ('Number of predictions must match the number of provided ground truth '
                                          'elements. If you use multiscale, are you sure, the number of predictions'
                                          'matches the number of ground truth elements?')

        if self.with_mask:

            assert len(y_gt) == 1, 'If mask is used, multiple outputs are not possible.'
            y_gt = y_gt[0]

            # mask = y_masked_gt[:, 0].contiguous().view(y_masked_gt.size(0), 1, y_masked_gt.size(2), y_masked_gt.size(3))  # batch_size x H x W

            # repeat mask along channels if the mask has only image size
            if mask.size() != y_gt.size():
                mask = mask.view(y_gt.size(0), 1, y_gt.size(2), y_gt.size(3))
                mask = mask.repeat(1, y_pred.size(1), 1, 1)

        # log_detail('loss input sizes pred: {}, gt: {}, mask: {}'.format(
        #     ' '.join(str(p[0].size()) for p in y_pred), ' '.join(str(p[0].size()) for p in y_gt),
        #     str(mask[0].size()) if mask is not None else None))

        # import ipdb
        # ipdb.set_trace()

        if self.binary:
            if self.with_mask:
                loss = 0

                for i in range(len(y_pred)):
                    weight = self.loss_weights[i] if self.loss_weights is not None else 1
                    loss += weight * binary_cross_entropy_masked(y_pred[i], y_gt, self.out_channel_weights, mask,
                                                                 self.class_mean)
            else:
                weights = self.out_channel_weights.view(1, -1, 1, 1) if self.out_channel_weights is not None else None
                loss = 0
                for i in range(len(y_pred)):
                    weight = self.loss_weights[i] if self.loss_weights is not None else 1
                    loss += weight * nnf.binary_cross_entropy(y_pred[i], y_gt[0], weight=weights)
        else:
            if self.with_mask:
                raise NotImplementedError('If you want to mask pixels in non binary mode use ignore_index')
            else:
                loss = 0
                for i in range(len(y_pred)):
                    weight = self.loss_weights[i] if self.loss_weights is not None else 1
                    loss += weight * nnf.nll_loss(y_pred[i], y_gt[i], weight=self.out_channel_weights, ignore_index=-1)

        return loss

    def load_state_dict(self, state_dict, strict=True):
        if self.transfer_mode:

            for k in self.transfer_exclude_parameters:
                del state_dict[k]

            return super().load_state_dict(state_dict, strict=False)
        else:
            return super().load_state_dict(state_dict, strict=strict)