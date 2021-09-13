import torch
from torch import nn

from ..blocks.blocks import _DecoderBlock
from ..dense.base import _DenseBase
from torch.nn import functional as nnf, Parameter
BatchNorm2d = nn.BatchNorm2d


class PyramidModule(nn.Module):

    def __init__(self, in_features=1000, mid_features=1000, out_features=1000):
        super().__init__()
        self.pyramid_convs = nn.ModuleList([
            nn.Sequential(nn.ReLU(), BatchNorm2d(in_features), nn.Conv2d(in_features, mid_features, (1, 1))),
            nn.Sequential(nn.ReLU(), BatchNorm2d(in_features), nn.Conv2d(in_features, mid_features, (1, 1))),
            nn.Sequential(nn.ReLU(), BatchNorm2d(in_features), nn.Conv2d(in_features, mid_features, (1, 1))),
            nn.Sequential(nn.ReLU(), BatchNorm2d(in_features), nn.Conv2d(in_features, mid_features, (1, 1)))
        ])
        self.conv_end1 = nn.Conv2d(in_features + 4 * mid_features, out_features, (1, 1), bias=False)

    def forward(self, x):
        intermediate_size = x.size(2), x.size(3)
        pyramid = [
            x,
            nnf.upsample(self.pyramid_convs[0](nnf.adaptive_avg_pool2d(x, (1, 1))), intermediate_size, mode='bilinear'),
            nnf.upsample(self.pyramid_convs[1](nnf.adaptive_avg_pool2d(x, (2, 2))), intermediate_size, mode='bilinear'),
            nnf.upsample(self.pyramid_convs[2](nnf.adaptive_avg_pool2d(x, (3, 3))), intermediate_size, mode='bilinear'),
            nnf.upsample(self.pyramid_convs[3](nnf.adaptive_avg_pool2d(x, (6, 6))), intermediate_size, mode='bilinear'),
        ]
        x = torch.cat(pyramid, dim=1)
        x = self.conv_end1(x)
        return x


class PSPNet(_DenseBase):

    def __init__(self, out_channels=10, binary=False, decoder_shape='m', with_mask=False, out_channel_weights=None,
                 dropout=None, base='drn105', use_act=False,
                 class_mean=False, pretrained=False, transfer_mode=False, with_skip=False):
        super().__init__(out_channels=out_channels, channel_range=(1, out_channels),
                         out_channel_weights=out_channel_weights, binary=binary, with_mask=with_mask,
                         class_mean=class_mean, transfer_mode=transfer_mode)

        self.transfer_exclude_parameters = ['conv_end2.2.weight', 'conv_end2.2.bias', 'thresholds']
        self.pretrained = pretrained
        self.dropout = dropout
        self.with_skip = with_skip
        self.decoder_shape = decoder_shape
        self.base = base
        self.use_act = use_act

        from drn.drn import drn_d_38, drn_d_54, drn_d_105

        if base == 'drn105':
            self.base_model = drn_d_105(pretrained=pretrained, out_map=True, out_middle=True)
        elif base == 'drn36':
            self.base_model = drn_d_38(pretrained=pretrained, out_map=True, out_middle=True)
        elif base == 'drn54':
            self.base_model = drn_d_54(pretrained=pretrained, out_map=True, out_middle=True)
        elif base == 'pnas':
            self.base_model = PNASFeatures(pretrained=pretrained, mid_level_features=True, avgpool=False)
        else:
            raise ValueError('Invalid base model: {}'.format(base))
        # self.final = nn.Sequential(
        #     nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(512, momentum=.95),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Conv2d(512, num_classes, kernel_size=1)
        # )

        if decoder_shape == 'm':
            dec_feats = 512
        elif decoder_shape == 's':
            dec_feats = 256
        elif decoder_shape == 'xs':
            dec_feats = 128
        else:
            raise ValueError('Invalid decoder shape {}'.format(decoder_shape))

        pyr_feats = (1000, 512) if not self.use_act else (512, 256)
        self.psp = PyramidModule(pyr_feats[0], pyr_feats[1], dec_feats)
        # self.bn_end = nn.BatchNorm2d(1000)

        if with_skip:
            self.preskip_conv = nn.Conv2d(dec_feats, dec_feats, kernel_size=1)
            self.skip_conn = _DecoderBlock(256 + dec_feats, dec_feats)

        seq = [
            nn.BatchNorm2d(dec_feats),  # momentum=.95?
            nn.ReLU(),
        ]

        if dropout is not None:
            seq += [nn.Dropout2d(p=dropout)]

        seq += [nn.Conv2d(dec_feats, out_channels, kernel_size=1)]
        self.conv_end2 = nn.Sequential(*seq)

        # self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

        self.features_only = False

        self.input_range = Parameter(torch.FloatTensor([1]).view(1, 1, 1, 1), requires_grad=False)
        self.input_mean = Parameter(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False)
        self.input_std = Parameter(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False)

    def name(self):
        return ''.join([self.__class__.__name__,
                        '-pretrain' if self.pretrained else '',
                        '-clsmean' if self.class_mean else '',
                        '-mask' if self.with_mask else '',
                        '-drop' + str(self.dropout) if self.dropout else '',
                        ('-dec_' + self.decoder_shape) if self.decoder_shape in {'s', 'xs'} else '',
                        '-skip' if self.with_skip else '',
                        '-use-act' if self.use_act else '',
                        ('-' + self.base) if self.base != 'drn105' else ''])

    def forward(self, x):
        x = self.normalize(x)

        original_size = x.size(2), x.size(3)
        x, activations = self.base_model(x)

        if self.use_act:
            x = activations[-1]

        # intermediate_size = x.size(2), x.size(3)

        # pyramid = [
        #     x,
        #     nnf.upsample(self.pyramid_convs[0](nnf.adaptive_avg_pool2d(x, (1, 1))), intermediate_size, mode='bilinear'),
        #     nnf.upsample(self.pyramid_convs[1](nnf.adaptive_avg_pool2d(x, (2, 2))), intermediate_size, mode='bilinear'),
        #     nnf.upsample(self.pyramid_convs[2](nnf.adaptive_avg_pool2d(x, (3, 3))), intermediate_size, mode='bilinear'),
        #     nnf.upsample(self.pyramid_convs[3](nnf.adaptive_avg_pool2d(x, (6, 6))), intermediate_size, mode='bilinear'),
        # ]
        #
        # x = torch.cat(pyramid, dim=1)
        # x = self.conv_end1(x)

        x = self.psp(x)
        # x = self.bn_end(x)

        if self.with_skip:
            skip_act = activations[-6]
            # print(111, skip_act.size())
            x = self.preskip_conv(x)
            x = self.skip_conn(x, skip_act)

        x = nnf.relu(x)
        x = self.conv_end2(x)

        # not sure if this is according to the original implementation
        x = nnf.upsample(x, original_size, mode='bilinear')

        if self.binary:
            return nnf.sigmoid(x),
        else:
            return nnf.log_softmax(x, dim=1),


