import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from ...core.logging import log_important, log_warning
from ..blocks.blocks import _DecoderBlock, _PredictionOutput, _Refine
from .base import _DenseBase
from .pspnet import PyramidModule


class _ResNetDense(_DenseBase):
    """
    decoder shape defines different factors for the number of maps in the decoder layers.
    """

    def __init__(self, base_network, decoder_shape='m', out_channels=10, out_channel_weights=None, with_mask=False,
                 binary=False, init_state_dict=None, pretrained=False, multipredict=False, thresholds=None,
                 pyramid=False, class_mean=False, dropout=None, small_resnet=False, transfer_mode=False):
        super().__init__(3, out_channels, out_channel_weights, binary, with_mask, class_mean=class_mean,
                         transfer_mode=transfer_mode)

        self.transfer_exclude_parameters = ['post_conv.weight', 'post_conv.bias', 'thresholds']
        self.pyramid = pyramid
        self.dropout = dropout
        self.multipredict = multipredict
        self.resnet = base_network(pretrained=pretrained)
        self.pretrained = pretrained

        if not self.pretrained:
            log_warning('You are not using the ImageNet-pretrained weights!')

        if init_state_dict is not None:
            log_important('using init state dict', init_state_dict)

            state_ext = torch.load(init_state_dict)['state_dict']  # external state dict
            state_int = self.state_dict()  # internal state dict

            external_states = {key: v for key, v in state_ext.items() if key in state_int and v.size() == state_int[key].size()}
            log_important('no weights found for', set(state_int.keys()).difference(external_states.keys()))

            # if external key is available use it, otherwise use internal
            common_state_dict = {k: external_states[k] if k in external_states else state_int[k] for k, v in state_ext.items()}
            self.load_state_dict(common_state_dict)

        self.decoder_shape = decoder_shape
        K = 16
        if decoder_shape == 's': dec_factors = (16, 8, 4, 1, 1)
        elif decoder_shape == 'm': dec_factors = (16, 8, 4, 2, 2)
        elif decoder_shape == 'm+': dec_factors = (24, 16, 4, 2, 2)
        elif decoder_shape == 'l': dec_factors = (16, 8, 4, 3, 3)
        elif decoder_shape == 'l+': dec_factors = (24, 16, 4, 3, 3)
        elif decoder_shape == 'xl': dec_factors = (16, 8, 4, 4, 4)
        elif decoder_shape == 'xl+': dec_factors = (24, 16, 4, 4, 4)
        elif decoder_shape == 'xxl': dec_factors = (16, 8, 6, 6, 6)
        elif decoder_shape == 'xxl+': dec_factors = (24, 16, 6, 6, 6)
        elif decoder_shape == 'bottleneck': dec_factors = (16, 8, 1, 1, 1)
        else:
            raise ValueError('Invalid decoder_shape')

        # decoder_factors = (16, 8, 2, 1, 1)
        # decoder_factors = (16, 8, 2, 1, 1)




        if not small_resnet:
            enc_factors = [192 * K, 32 * K, 16 * K, 4 * K, 3]
        else:
            enc_factors = [48 * K, 8 * K, 4 * K, 4 * K, 3]

        if pyramid:
            # old: 2048
            feats = {'d3': 512, 'd4': 1024, 'd5': 2048, 'u2': 256, 'u3': 128, 'u4': 64}[pyramid]
            self.pyramid = PyramidModule(in_features=feats, mid_features=512, out_features=feats)
            self.pyramid_layout = pyramid
        else:
            self.pyramid = None
            self.pyramid_layout = None

        self.decoder2 = _DecoderBlock(enc_factors[0], dec_factors[0]*K, dropout=dropout)
        self.decoder3 = _DecoderBlock(dec_factors[0]*K + enc_factors[1], dec_factors[1]*K, dropout=dropout)
        self.decoder4 = _DecoderBlock(dec_factors[1]*K + enc_factors[2], dec_factors[2]*K, dropout=dropout)
        self.decoder5 = _DecoderBlock(dec_factors[2]*K + enc_factors[3], dec_factors[3]*K, dropout=dropout)
        self.decoder6 = _DecoderBlock(dec_factors[3]*K + enc_factors[4], dec_factors[4]*K, dropout=dropout)
        self.decoders = [self.decoder2, self.decoder3, self.decoder4, self.decoder5, self.decoder6]
        self.post_conv = nn.Conv2d(dec_factors[4]*K, self.out_channels, (1, 1))



        ########   Refination test ###########

#        if not small_resnet:
#            k_r=3
#            enc_factors_refine_extra = [512 * k_r, 256 * k_r, 128 * k_r, 64 * k_r, 0 , 0 ]
#        else:
#            k_r=1
#            #enc_factors_refine = [512 * k_r, 256 * k_r, 128 * k_r, 64 * k_r, 64, 3]
#            enc_factors_refine_extra = [ 0, 0, 0, 0, 0, 0]
#
#        
#        enc_factors_refine = [512 , 256 , 128 , 64 , 64, 3]#
#
#        self.post_conv_refination = nn.Conv2d(16,12,(1,1))
#        self.dec2 = _Refine(enc_factors_refine[1] + enc_factors_refine_extra[1],enc_factors_refine[0] + enc_factors_refine_extra[0],1)
#        self.dec3 = _Refine(enc_factors_refine[2] + enc_factors_refine_extra[2], enc_factors_refine[1], 2)
#        self.dec4 = _Refine(enc_factors_refine[3] + enc_factors_refine_extra[3], enc_factors_refine[2], 3)
#        self.dec5 = _Refine(enc_factors_refine[4] + enc_factors_refine_extra[4], enc_factors_refine[3], 4)
#        self.dec6 = _Refine(enc_factors_refine[5] + enc_factors_refine_extra[5], int(enc_factors_refine[4] * 0.5) ,5)

        #self.post_conv_refination = nn.Conv2d(16,12,(1,1))
        #self.dec2 = _Refine(256,512,1)
        #self.dec3 = _Refine(128,256,2)
        #self.dec4 = _Refine(64,128,3)
        #self.dec5 = _Refine(64,64,4)
        #self.dec6 = _Refine(3,32,5)
        ########   Refination test ###########

        if self.multipredict:
            self.pred_out = nn.ModuleList([
                _PredictionOutput(dec_factors[i] * K, self.out_channels, with_softmax=True, dropout=dropout) for i in range(1, 4)
            ])

        if thresholds:
            self.set_thresholds(thresholds)

    def name(self):
        return '{}-{}{}{}{}{}{}{}{}'.format(self.__class__.__name__, self.decoder_shape,
                                            '-pyr' + self.pyramid_layout if self.pyramid else '',
                                            '-pretrain' if self.pretrained else '',
                                            '-mulpred' if self.multipredict else '',
                                            '-clsmean' if self.class_mean else '',
                                            '-mask' if self.with_mask else '',
                                            '-drop' + str(self.dropout) if self.dropout is not None else '',
                                            '-cw' if self.out_channel_weights is not None else '')

    def n_parameters(self):
        return sum([np.prod(p.size()) for p in self.parameters()])

    # def get_features(self, x, decode_steps=None):
    #     x0 = self.normalize(x)
    #
    #     x1 = self.resnet.conv1(x0)
    #     x1 = self.resnet.bn1(x1)
    #     x1 = self.resnet.relu(x1)
    #     x1 = self.resnet.maxpool(x1)
    #
    #     x2 = self.resnet.layer1(x1)
    #     x3 = self.resnet.layer2(x2)
    #     x4 = self.resnet.layer3(x3)
    #     x5 = self.resnet.layer4(x4)
    #
    #     # if self.pyramid is not None:
    #     #     x5 = self.pyramid(x5)
    #
    #     x_up = x5
    #
    #     for decoder, x_skip in list(zip(self.decoders, [x4, x3, x2, x1, x0]))[:decode_steps]:
    #         x_up = decoder(x_up, x_skip)
    #
    #     # x_up = self.decoder6(x_up, x0)
    #
    #     return x_up,

    def forward(self, x):


        x0 = self.normalize(x)
        #print('size x0: ',x0.shape)
        x1 = self.resnet.conv1(x0)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)
        x1 = self.resnet.maxpool(x1)
        #print('size x1: ',x1.shape)

        x2 = self.resnet.layer1(x1)
        #print('size x2: ',x2.shape)
        x3 = self.resnet.layer2(x2)

        if self.pyramid_layout == 'd3': x3 = self.pyramid(x3)
        x4 = self.resnet.layer3(x3)
        #print('size x3: ',x3.shape)
        if self.pyramid_layout == 'd4': x4 = self.pyramid(x4)
        x5 = self.resnet.layer4(x4)
        #print('size x4: ',x4.shape)
        #print('size x5: ',x5.shape)
        if self.pyramid_layout == 'd5': x5 = self.pyramid(x5)

        # if self.pyramid is not None:
        #     x5 = self.pyramid(x5)

        

        ########   Refination test ###########
#        x_up2 = self.dec2(x5,x4)
#        #print("x_up2: ",x_up2.shape)
#        x_up3 = self.dec3(x_up2,x3)
#        #print("x_up3: ",x_up3.shape)
#        x_up4 = self.dec4(x_up3,x2)
#        #print("x_up4: ",x_up4.shape)
#        x_up5 = self.dec5(x_up4,x1)
#        #print("x_up5: ",x_up5.shape)
#        x_up6 = self.dec6(x_up5,x0)   
#        #print("x_up6: ",x_up6.shape)
#        
#        x_up = self.post_conv_refination(x_up6)
#        #print('x_up : ',x_up.shape)

        ########   Refination test ###########


        ########   Original  ###########

        x_up2 = self.decoder2(x5, x4)
        #print('size x_up2: ',x_up2.shape)
        if self.pyramid_layout == 'u2': x_up2 = self.pyramid(x_up2)
        x_up3 = self.decoder3(x_up2, x3)
        #print('size x_up3: ',x_up3.shape)
        if self.pyramid_layout == 'u3': x_up3 = self.pyramid(x_up3)
        x_up4 = self.decoder4(x_up3, x2)
       # print('size x_up4: ',x_up4.shape)
        if self.pyramid_layout == 'u4': x_up4 = self.pyramid(x_up4)
        x_up5 = self.decoder5(x_up4, x1)
        #print('size x_up5: ',x_up5.shape)
        x_up6 = self.decoder6(x_up5, x0)
        #print('size x_up6: ',x_up6.shape)

        x_up = self.post_conv(x_up6)
        #print('x_up--> ' , x_up.shape)

        ########   Original  ###########

        # x_up1 = self.decoder1(x5)
        # x_up2 = self.decoder2(torch.cat((x_up1, x4), dim=1))
        # x_up3 = self.decoder3(torch.cat((x_up2, x3), dim=1))
        # x_up4 = self.decoder4(torch.cat((x_up3, x2), dim=1))
        # x_up5 = self.decoder5(torch.cat((x_up4, x), dim=1))

        # sizes
        # x torch.Size([1, 3, 550, 824])
        # x0 torch.Size([1, 3, 550, 824])
        # x1 torch.Size([1, 64, 138, 206])
        # x2 torch.Size([1, 256, 138, 206])
        # x3 torch.Size([1, 512, 69, 103])
        # x4 torch.Size([1, 1024, 35, 52])
        # x5 torch.Size([1, 2048, 18, 26])

        # x_up2 torch.Size([1, 256, 35, 52])
        # x_up3 torch.Size([1, 128, 69, 103])
        # x_up4 torch.Size([1, 64, 138, 206])
        # x_up5 torch.Size([1, 32, 138, 206])
        # x_up6 torch.Size([1, 32, 550, 824])
        # x_up torch.Size([1, 12, 550, 824])

        
        # x_up *= 0.01  # prevents extreme values at the beginning of the training

        if self.multipredict:
            x_up *= 0.4
            for i, x_up_in in enumerate([x_up3, x_up4, x_up5]):
                x_up += 0.2*self.pred_out[i](x_up_in, (x_up.size(2), x_up.size(3)))

        if self.binary:
            class_pred = nnf.sigmoid(x_up)
        else:
            # not necessary because loss does it
            class_pred = nnf.log_softmax(x_up, dim=1)

        return class_pred,


class ResNet18Dense(_ResNetDense):

    def __init__(self, **kwargs):
        super().__init__(resnet18, **kwargs, small_resnet=True)


class ResNet34Dense(_ResNetDense):

    def __init__(self, **kwargs):
        super().__init__(resnet34, **kwargs,small_resnet=True)


class ResNet50Dense(_ResNetDense):

    def __init__(self, **kwargs):
        super().__init__(resnet50, **kwargs)


class ResNet101Dense(_ResNetDense):

    def __init__(self, **kwargs):
        super().__init__(resnet101, **kwargs)


class ResNet152Dense(_ResNetDense):

    def __init__(self, **kwargs):
        super().__init__(resnet152, **kwargs)

