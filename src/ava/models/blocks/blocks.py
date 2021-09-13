import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as nnf

torch_version = torch.__version__

class _PredictionOutput(nn.Module):

    def __init__(self, input_channels, n_classes, with_softmax=False, upsample_mode='bilinear', dropout=None):
        super().__init__()
        self.upsample_mode = upsample_mode
        self.with_softmax = with_softmax
        self.n_classes = n_classes
        self.input_channels = input_channels

        self.dropout = nn.Dropout2d()

        n_features = int(0.5 * input_channels + 0.5 * n_classes)

        seq = [nn.Conv2d(input_channels, n_features, kernel_size=3, padding=1)]
        if dropout is not None:
            seq += [nn.Dropout2d(p=dropout)]
        seq += [nn.Conv2d(n_features, n_classes, kernel_size=3, padding=1)]

        self.convs = nn.Sequential(*seq)

        # self.convs = nn.Sequential(
        #     nn.Conv2d(input_channels, n_features, kernel_size=3, padding=1),
        #     nn.Conv2d(n_features, n_classes, kernel_size=3, padding=1),
        # )

    def forward(self, x, target_size):
        x_up = nnf.upsample(x, target_size, mode=self.upsample_mode)
        pred = self.convs(x_up)

        if self.with_softmax:
            pred = nnf.softmax(pred, dim=1)

        return pred


class _DecoderBlock(nn.Module):

    def __init__(self, input_channels, output_channels, dropout=None):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(p=dropout) if dropout is not None else None

    def forward(self, x, skip_x):
        x_up = nnf.upsample(x, size=(skip_x.size(2), skip_x.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x_up, skip_x], dim=1)
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)

        x = nnf.relu(x)
        return x



class _Refine(nn.Module):
    def __init__(self, input_channels_1, input_channels_2,i, dropout=None):
        super().__init__()

        k=512
        ki = int(k/2**(i-1))
        knext = int(ki/2) #if i!=4 else ki
        aux = 1024 if i < 4 else 512
        self.conv1 = nn.Conv2d(input_channels_1, aux, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(aux, ki, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ki, knext,kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(input_channels_2, ki, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(ki, knext, kernel_size=3, padding=1)
       #self.up = nn.Upsample(size =(skip_x.size(2), skip_x.size(3)),mode = 'bilinear')
        #self.up = nn.UpsampleBilinear2d(size =())                                                                              


    def forward(self, x, skip_x):

        F = skip_x
        S = self.conv1(F)
        #print(S.shape)
        S = nnf.relu(S)
        S = self.conv2(S)
        #print(S.shape)
        S = nnf.relu(S)
        S = self.conv3(S)
        #print(S.shape)
        M = x
        #print(M.shape)
        if torch_version == '1.6.0':
            M = nnf.interpolate(M,size=(skip_x.size(2), skip_x.size(3)) ,mode='bilinear', align_corners=False)
        else:
            M = nnf.upsample(M, size=(skip_x.size(2), skip_x.size(3)), mode='bilinear', align_corners=False)
        #print(M.shape)      
        M = self.conv4(M)
        #print(M.shape)      
        M = nnf.relu(M)
        M = self.conv5(M)
        #print(M.shape)      

        M = nnf.relu(S + M)
        #print(M.shape)
        #x = nnf.interpolate(M,size=(skip_x.size(2), skip_x.size(3)) ,mode='bilinear', align_corners=False)
        #print(M.shape)
        return M




class _RefinerBlockBasic(nn.Module):

    def __init__(self, input_channels, output_channels, dropout=None):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout2d(p=dropout) if dropout is not None else None

    def forward(self, x, skip_x):
        x = nnf.interpolate(x,size=(skip_x.size(2), skip_x.size(3)) ,mode='bilinear', align_corners=False)
        skip_x = self.conv(skip_x)
        skip_x = nnf.relu(skip_x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv(x)
        x = nnf.relu(x)
        #upsample 2x
        x = nnf.interpolate(x,scale_factor=2 ,mode='bilinear', align_corners=False)

        return x


class _RefinerBlockExtended(nn.Module):

    def __init__(self, input_channels, output_channels, dropout=None):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout2d(p=dropout) if dropout is not None else None

    def forward(self, x, skip_x):
        skip_x = self.conv(skip_x)
        skip_x = nnf.relu(skip_x)
        skip_x = self.conv(skip_x)
        x = self.conv(x)
        x = skip_x + x
        x = nnf.relu(x)
        #upsample 2x
        x = nnf.interpolate(x,scale_factor=2 ,mode='bilinear', align_corners=False)

        return x